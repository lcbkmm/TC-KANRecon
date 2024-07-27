

from __future__ import annotations
from torch.nn import init
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.utils import ensure_tuple_rep
from torch import nn
from unet.MF_UKAN import TimeEmbedding, ResBlock, DownSample, OverlapPatchEmbed, shiftedBlock
from generative.networks.nets.diffusion_model_unet import get_down_block, get_mid_block, get_timestep_embedding
from config.diffusion.config_controlnet import Config

class ControlNetConditioningEmbedding(nn.Module):
    """
    用于将控制条件编码到潜在空间中的网络。
    """

    def __init__(
            self, spatial_dims: int, in_channels: int, out_channels: int,
            num_channels: Sequence[int] = (16, 32, 96, 256)
    ):
        super().__init__()

        # 输入卷积层，用于初始特征提取
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.blocks = nn.ModuleList([])

        # 构建一系列卷积块
        for i in range(len(num_channels) - 2):
            channel_in = num_channels[i]
            channel_out = num_channels[i + 1]

            self.blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channel_in,
                    out_channels=channel_in,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            )
            self.blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channel_in,
                    out_channels=channel_out,
                    strides=2,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            )

        self.conv_out = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=num_channels[-2],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def copy_weights_to_controlnet(controlnet: nn.Module, diffusion_model: nn.Module, verbose: bool = True) -> None:
    """
    Args:
        controlnet: ControlNet 实例
        diffusion_model: DiffusionModelUnet 或 SPADEDiffusionModelUnet 实例
        verbose: 如果为 True，将打印匹配和不匹配的键。
    """

    output = controlnet.load_state_dict(diffusion_model.state_dict(), strict=False)
    if verbose:
        dm_keys = [p[0] for p in list(diffusion_model.named_parameters()) if p[0] not in output.unexpected_keys]
        print(
            f"Copied weights from {len(dm_keys)} keys of the diffusion model into the ControlNet:"
            f"\n{'; '.join(dm_keys)}\nControlNet missing keys: {len(output.missing_keys)}:"
            f"\n{'; '.join(output.missing_keys)}\nDiffusion model incompatible keys: {len(output.unexpected_keys)}:"
            f"\n{'; '.join(output.unexpected_keys)}"
        )


class MC_MODEL(nn.Module):
    """
    Args:
        spatial_dims: 空间维度数（例如，2 表示 2D，3 表示 3D）。
        in_channels: 输入通道数。
        num_res_blocks: 每个阶段的残差块数量。
        num_channels: 各个阶段的输出通道数。
        attention_levels: 列表，表示在哪些阶段添加注意力机制。
        norm_num_groups: 归一化时的组数。
        norm_eps: 归一化的 epsilon 值。
        resblock_updown: 如果为 True，则使用残差块进行上下采样。
        num_head_channels: 每个注意力头的通道数。
        with_conditioning: 如果为 True，则添加空间变换器以进行条件处理。
        transformer_num_layers: Transformer 块的层数。
        cross_attention_dim: 使用的上下文维度数。
        num_class_embeds: 如果指定（作为整数），则该模型将使用类条件生成。
        upcast_attention: 如果为 True，则将注意力操作提升到全精度。
        use_flash_attention: 如果为 True，则使用内存效率高的闪存注意力机制。
        conditioning_embedding_in_channels: 条件嵌入的输入通道数。
        conditioning_embedding_num_channels: 条件嵌入的通道数。
    """

    def __init__(
            self, T, ch, ch_mult, attn, num_res_blocks,
            dropout,spatial_dims = 2,
            num_zero_res_blocks = (2, 2, 2, 2),
            mlp_ratios = (1, 1)
    ):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'

        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        attn = []  # 注意力机制的层索引，初始化为空列表
        self.head = nn.Conv2d(4, ch, kernel_size=3, stride=1, padding=1)  # 初始卷积层
        self.downblocks = nn.ModuleList()  # 下采样块列表
        self.controlnet_down_blocks = nn.ModuleList()  # controlnet下采样块列表
        self.controlnet_kan_blocks = nn.ModuleList() # kan_block块列表
        chs = [ch]  # 记录下采样过程中使用的通道数
        now_ch = ch

        controlnet_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=now_ch,
            out_channels=now_ch,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        # 初始化参数为0
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)
        # 构建下采样块
        self.zero_res_chs = [now_ch]
        for i, mult in enumerate(ch_mult):
            h_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, h_ch=h_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = h_ch
                chs.append(now_ch)
                controlnet_block = Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=now_ch,
                        out_channels=now_ch,
                        strides=1,
                        kernel_size=1,
                        padding=0,
                        conv_only=True,
                    )
                # 初始化参数为0
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)
                controlnet_block = Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=now_ch,
                        out_channels=now_ch,
                        strides=1,
                        kernel_size=1,
                        padding=0,
                        conv_only=True,
                    )
                # 初始化参数为0
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)
                
            self.chs = chs

        # 额外的特征提取层
        embed_dims = [256, 320, 512]
        norm_layer = nn.LayerNorm
        dpr = [0.0, 0.0, 0.0]
        self.patch_embed3 = OverlapPatchEmbed(img_size=64 // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=64 // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])

        self.kan_block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=mlp_ratios[0], drop_path=dpr[0], norm_layer=norm_layer)])

        self.controlnet_kan_blocks.append(zero_module(Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=embed_dims[1] * mlp_ratios[0],
                    out_channels=embed_dims[1] * mlp_ratios[0],
                    strides=1,
                    kernel_size=1,
                    padding=0,
                    conv_only=True,
                )))

        self.kan_block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], mlp_ratio=mlp_ratios[1], drop_path=dpr[1], norm_layer=norm_layer)])

        self.controlnet_kan_blocks.append(zero_module(Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=embed_dims[2] * mlp_ratios[1],
                    out_channels=embed_dims[2] * mlp_ratios[1],
                    strides=1,
                    kernel_size=1,
                    padding=0,
                    conv_only=True,
                )))

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            spatial_dims=2,
            in_channels=1,
            num_channels=(64, 128, 192, 256),
            out_channels=64,
        )

        self.initialize()  # 初始化权重

    def initialize(self):
        # 使用 Xavier 均匀分布初始化卷积层的权重
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)

    def forward(self, x, t, controlnet_cond, conditioning_scale=1.0):
        # 时间嵌入
        t_emb = self.time_embedding(t)
        temb = t_emb.to(dtype=x.dtype)

        # 初始卷积
        h = self.head(x)
        # 引入条件
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        h += controlnet_cond

        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        # 额外的特征提取块
        B = x.shape[0]

        # h -- [B, H*W, embed_dim]
        h, H, W = self.patch_embed3(h)
        for i, blk in enumerate(self.kan_block1):
            h = blk(h, H, W, temb)
        h = self.norm3(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        kan_hs = [h]

        h, H, W = self.patch_embed4(h)
        for i, blk in enumerate(self.kan_block2):
            h = blk(h, H, W, temb)
        h = self.norm4(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        kan_hs.append(h)

        # 6. Control net blocks
        controlnet_down_block_res_samples = ()
        for h, controlnet_block in zip(hs, self.controlnet_down_blocks):
            h = controlnet_block(h)
            controlnet_down_block_res_samples += (h,)

        for kan_h, controlnet_kan_block in zip(kan_hs, self.controlnet_kan_blocks):
            kan_h = controlnet_kan_block(kan_h)
            controlnet_down_block_res_samples += (kan_h,)

        down_block_res_samples = controlnet_down_block_res_samples

        # 6. 缩放
        down_block_res_samples = [h * conditioning_scale for h in down_block_res_samples]
        # mid_block_res_sample *= conditioning_scale
        mid_block_res_sample = down_block_res_samples[-1]

        return down_block_res_samples, mid_block_res_sample
