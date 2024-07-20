from dataclasses import dataclass


@dataclass
class Config():
    train_bc = 16
    eval_bc = 4
    num_epochs = 80
    data_path = ['/mntcephfs/lab_data/wangcm/fzj/advanced_VT/dataset/data_sloeaffa', 
                '/mntcephfs/lab_data/wangcm/fzj/advanced_VT/dataset/data_slolaffa', 
                '/mntcephfs/lab_data/wangcm/fzj/advanced_VT/dataset/early_koufu_ea', 
                '/mntcephfs/lab_data/wangcm/fzj/advanced_VT/dataset/early_koufu_la']
    eval_path = '/mntcephfs/lab_data/wangcm/fzj/adv-Diff/dataset/308_12_26/grey/train'
    single_channel = False
    mode = 'double'
    val_length = 2000

    # train process configuration
    val_inter = 2
    save_inter = 80
    sample_size = 512

    # vqgan parameters
    in_channels = 1
    out_channels = 1
    up_and_down = (128, 256, 512)
    num_res_layers = 2
    num_embeddings = 512
    vae_path = '/mntcephfs/lab_data/wangcm/fzj/diffusion-series/weights/exp_11_22/model_save'

    # model parameters
    sd_num_channels = (320, 640, 1280, 1280)
    attention_levels = (False, True, True, True)

    
    # accelerate config
    split_batches = False
    mixed_precision = 'fp16'
    log_with = 'tensorboard'
    project_dir = 'weights/exp_5_12'
    gradient_accumulation_steps = 1
