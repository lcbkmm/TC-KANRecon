
from torch.utils.tensorboard import SummaryWriter
import torch as th
import os
class Visualizer:
    def __init__(self, weights_up_dir, image_size, model, diffusion):
        weights_up_dir = self.check_dir(weights_up_dir)
        self.recorder = SummaryWriter(weights_up_dir)
        self.model = model
        self.diffusion = diffusion
        self.image_title = 'Default'
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
            
    def check_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
    
    def inverse_transform(self, tensor):
        return (tensor + 1) / 2
        
    def performance_display(self, val_iter, step):
        batch, cond = next(val_iter)
        sample = self.diffusion.p_sample_loop(
            self.model,
            (1, 3, self.image_size[0], self.image_size[1]),
            clip_denoised=True,
            model_kwargs=cond,
            device=batch.device
        )
        sample = th.cat([cond['condition'], sample, batch], dim=0)
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        self.recorder.add_images('p_result', sample.cpu().detach(), step)
        self.recorder.flush()
        
    def display_single(self, tensor, step, normalize=True):
        '''
        tensor range: (-1, 1) if normalize=True else (0, 1)
        step: milestone you want to show in tensorboard
        '''
        if normalize:
            tensor = self.inverse_transform(tensor)
        self.recorder.add_images(self.image_title, tensor.cpu().detach(), step)
        self.recorder.flush()
        
    def tb_draw_scalars(self, value, which_epoch):
        self.recorder.add_scalar('Rescale_loss', value, which_epoch)
        self.recorder.flush()