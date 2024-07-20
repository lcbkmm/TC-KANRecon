from dataclasses import dataclass


@dataclass
class Config():
    train_bc = 4
    eval_bc = 4
    num_epochs = 2000
    old = False
    data_path = ['/home/chenyifei/data/CYFData/FZJ/387_06_10/train', '/home/chenyifei/data/CYFData/FZJ/387_06_10/test']
    eval_path = '/home/chenyifei/data/CYFData/FZJ/387_06_10/test'
    single_channel = False
    mode = 'double'
    val_length = 2000

    # train process configuration
    val_inter = 80
    save_inter = 100
    sample_size = (512, 640)
    vae_path = '/data/publicData/CYFData/FZJ/weights_of_models/weights_dif_series/exp_6_8/model_save'


    # stable_model parameters
    sd_num_channels = (128, 256, 512, 1024)
    attention_levels = (False, True, True, True)
    sd_resume_path = 'weights/exp_6_15/model_save/model.pth'
    controlnet_path = 'weights/exp_6_15/model_save/model.pth'
    
    # controlnet_model parameters

    
    # accelerate config
    split_batches = False
    mixed_precision = 'fp16'
    log_with = 'tensorboard'
    project_dir = 'weights/exp_6_16'
    gradient_accumulation_steps = 1
