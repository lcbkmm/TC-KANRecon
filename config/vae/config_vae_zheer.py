from dataclasses import dataclass


@dataclass
class Config():
    train_bc = 1
    eval_bc = 8
    num_epochs = 2000
    mode = 'double'
    data_path = '/home/chenyifei/data/CYFData/FZJ/308_1.06/color/train'
    eval_path = '/home/chenyifei/data/CYFData/FZJ/308_1.06/color/test'
    single_channel = True
    mode = 'double'
    val_length = 2000

    # train process configuration
    val_inter = 120
    save_inter = 200
    sample_size = 512

    # model parameters
    in_channels = 1
    out_channels = 1
    condition_channels=3
    up_and_down = (128, 256, 512)
    num_res_layers = 2
    num_embeddings = 512
    
    resume_path = '/home/chenyifei/data/CYFData/FZJ/mri-autoencoder-v0.1'

    autoencoder_warm_up_n_epochs = 0
    # accelerate config
    split_batches = False
    mixed_precision = 'fp16'
    log_with = 'tensorboard'
    project_dir = 'weights/exp_6_5'
    gradient_accumulation_steps = 1
