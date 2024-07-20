from dataclasses import dataclass


@dataclass
class Config():
    train_bc = 8
    eval_bc = 8
    num_epochs = 120
    data_path = 'mridata/firstMRI/trainImg'
    eval_path = 'mridata/firstMRI/trainImg'
    single_channel = True
    mode = 'double'
    val_length = 2000

    # train process configuration
    val_inter = 3
    save_inter = 5
    sample_size = (256, 256)

    # model parameters
    in_channels = 1
    out_channels = 1
    up_and_down = (128, 256, 512)
    num_res_layers = 2
    vae_path = ''
    dis_path = ''
    autoencoder_warm_up_n_epochs = 100
    
    # mask
    mask_path = 'mask/mask_4.pt'
    # accelerate config
    split_batches = False
    mixed_precision = 'fp16'
    log_with = 'tensorboard'
    project_dir = 'weights/vae/exp_7_20'
    gradient_accumulation_steps = 1
