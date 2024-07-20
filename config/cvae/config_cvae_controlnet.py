from dataclasses import dataclass


@dataclass
class Config():
    train_bc = 8
    eval_bc = 8
    num_epochs = 1000
    data_path = '/home/chenyifei/data/CYFData/FZJ/308_1.06/color/train'
    eval_path = '/home/chenyifei/data/CYFData/FZJ/308_1.06/color/test'
    single_channel = False
    mode = 'double'
    val_length = 2000

    # train process configuration
    val_inter = 80
    save_inter = 100
    sample_size = 512

    # vqgan parameters
    in_channels = 1
    out_channels = 1
    up_and_down = (128, 256, 512)
    num_res_layers = 2
    num_embeddings = 512
    input_path = '/home/chenyifei/data/CYFData/FZJ/weights_of_models/weights_dif_series/exp_2_26/input_save'
    output_path = '/home/chenyifei/data/CYFData/FZJ/weights_of_models/weights_dif_series/exp_2_26/output_save'


    # stable_model parameters
    sd_num_channels = (128, 256, 512, 1024)
    attention_levels = (False, True, True, True)
    sd_resume_path = 'weights/exp_5_11/model_save/model.pth'
    controlnet_path = 'weights/exp_5_6/model_save/model.pth'
    
    # controlnet_model parameters
    

    
    autoencoder_warm_up_n_epochs = 0
    # accelerate config
    split_batches = False
    mixed_precision = 'fp16'
    log_with = 'tensorboard'
    project_dir = 'weights/exp_6_10'
    gradient_accumulation_steps = 1
