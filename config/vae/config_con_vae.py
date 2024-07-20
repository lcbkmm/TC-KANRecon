from dataclasses import dataclass


@dataclass
class Config():
    train_bc = 16
    eval_bc = 16
    num_epochs = 200
    data_path = ['/mntcephfs/lab_data/wangcm/fzj/advanced_VT/dataset/data_sloeaffa', 
                '/mntcephfs/lab_data/wangcm/fzj/advanced_VT/dataset/data_slolaffa', 
                '/mntcephfs/lab_data/wangcm/fzj/advanced_VT/dataset/early_koufu_ea', 
                '/mntcephfs/lab_data/wangcm/fzj/advanced_VT/dataset/early_koufu_la']
    eval_path = '/mntcephfs/lab_data/wangcm/fzj/adv-Diff/dataset/308_12_26/grey/train'
    single_channel = True
    mode = 'double'
    val_length = 2000

    # train process configuration
    val_inter = 2
    save_inter = 10
    sample_size = 512

    # model parameters
    in_channels = 1
    out_channels = 1
    condition_channels=3
    up_and_down = (128, 256, 512)
    num_res_layers = 2
    num_embeddings = 512
    
    out_vae_path = 'weights/exp_5_3/gen_save/vqgan.pth'
    in_vae_path = 'weights/exp_5_2/gen_save/vqgan.pth'

    autoencoder_warm_up_n_epochs = 0
    # accelerate config
    split_batches = False
    mixed_precision = 'fp16'
    log_with = 'tensorboard'
    project_dir = 'weights/exp_5_9'
    gradient_accumulation_steps = 1
