import numpy as np
from torch.utils import data
import glob
from torchvision import transforms
import cv2
from os.path import join
from PIL import Image
import os
from natsort import natsorted
from os.path import dirname as di
import torch
import sys;sys.path.append('./')
from utils.common import load_config
import pickle

def get_image_address(updir):
    files = natsorted(os.listdir(updir))
    slo_list = []
    for file_name in files:
        # 构建旧路径
        old_path = os.path.join(updir, file_name)

        # 获取文件名
        base_name, picture_form = os.path.splitext(file_name)

        if os.path.isfile(old_path):
            if len(base_name.split('.')) == 1:
                new_name = f'{base_name}{picture_form}'
                slo_list.append(os.path.join(updir, new_name))
    return slo_list

def get_address_list(up_dir, picture_form: str):
    if up_dir[-1] != '/':
        up_dir = f'{up_dir}/'
    return glob.glob(up_dir+'*.'+picture_form)



class Double_dataset(data.Dataset):
    def __init__(self, data_path, img_size, 
                 mode='double', read_channel='color', mask=None,
                 data_aug=True):
        '''
        data_path: the up dir of data
        img_size: what size of image you want to read (tuple, int)
        mode: vary from: 1. 'double' 2. 'first' 3. 'second' 
        read_channel: 'color' or 'gray' 
        '''
        super(Double_dataset, self).__init__()
        if isinstance(data_path, list):
            self.img_path = []
            for path in data_path:
                self.img_path += get_image_address(path)
        else:
            self.img_path = get_image_address(data_path)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        
        basic_trans_list = [
            transforms.ToTensor(),
            # transforms.Resize((512, 640), antialias=True),
            ]
        
        self.data_aug = data_aug
        if data_aug:
            self.augmentator = transforms.Compose([
                # transforms.RandomRotation(5), 
                transforms.Resize((256, 256)), 
                # transforms.RandomVerticalFlip(),
                # transforms.RandomHorizontalFlip(), 
                transforms.Normalize(mean=0.5, std=0.5)
            ])
        else: 
            basic_trans_list.append(transforms.Resize(img_size, antialias=True))
            basic_trans_list.append(transforms.Normalize(mean=0.5, std=0.5))
        self.transformer = transforms.Compose(basic_trans_list) 
        self.mode = mode
        if read_channel == 'color':
            self.img_reader = self.colorloader
        else:
            self.img_reader = self.grayloader

        self.slo_reader = self.grayloader
        self.mask = mask
    
    def double_get(self, pt_path, mask) -> list:
        f_kspace = pickle.load(open(pt_path, 'rb'))['img']
        
        f_kspace1 = np.fft.fft2(f_kspace)
        f_kspace1 = f_kspace1 * mask
        mri_mask = np.fft.ifft2(f_kspace1)
        
        mri = np.abs(f_kspace)
        mri_mask = np.abs(mri_mask)
        
        mri_min = np.min(mri)
        mri_max = np.max(mri)
        mri = 2 * (mri - mri_min) / (mri_max - mri_min) - 1
        
        mri_mask_min = np.min(mri_mask)
        mri_mask_max = np.max(mri_mask)
        mri_mask = 2 * (mri_mask - mri_mask_min) / (mri_mask_max - mri_mask_min) - 1
        
        
        mri_mask = self.transformer(mri_mask)
        mri = self.transformer(mri)
        
        if self.data_aug:
            mri = self.augmentator(mri)
            mri_mask = self.augmentator(mri_mask)
        else:
            mri = mri
            mri_mask = mri_mask
        return mri, mri_mask
    
    def __getitem__(self, index)->list:
        slo_name = self.img_path[index]
        var_list, info = None, None
        if self.mode == 'double':
            var_list, info = self.double_get(slo_name, mask=self.mask)
        
        return var_list, info
    
    def __len__(self):
        return len(self.img_path)
    
    def colorloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def grayloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
   
def double_form_dataloader(updir, image_size, batch_size, mode, 
                           read_channel='color', mask = None, data_aug=True, 
                           shuffle=True, drop_last=True):
    dataset = Double_dataset(updir, image_size, mode, read_channel, mask, data_aug)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)