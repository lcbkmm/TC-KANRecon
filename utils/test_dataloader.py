from torch.utils import data
import glob
from torchvision import transforms
import cv2 as c
from os.path import join
from PIL import Image
import os
from natsort import natsorted
from os.path import dirname as di
import torch
import yaml


def split_bmp_files(bmp_files):
    file_groups = {}
    for file in bmp_files:
        base_name = os.path.basename(file)
        prefix = os.path.splitext(base_name)[0].split('.')[0]  # 获取基础名称前缀
        if prefix not in file_groups:
            file_groups[prefix] = []
        file_groups[prefix].append(file)
    return file_groups


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        for key in config.keys():
            if type(config[key]) == list:
                config[key] = tuple(config[key])
        return config


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
    return glob.glob(up_dir + '*.' + picture_form)


class city_dataset(data.Dataset):
    def __init__(self, updir, image_size):
        super(city_dataset, self).__init__()
        self.img_path = get_address_list(updir, 'jpg')
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

    def convert_to_cuda(self, x, device=None):
        if device is None:
            return x.cuda()
        else:
            return x.to(device)

    def __getitem__(self, index):
        whole_img = self.pic_loader(self.img_path[index])
        width = whole_img.shape[1]
        target, condition = whole_img[:, 0:width // 2], whole_img[:, width // 2:width]
        var_list = [target, condition]

        var_list = map(self.transformer, var_list)
        target, condition = map(self.convert_to_cuda, var_list)

        kwargs = {}
        kwargs['condition'] = condition
        return condition, target

    def __len__(self):
        return len(self.img_path)

    def pic_loader(self, path):
        pic = c.imread(path)
        return c.cvtColor(pic, c.COLOR_BGR2RGB)


class slo_ffa_dataset(data.Dataset):
    def __init__(self, up_dir, img_size):
        super(slo_ffa_dataset, self).__init__()
        fu_path = join(up_dir, "Images/")
        self.an_path = join(up_dir, "Masks/")
        self.fu_path = get_address_list(fu_path, "png")
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        # Now we define the transforms in the dataset

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0], img_size[1])),
            transforms.Normalize(mean=0.5, std=0.5)])

        self.transformer_mini = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0] // 2, img_size[1] // 2)),
            transforms.Normalize(mean=0.5, std=0.5)])

    def __getitem__(self, index):
        fun_filename = self.fu_path[index]
        middle_filename = fun_filename.split("/")[-1].split(".")[0]
        first_num, second_num = int(middle_filename.split("_")[0]), int(middle_filename.split("_")[1])

        XReal_A, XReal_A_half = self.convert_to_resize(self.funloader(fun_filename))
        an_filename = str(first_num) + "_mask_" + str(second_num) + ".png"
        an_file_path = self.an_path + an_filename
        XReal_B, XReal_B_half = self.convert_to_resize(self.funloader(an_file_path))

        XReal_B, XReal_A = map(self.convert_to_cuda, [XReal_B, XReal_A])
        target = XReal_B
        kwargs = {}
        kwargs['condition'] = XReal_A
        condition = XReal_A
        return target, condition

    def convert_to_resize(self, X):
        y1 = self.transformer(X)
        y2 = self.transformer_mini(X)
        return y1, y2

    def convert_to_cuda(self, x, device=None):
        if device is None:
            return x.cuda()
        else:
            return x.to(device)

    def __len__(self):
        return len(self.fu_path)

    def funloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def angloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class multi_slo_ffa_dataset(data.Dataset):
    def __init__(self, fu_path, img_size, noise_level=0):
        super(multi_slo_ffa_dataset, self).__init__()
        # we will get the tensor from multi directory

        self.fu_path = fu_path
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0], img_size[1])),
            transforms.RandomAffine(degrees=2 * noise_level, translate=[0.04 * noise_level, 0.04 * noise_level],
                                    scale=[1 - 0.04 * noise_level, 1 + 0.04 * noise_level], fill=-1),
            transforms.Normalize(mean=0.5, std=0.5)])

        self.transformer_mini = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0] // 2, img_size[1] // 2)),
            transforms.Normalize(mean=0.5, std=0.5)])

    def __getitem__(self, index):
        fun_filename = self.fu_path[index]
        super_parrent_dir = di(di(fun_filename))
        middle_filename = fun_filename.split("/")[-1].split(".")[0]
        first_num, second_num = int(middle_filename.split("_")[0]), int(middle_filename.split("_")[1])
        an_filename = str(first_num) + "_mask_" + str(second_num) + ".png"
        XReal_A, XReal_A_half = self.convert_to_resize(self.funloader(fun_filename))
        an_file_path = join(super_parrent_dir, 'Masks', an_filename)
        XReal_B, XReal_B_half = self.convert_to_resize(self.funloader(an_file_path))
        return XReal_B, XReal_A

    def convert_to_resize(self, X):
        y1 = self.transformer(X)
        y2 = self.transformer_mini(X)
        return y1, y2

    def __len__(self):
        return len(self.fu_path)

    def funloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def angloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class Double_dataset(data.Dataset):
    def __init__(self, data_path, img_size, mode='double', read_channel='color', data_aug=True, time_table_path='', ):
        '''
        data_path: the up dir of data
        img_size: what size of image you want to read (tuple, int)
        mode: vary from: 1. 'double' 2. 'first' 3. 'second' 
        read_channel: 'color' or 'gray'
        table_path: time information
        '''
        super(Double_dataset, self).__init__()
        self.img_path = get_image_address(data_path)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        basic_trans_list = [
            transforms.ToTensor(),
            # transforms.Resize((512, 640), antialias=True),
            transforms.Normalize(mean=0.5, std=0.5)]

        self.data_aug = data_aug
        bmp_files = get_address_list(data_path, picture_form='bmp')
        group_dict = split_bmp_files(natsorted(bmp_files))
        for _, group in group_dict.items():
            natsorted(group)
        self.whole_list = list(group_dict.keys())
        if data_aug:
            self.augmentator = transforms.Compose([
                # transforms.RandomRotation(30), 
                transforms.CenterCrop(img_size),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            basic_trans_list.append(transforms.Resize(img_size, antialias=True))
        self.transformer = transforms.Compose(basic_trans_list)
        self.mode = mode
        if read_channel == 'color':
            self.img_reader = self.colorloader
        else:
            self.img_reader = self.grayloader
        self.time_table = load_config(time_table_path)
        self.group_dict = group_dict

    def double_get(self, slo_path, key) -> list:
        parrent_dir = di(slo_path)
        slo_file = os.path.basename(slo_path)
        num, suffix = os.path.splitext(slo_file)
        first_path = os.path.join(parrent_dir, f'{num}.1{suffix}')
        second_path = os.path.join(parrent_dir, f'{num}.2{suffix}')
        var_list = map(self.img_reader, [slo_path, first_path, second_path])
        var_list = map(self.transformer, var_list)
        if self.data_aug:
            var_list = torch.cat(list(var_list))
            slo, ffa_first, ffa_second = torch.chunk(self.augmentator(var_list), 3)
        else:
            slo, ffa_first, ffa_second = var_list

        slo_and_ffa = self.group_dict[key]
        whole_ffa = slo_and_ffa[:-1]
        ffa_fir = whole_ffa[0]
        ffa_sec = whole_ffa[1]
        fir_t = self.time_table[os.path.basename(ffa_fir)]
        sec_t = self.time_table[os.path.basename(ffa_sec)]
        time = sec_t - fir_t
        return (slo, ffa_first, ffa_second), (slo_file, time)

    def first_get(self, slo_path) -> list:
        parrent_dir = di(slo_path)
        slo_file = os.path.basename(slo_path)
        num, suffix = os.path.splitext(slo_file)
        first_path = os.path.join(parrent_dir, f'{num}.1{suffix}')
        var_list = map(self.img_reader, [slo_path, first_path])
        var_list = map(self.transformer, var_list)
        if self.data_aug:
            var_list = torch.cat(list(var_list))
            slo, ffa_first = torch.chunk(self.augmentator(var_list), 3)
        else:
            slo, ffa_first = var_list
        return (slo, ffa_first), slo_file

    def second_get(self, slo_path) -> list:
        parrent_dir = di(slo_path)
        slo_file = os.path.basename(slo_path)
        num, suffix = os.path.splitext(slo_file)
        second_path = os.path.join(parrent_dir, f'{num}.2{suffix}')
        var_list = map(self.img_reader, [slo_path, second_path])
        var_list = map(self.transformer, var_list)
        if self.data_aug:
            var_list = torch.cat(list(var_list))
            slo, ffa_second = torch.chunk(self.augmentator(var_list), 3)
        else:
            slo, ffa_second = var_list
        return (slo, ffa_second), slo_file

    def __getitem__(self, index) -> list:
        key = self.whole_list[index]
        slo_name = self.img_path[index]
        var_list, info = None, None
        if self.mode == 'double':
            var_list, info = self.double_get(slo_name, key)
        elif self.mode == 'first':
            var_list, info = self.first_get(slo_name)
        elif self.mode == 'second':
            var_list, info = self.second_get(slo_name)
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


def form_dataloader(updir, image_size, batch_size, shuffle=True):
    dataset = city_dataset(updir, image_size)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=True)


def VT_form_dataloader(updir, image_size, batch_size, shuffle=True):
    dataset = slo_ffa_dataset(updir, image_size)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=True)


def double_form_dataloader(updir, image_size, batch_size, mode,
                           read_channel='color', data_aug=True, shuffle=True, drop_last=True, time_table_path=None):
    dataset = Double_dataset(updir, image_size, mode, read_channel, data_aug, time_table_path)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)


def split_dataloader(up_dir, img_size, val_length, train_bc, eval_bc, to_shuffle):
    import random
    if not isinstance(up_dir, list) and not isinstance(up_dir, tuple):
        up_dir = [up_dir]
    fu_path = []
    for i in range(len(up_dir)):
        fu_path += get_address_list(join(up_dir[i], "Images/"), "png")
    random.shuffle(fu_path)

    if val_length > 0:
        val_path = fu_path[:val_length]
        train_path = fu_path[val_length:]
        val_dataset = multi_slo_ffa_dataset(val_path, img_size)
        train_dataset = multi_slo_ffa_dataset(train_path, img_size)
        train_dataloader = data.DataLoader(train_dataset, train_bc, shuffle=to_shuffle, drop_last=True)
        val_dataloader = data.DataLoader(val_dataset, eval_bc, shuffle=to_shuffle, drop_last=True)
        return train_dataloader, val_dataloader
    else:
        train_dataset = multi_slo_ffa_dataset(fu_path, img_size)
        train_dataloader = data.DataLoader(train_dataset, train_bc, shuffle=to_shuffle, drop_last=True)
        return train_dataloader, None


if __name__ == "__main__":
    test_dataloader = double_form_dataloader('E:\\Eye\\Diffusion\\diffusion-series\\dataset\\color\\train', (512, 512),
                                             1, 'double', data_aug=True,
                                             time_table_path='E:\\Eye\\Diffusion\\diffusion-series\\dataset\\time.yaml')
    for batch in test_dataloader:
        batch, filename, time = batch[0], batch[1][0], batch[1][1]
        slo, first_ffa, second_ffa = batch[0], batch[1], batch[2]
        print("Shape of slo: ", slo.shape)
        print("name:", filename)
        print("time:", time)

