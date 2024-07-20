from __future__ import print_function, division
import torchvision.transforms as transforms
from torch.utils import data
from torchvision.datasets import ImageFolder

    
def split_dataloader(up_dir, img_size, val_length, train_bc, eval_bc, to_shuffle=True):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size[0], img_size[1]), antialias=True), 
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    all_dataset = ImageFolder(root=up_dir, transform=train_transform)
    if val_length > 0:
        data_len = len(all_dataset) - val_length
        print(f'Dataset length: {data_len} and the validation length: {val_length}')
        train_dataset, val_dataset = data.random_split(all_dataset, [data_len, val_length])
        train_dataloader = data.DataLoader(train_dataset, train_bc, to_shuffle, drop_last=True)
        val_dataloader = data.DataLoader(val_dataset, eval_bc, True, drop_last=True)
        return train_dataloader, val_dataloader
    else:
        train_dataloader = data.DataLoader(all_dataset, train_bc, to_shuffle, drop_last=True)
        return train_dataloader, 0
    
if __name__ == "__main__":
    train_loader, _ = split_dataloader('/home/fanzhaojiehd/cdpm/data/MedNIST', 
                                256, 
                                100, 
                                8, 
                                4)
    iterable = iter(train_loader)
    for i in range(10):
        test = next(iterable)
        print(test[1])