import numpy as np
import yaml
import inspect
from shutil import copyfile, copy
import os
import torch
from torchvision.utils import make_grid

def one_to_three(x):
    return torch.cat([x]*3, dim=1)


def save_config_to_yaml(config_obj, parrent_dir: str):
    """
    Saves the given Config object as a YAML file. The output file name is derived
    from the module name where the Config class is defined.

    Args:
        config_obj (Config): The Config object to be saved.
        module_name (str): Name of the Python module containing the Config class definition.
    """
    os.makedirs(parrent_dir, exist_ok=True)
    # Get the source file path for the specified module
    module_path = inspect.getfile(config_obj)
    
    # Extract the base file name without extension
    base_filename = os.path.splitext(os.path.basename(module_path))[0]

    # Construct the output YAML file name
    output_file = f"{base_filename}.yaml"
    output_file = os.path.join(parrent_dir, output_file)

    # Convert the Config object to a dictionary
    config_dict = config_obj.__dict__
    config_dict = {k: v for k, v in config_obj.__dict__.items() if not k.startswith('__')}
    # Save the dictionary as a YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump(config_dict, yaml_file, sort_keys=False)

    return config_dict


def copy_yaml_to_folder(yaml_file, folder):
    """
    将一个 YAML 文件复制到一个文件夹中
    :param yaml_file: YAML 文件的路径
    :param folder: 目标文件夹路径
    """
    # 确保目标文件夹存在
    os.makedirs(folder, exist_ok=True)

    # 获取 YAML 文件的文件名
    file_name = os.path.basename(yaml_file)

    # 将 YAML 文件复制到目标文件夹中
    copy(yaml_file, os.path.join(folder, file_name))

def force_remove_empty_dir(path):
    try:
        os.rmdir(path)
        print(f"Directory '{path}' removed successfully.")
    except FileNotFoundError:
        print(f"Directory '{path}' not found.")
    except OSError as e:
        print(f"Error removing directory '{path}': {e}")

def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        for key in config.keys():
            if type(config[key]) == list:
                config[key] = tuple(config[key])
        return config
    
def get_parameters(fn, original_dict):
    new_dict = dict()
    arg_names = inspect.getfullargspec(fn)[0]
    for k in original_dict.keys():
        if k in arg_names:
            new_dict[k] = original_dict[k]
    return new_dict

def write_config(config_path, save_path):
    copyfile(config_path, save_path)
    
    
def check_dir(dire):
    if not os.path.exists(dire):
        os.makedirs(dire)
    return dire

def combine_tensors_2_tb(tensor_list:list=None):
    image = torch.cat(tensor_list, dim=-1)
    image = (make_grid(image, nrow=1).unsqueeze(0)+1)/2
    return image.clamp(0, 1)

