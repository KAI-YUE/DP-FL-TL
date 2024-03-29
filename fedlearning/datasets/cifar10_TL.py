import os
import pickle
import numpy as np

from torchvision import datasets
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torch import tensor, long

def CIFAR10_224(data_path):
    channel = 3
    im_size = (224, 224)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    train_transform = T.Compose([T.Resize(224), T.RandomHorizontalFlip(), T.RandomCrop(size=224, padding=20), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_transform = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    
    dst_train = MyCIFAR10(data_path, train=True, download=True, transform=train_transform)
    dst_test = datasets.CIFAR10(data_path, train=False, download=False, transform=test_transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    
    properties = {
        "channel": channel,
        "im_size": im_size,
        "num_classes": num_classes,
        "n_train": 50000,
        "class_names": class_names,
        "dst_train": dst_train,
        "dst_test": dst_test,
        "test_transform": test_transform,
    }
    
    class dataset_properties: pass
    for key, value in properties.items():
        setattr(dataset_properties, key, value)

    return dataset_properties


class MyCIFAR10(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.cifar10 = datasets.CIFAR10(file_path, train=train, download=download, transform=transform)
        self.targets = self.cifar10.targets
        self.classes = self.cifar10.classes

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        # return data, target, index + int(256*self.clone_ver)
        return (data, target)

    def __len__(self):
        return len(self.cifar10)

    def update_clone_ver(self, cur_ver):
        self.clone_ver = cur_ver