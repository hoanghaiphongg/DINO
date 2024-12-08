import numpy as np
from sklearn.model_selection import train_test_split

import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, SubsetRandomSampler

from .augmentation import DataAugmentation


def get_dataset(global_img_size=224, 
                local_img_size=96, 
                global_crops_scale=(0.4, 1.0), 
                local_crops_scale=(0.05, 0.4), 
                local_crops_number=8, 
                batch_size=32):

    train_transforms_ = DataAugmentation(
        global_img_size=global_img_size,
        local_img_size=local_img_size,
        global_crops_scale=global_crops_scale,
        local_crops_scale=local_crops_scale,
        local_crops_number=local_crops_number,
    )

    basic_transforms_ = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_data = dsets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transforms_
    )

    train_index, valid_index = train_test_split(
        np.arange(len(train_data)),
        test_size=0.2,
        shuffle=True,
        stratify=train_data.targets,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_index),
        drop_last=True,
    )

    valid_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(valid_index),
        drop_last=True,
    )

    test_data = dsets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=basic_transforms_
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return {
        'train': train_loader,
        'number_of_train': len(train_index),
        'valid': valid_loader,
        'number_of_valid': len(valid_index),
        'test': test_loader,
        'number_of_test': len(test_data),
    }