#dataset.py
import numpy as np
from sklearn.model_selection import train_test_split

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder

from augmentation import DataAugmentation



def get_dataset(global_img_size=224, 
                local_img_size=96, 
                global_crops_scale=(0.4, 1.0), 
                local_crops_scale=(0.05, 0.4), 
                local_crops_number=8, 
                batch_size=32,
                data_dir='/kaggle/input/stable-imagenet1k/imagenet1k'):  # Thêm tham số cho đường dẫn dữ liệu

    # Định nghĩa các phép biến đổi cho tập huấn luyện
    train_transforms_ = DataAugmentation(
        global_img_size=global_img_size,
        local_img_size=local_img_size,
        global_crops_scale=global_crops_scale,
        local_crops_scale=local_crops_scale,
        local_crops_number=local_crops_number,
    )

    # Định nghĩa các phép biến đổi cho tập kiểm tra và xác thực (validation)
    basic_transforms_ = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Sử dụng ImageFolder để tải dữ liệu
    dataset = ImageFolder(root=data_dir, transform=train_transforms_)

    # Chia bộ dữ liệu thành train, validation, test
    train_size = int(0.8 * len(dataset))  # 80% dữ liệu cho train
    valid_size = int(0.1 * len(dataset))  # 10% dữ liệu cho validation
    test_size = len(dataset) - train_size - valid_size  # phần còn lại cho test

    # Chia bộ dữ liệu thành các tập con train, validation, test
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # Tạo DataLoader cho từng tập dữ liệu
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Trả về các DataLoader và thông tin về số lượng hình ảnh trong mỗi tập
    return {
        'train': train_loader,
        'number_of_train': len(train_dataset),
        'valid': valid_loader,
        'number_of_valid': len(valid_dataset),
        'test': test_loader,
        'number_of_test': len(test_dataset),
    }
