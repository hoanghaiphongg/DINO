import random
from PIL import Image, ImageFilter, ImageOps

import torchvision.transforms as transforms


class GaussianBlur(object):
    
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.p
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class DataAugmentation(object):
    
    def __init__(
        self,
        global_img_size=224,
        local_img_size=96,
        global_crops_scale=(0.4, 1.),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8,
    ):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global view
        self.global_view1 = transforms.Compose([
            transforms.RandomResizedCrop(global_img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=1.),
            normalize,
        ])

        # second global view
        self.global_view2 = transforms.Compose([
            transforms.RandomResizedCrop(global_img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            normalize,
        ])

        # local view
        self.local_crops_number = local_crops_number
        self.local_view = transforms.Compose([
            transforms.RandomResizedCrop(local_img_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, img):
        augmentation = []
        augmentation.append(self.global_view1(img))
        augmentation.append(self.global_view2(img))
        for _ in range(self.local_crops_number):
            augmentation.append(self.local_view(img))
        return augmentation