import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image
import torchvision.transforms as T
import os


class ImageFolder(Dataset):

    def __init__(self, root):
        imgs = [os.path.join(root, img_path) for img_path in os.listdir(root)]

        self.root = root
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path)
        # img = T.functional.crop(img, 600,850,450,450)
        img = T.functional.crop(img, 100,1200,1900,1400)
        img = T.ToTensor()(img)
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        
        if os.path.basename(path)[0] == '1':
            return img, 1
        else:
            return img, 0

