import torch
from torch import nn
import torchvision.transforms as T
from torchvision import datasets, models, transforms
import os
from PIL import Image

for img in os.listdir('./pictures/test'):
    img = Image.open(os.path.join('./pictures/test', img))
    img = T.functional.crop(img, 600,850,450,450)
    img = T.ToTensor()(img)

    model = models.resnet18(pretrained=False)
    nr_filters = model.fc.in_features
    model.fc = nn.Linear(nr_filters, 1)

    model.load_state_dict(torch.load('weights/model.ckpt.9'))
    model.eval()
    print(torch.sigmoid(model(img.unsqueeze(0))))

