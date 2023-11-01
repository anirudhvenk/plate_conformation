import torch
from torch import nn
import torchvision.transforms as T
from torchvision import datasets, models, transforms
import os
from PIL import Image
from model import Net

model = models.resnet50(pretrained=True)
nr_filters = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(nr_filters, 1),
)
model.load_state_dict(torch.load('weights/model.ckpt.7'))
model.eval()
model.to('cuda')


for img in os.listdir('./pictures/cropped_test/backward'):
    img = Image.open(os.path.join('./pictures/cropped_test/backward', img))
    img = T.Resize(size=(224,224))(img)
    img = T.ToTensor()(img).unsqueeze(0)
    print(torch.sigmoid(model(img.to('cuda'))))
    