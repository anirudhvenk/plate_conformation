import torch
from torchvision import models, datasets, transforms
from torch import nn
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from model import Net

train_transforms = transforms.Compose([
    transforms.Resize(size=(232,232)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
test_transforms = transforms.Compose([
    transforms.Resize(size=(232,232)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_data = datasets.ImageFolder('./pictures/cropped_train',transform=train_transforms)
test_data = datasets.ImageFolder('./pictures/cropped_test',transform=test_transforms)

trainloader = DataLoader(train_data, shuffle=True, batch_size=3)
testloader = DataLoader(test_data, shuffle=True, batch_size=3)

model = models.resnet50(pretrained=True)
for params in model.parameters():
  params.requires_grad_ = False

nr_filters = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(nr_filters, 1),
)
model.to('cuda')

loss_fn = BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0005)

for epoch in range(50):
    epoch_loss = 0
    for img, y in tqdm(trainloader):
        optimizer.zero_grad()
        y_hat = model(img.to('cuda'))
        loss = loss_fn(y_hat, y.unsqueeze(1).float().cuda())
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss/len(trainloader)
    
    print(epoch_loss.item())
    
    with torch.no_grad():
        cum_loss = 0
        for img, y in tqdm(testloader):
            y_hat = model(img.to('cuda'))
            loss = loss_fn(y_hat, y.unsqueeze(1).float().cuda())
            print(torch.sigmoid(y_hat))
            print(y.unsqueeze(1))
            cum_loss += loss/len(testloader)
        
        print(cum_loss.item())
        
    torch.save(model.state_dict(), f'weights/model.ckpt.{epoch}')
