import torch
from torchvision import models
from torch import nn
from data import ImageFolder
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

training_data = ImageFolder('./pictures/train')
testing_data = ImageFolder('./pictures/test2')
train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=1, shuffle=True)

model = models.resnet18(pretrained=False)
nr_filters = model.fc.in_features
model.fc = nn.Linear(nr_filters, 1)
model.to(torch.device('cuda'))

loss_fn = BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.fc.parameters())

for epoch in range(10):
    epoch_loss = 0
    for img, y in tqdm(train_dataloader):
        optimizer.zero_grad()
        y_hat = model(img)
        loss = loss_fn(y_hat, y.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss/len(train_dataloader)
    
    print(epoch_loss.item())
    
    with torch.no_grad():
        cum_loss = 0
        for img, y in tqdm(test_dataloader):
            y_hat = model(img)
            loss = loss_fn(y_hat, y.unsqueeze(1).float())
            cum_loss += loss/len(test_dataloader)
        
        print(cum_loss.item())
        
    torch.save(model.state_dict(), f'weights/model.ckpt.{epoch}')
