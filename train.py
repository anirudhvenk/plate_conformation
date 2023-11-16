"""
Trains ResNet18 on a set of cropped plate images.

Usage:
    train.py --train_dir=<train_dir> --weights_dir=<weights_dir>
"""
import torch
from torchvision import models, datasets, transforms
from torch import nn
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from docopt import docopt
import os
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    args = docopt(__doc__)
    train_dir = args['--train_dir']
    weights_dir = args['--weights_dir']

    train_transforms = transforms.Compose([
        transforms.Resize(size=(232,232)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

    trainloader = DataLoader(train_data, shuffle=True, batch_size=20)
    testloader = DataLoader(test_data, shuffle=True, batch_size=20)

    model = models.resnet18(pretrained=True)
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

    best_accuracy = 0
    for epoch in range(100):
        epoch_loss = 0
        for img, y in tqdm(trainloader):
            optimizer.zero_grad()
            y_hat = model(img.to('cuda'))
            loss = loss_fn(y_hat, y.unsqueeze(1).float().cuda())
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss/len(trainloader)
        
        print("Training loss: ", epoch_loss.item())
        
        with torch.no_grad():
            cum_loss = 0
            y_true = []
            y_pred = []
            for img, y in tqdm(testloader):
                y_hat = model(img.to('cuda'))
                loss = loss_fn(y_hat, y.unsqueeze(1).float().cuda())
                cum_loss += loss/len(testloader)
                
                y_true.extend(torch.round(torch.sigmoid(y_hat)).cpu().numpy())
                y_pred.extend(y.numpy())
        
            accuracy = accuracy_score(y_pred, y_true)
            print("Validation loss: ", cum_loss.item())
            print("Validation accuracy: ", accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(weights_dir, 'model.best'))
            
        torch.save(model.state_dict(), os.path.join(weights_dir, f'model.ckpt.{epoch}'))
