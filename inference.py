"""
Outputs an inference (forward or backward plate) for a given image.

Usage:
    inference.py --img_path=<img_path> --weights_path=<weights_path>
"""
import torch
from torch import nn
from torchvision import models, datasets, transforms
from torchvision import transforms
from PIL import Image
from docopt import docopt

if __name__ == '__main__':
    args = docopt(__doc__)
    img_path = args['--img_path']
    weights_path = args['--weights_path']

    model = models.resnet18(pretrained=True)
    for params in model.parameters():
        params.requires_grad_ = False

    nr_filters = model.fc.in_features
    model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(nr_filters, 1),
    )
    model.to('cuda')

    model.load_state_dict(torch.load(weights_path))
    model.eval()

    test_transforms = transforms.Compose([
    transforms.Resize(size=(232,232)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ])

    raw_img = Image.open(img_path)
    cropped = raw_img.crop((1300,1700,1600,2000))
    img = test_transforms(cropped).unsqueeze(0).to('cuda')

    with torch.no_grad():
        y_hat = model(img)
        prediction = torch.round(torch.sigmoid(y_hat)).cpu().numpy()

    print("Prediction: ", prediction)
