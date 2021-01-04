import torchvision

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, feat_dim = 2048, dim_output=18):
        super(Model, self).__init__()

        self.dim_output = dim_output
        self.feat_dim = feat_dim
        
        self.conv1 = nn.Conv2d(feat_dim, feat_dim, kernel_size=1)
        self.fc1 = nn.Linear(feat_dim, feat_dim//4) # 2048 -> 512
        self.fc2 =  nn.Linear(feat_dim//4, feat_dim//8)
        self.fc3 =  nn.Linear(feat_dim//8, dim_output)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        
        
        self.backbone = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
        # # Fix Initial Layers
        for p in list(self.backbone.children())[:-5]:
            p.requires_grad = False
        # # get the structure until the last FC layer
        modules = list(self.backbone.children())[:-1]
        
        self.backbone = nn.Sequential(*modules)
    
    def forward(self, img):
        batch_size = img.shape[0]
        x = self.backbone(img)
        x = self.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc1(x.view(batch_size, -1)))
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
