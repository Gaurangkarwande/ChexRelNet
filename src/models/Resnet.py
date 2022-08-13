from torch import nn
import torch.nn.functional as F
import torchxrayvision as xrv

#Input is Batch x image_ch x image_h x image_w
#output is batch x out_features

class RegionalCNN(nn.Module):
    def __init__(self, resnet_out_features: int, out_features: int, freeze: bool, dropout: float):
        super().__init__()
        self.resnet = xrv.autoencoders.ResNetAE(weights="101-elastic")
        if freeze: self.freeze_layers()
        self.resnet_outdim = resnet_out_features         #if input size = 224x224, then 512*3*3, input size = 128x128 then 512*2*2
        self.dropout = dropout
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.resnet_outdim, out_features)
        self.batch_norm = nn.BatchNorm2d(512)
    
    def freeze_layers(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)["z"]
        x = self.batch_norm(x)
        x = x.view(-1, self.resnet_outdim)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        x = x.relu()
        return x