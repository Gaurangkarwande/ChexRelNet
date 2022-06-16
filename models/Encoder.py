import torch
from torch import nn
from models.Resnet import RegionalCNN 

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.regional_cnn = RegionalCNN(resnet_out_features=config['resnet_out'], out_features=config['gcn_in'], freeze=config['cnn_freeze'], dropout=config['cnn_dropout'])

    def forward(self, x):        #x is num_graphs*num_nodes x 1 x 224 x 224      num_graphs ~ batch_size
        x = self.regional_cnn(x)
        return x


        
        


