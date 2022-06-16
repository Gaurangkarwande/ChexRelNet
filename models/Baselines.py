import torch
from torch import nn
from models.Encoder import Encoder

class SiameseModel(nn.Module):
    def __init__(self, config):
        super(SiameseModel, self).__init__()
        self.encoder = Encoder(config)
        self.fc_net = nn.Sequential(nn.Linear(in_features=config['gcn_in']*2, out_features=config['siam_fc_net_hidden']), \
                        nn.LeakyReLU(), nn.Dropout(config['cnn_dropout']), \
                        nn.Linear(in_features=config['siam_fc_net_hidden'], out_features=config['n_classes']))
        self.dropout = nn.Dropout(config['cnn_dropout'], inplace=True)


    def forward(self, prev_image, cur_image):
        batch_size = prev_image.shape[0]
        global_encoding = self.encoder(torch.cat((prev_image, cur_image), dim=0))
        output = torch.cat((global_encoding[:batch_size], global_encoding[batch_size:]), dim=1)
        output = self.fc_net(output)
        del global_encoding
        return output
    
