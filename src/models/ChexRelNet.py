import torch
from torch import nn
from models.Encoder import Encoder
from models.GraphModels import GCN

class ChexRelNet(nn.Module):
    def __init__(self, edge_index, config, regions):
        super().__init__()
        self.encoder = Encoder(config, regions)
        self.global_fc_net = nn.Sequential(nn.Linear(in_features=config['gcn_in'], out_features=config['gcn_out']))
        self.gcn = GCN(regions, config)
        self.fc_net = nn.Sequential(nn.Linear(in_features=config['gcn_out']*3, out_features=config['fc_net_hidden']), \
                        nn.LeakyReLU(), nn.Dropout(config['cnn_dropout']), \
                        nn.Linear(in_features=config['fc_net_hidden'], out_features=config['n_classes']))       #*3
        self.dropout = nn.Dropout(config['cnn_dropout'], inplace=True)
        self.edge_index = edge_index
        self.num_regions = 2 * len(regions)


    def forward(self, x, prev_image, cur_image, num_graphs, bbox, device): 
        batch_size = prev_image.shape[0]
        bbox = bbox + torch.arange(0, num_graphs*self.num_regions, self.num_regions).to(device) #to extract relevant nodes from num_graph*num_node sized batch
        edge_index = self.edge_index.to(device)
        cnn_encoding = self.encoder(x)                          #num_graphs*num_nodes x gcn_in
        graph_encoding = self.gcn(cnn_encoding, edge_index, bbox)    #batch x gcn_out
        del cnn_encoding; del x
        global_encoding = self.encoder(torch.cat((prev_image, cur_image), dim=0))
        global_encoding = self.global_fc_net(global_encoding)
        del prev_image; del cur_image
        output = torch.cat((graph_encoding, global_encoding[:batch_size], global_encoding[batch_size:]), dim=1)     #graph_encoding
        del global_encoding
        output = self.fc_net(output)                                #batch x 1
        del graph_encoding 
        return output