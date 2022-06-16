import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphConv

#input is batch x num_nodes x node_features
#num_nodes = 2*num_regions = 2*18
#node_features = 1000
#output is batch x output features

class GCN(torch.nn.Module):
    def __init__(self, regions, config):
        super().__init__()
        self.in_head = 5
        self.out_head = 3
        self.dropout = config['gcn_dropout']
        self.regions = regions
        
        self.conv1 = GATConv(config['gcn_in'], config['gcn_hidden'], heads=self.in_head, dropout=self.dropout)
        self.conv2 = GATConv(config['gcn_hidden']*self.in_head, config['gcn_out'], heads=self.out_head, dropout=self.dropout, concat=False)


    def forward(self, x, edge_index, bbox):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = x[bbox]
        return x


class GCN_shallow(torch.nn.Module):
    def __init__(self, regions, config):
        super().__init__()
        self.in_head = 5
        self.out_head = 3
        self.dropout = config['gcn_dropout']
        self.regions = regions
        
        self.conv1 = GATConv(config['gcn_in'], config['gcn_out'], heads=7, dropout=self.dropout, concat=False)


    def forward(self, x, edge_index, bbox):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x[bbox]
        return x


class GCN_noAtt(torch.nn.Module):
    def __init__(self, regions, config):
        super().__init__()
        self.dropout = config['gcn_dropout']
        self.regions = regions
        
        self.conv1 = GraphConv(config['gcn_in'], config['gcn_hidden'])
        self.conv2 = GraphConv(config['gcn_hidden'], config['gcn_out'])


    def forward(self, x, edge_index, bbox):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = x[bbox]
        
        return x

class GCN_deep(torch.nn.Module):
    def __init__(self, regions, config):
        super().__init__()
        self.in_head = 5
        self.out_head = 3
        self.dropout = config['gcn_dropout']
        self.regions = regions
        
        self.conv1 = GATConv(config['gcn_in'], config['gcn_hidden'], heads=self.in_head, dropout=self.dropout)
        self.conv2 = GATConv(config['gcn_hidden']*self.in_head, config['gcn_hidden'], heads=self.in_head, dropout=self.dropout)
        self.conv3 = GATConv(config['gcn_hidden']*self.in_head, config['gcn_out'], heads=self.out_head, dropout=self.dropout, concat=False)


    def forward(self, x, edge_index, bbox):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = x[bbox]
        return x