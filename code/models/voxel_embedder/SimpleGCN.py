import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCN2Conv

class GCN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim, edge_feature_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(feature_dim, hidden_dim)
        
        self.conv1 = GCN2Conv(hidden_dim, 0.25)
        self.conv2 = GCN2Conv(hidden_dim, 0.25)
        self.conv3 = GCN2Conv(hidden_dim, 0.25)

        self.dropout = torch.nn.Dropout(p=0.25)
        self.batchnorm = torch_geometric.nn.norm.BatchNorm(hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr[:,-1] 

        x = self.linear1(x)

        h = self.conv1(x, x, edge_index, edge_weights)
        h = self.batchnorm(h)
        h = self.dropout(F.relu(h))

        h = self.conv2(h, x, edge_index, edge_weights)
        h = self.batchnorm(h)
        h = self.dropout(F.relu(h))

        h = self.conv3(h, x, edge_index, edge_weights)
        h = self.batchnorm(h)
        h = self.dropout(F.relu(h))
        
        o = self.linear2(h)
        return h,o