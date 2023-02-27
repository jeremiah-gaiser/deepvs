import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATv2Conv

class GCN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim, edge_feature_dim,  heads=2):
        super().__init__()
        self.linear1 = torch.nn.Linear(feature_dim, hidden_dim)
        
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, dropout=0.25, add_self_loops=False, edge_dim=edge_feature_dim)  
        self.conv2 = GATv2Conv(hidden_dim*heads, hidden_dim, heads=heads, dropout=0.25, add_self_loops=False, edge_dim=edge_feature_dim)  
        self.conv3 = GATv2Conv(hidden_dim*heads, hidden_dim, heads=heads, dropout=0.25, add_self_loops=False, edge_dim=edge_feature_dim)  

        self.batchnorm = torch_geometric.nn.norm.BatchNorm(hidden_dim*heads)
        self.linear2 = torch.nn.Linear(hidden_dim*heads, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.linear1(x)

        h = self.conv1(x, edge_index, edge_attr) 
        h = self.batchnorm(h)
        h = F.silu(h)

        h = self.conv2(h, edge_index, edge_attr) 
        h = self.batchnorm(h)
        h = F.silu(h)

        h = self.conv3(h, edge_index, edge_attr) 
        h = F.silu(h)
        
        o = self.linear2(h)
        return h,o