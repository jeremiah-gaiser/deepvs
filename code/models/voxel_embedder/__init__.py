import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch_geometric.nn import AttentiveFP
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool

class AtomicAttentiveFP(AttentiveFP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atom_classifier = nn.Linear(kwargs['hidden_channels'], 
                                         kwargs['out_channels'])
        
    def forward(self, x, edge_index, edge_attr, batch):
        """"""
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.atom_grus[0](h, x).relu_()

        for conv, gru in zip(self.atom_convs[1:], self.atom_grus[1:]):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.atom_classifier(x), self.lin2(out)

class GCN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim, edge_dim):
        super().__init__()

        self.conv1 = AtomicAttentiveFP(in_channels=feature_dim, 
                                 hidden_channels=hidden_dim, 
                                 out_channels=out_dim,
                                 edge_dim=edge_dim,
                                 num_layers=6,
                                 num_timesteps=6,
                                 dropout=0.25)
    
    def forward(self, data):
        x, edge_index, edge_weights, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.conv1(x, edge_index, edge_weights, batch)
        return h