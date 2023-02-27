from torch_geometric.nn import GCN2Conv

class GCN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(node_dims, hidden)
        
        self.conv1 = GCN2Conv(hidden, 0.2, add_self_loops=False)
        self.conv2 = GCN2Conv(hidden, 0.2, add_self_loops=False)
        self.conv3 = GCN2Conv(hidden, 0.2, add_self_loops=False)

        self.linear2 = torch.nn.Linear(hidden, len(INTERACTION_LABELS))

    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr 

        x = self.linear1(x)

        h = self.conv1(x, x, edge_index, edge_weights)
        h = F.relu(h)

        h = self.conv2(h, x, edge_index, edge_weights)
        h = F.relu(h)

        h = self.conv3(h, x, edge_index, edge_weights)
        h = F.relu(h)
        
        o = self.linear2(h)
        return h,o