hidden = 512 
INTERACTION_TYPES = protein_config['interaction_labels']
NODE_DIMS = 38 
EDGE_DIMS = 9
DUMMY_INDEX = protein_config['atom_labels'].index('DUMMY')
MAX_EDGE_WEIGHT = 15.286330223083496

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(NODE_DIMS, hidden)
        
        self.conv1 = GCN2Conv(hidden, 0.2, add_self_loops=False)
        self.conv2 = GCN2Conv(hidden, 0.2, add_self_loops=False)
        self.conv3 = GCN2Conv(hidden, 0.2, add_self_loops=False)

        self.linear2 = torch.nn.Linear(hidden, len(INTERACTION_TYPES))

    def forward(self, data):
        x, edge_index, edge_weights = data.x[:,:-3], data.edge_index, data.edge_attr[:,-1] / MAX_EDGE_WEIGHT

        x = self.linear1(x)

        h = self.conv1(x, x, edge_index, edge_weights)
        h = F.relu(h)

        h = self.conv2(h, x, edge_index, edge_weights)
        h = F.relu(h)

        h = self.conv3(h, x, edge_index, edge_weights)
        h = F.relu(h)
        
        o = self.linear2(h)
        return h

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN().to(device)
model.load_state_dict(torch.load(pocket_model_file, map_location=torch.device('cpu')))