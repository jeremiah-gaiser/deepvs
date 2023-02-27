import pickle
import re
import numpy as np
import sys
import os
import glob
import torch
from torch import nn
import torch_geometric
import random
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from code.utils.get_path import get_path
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

from code.models.voxel_embedder.SimpleGCN import GCN
# from code.models.voxel_embedder.AttentiveGCN import GCN
# from code.models.voxel_embedder.GAT import GCN

def add_self_loops(x, edge_index, edge_attr, edge_labels, voxel_label_index):
    node_count = x.size(0)
    
    new_edge_index = torch.hstack((edge_index, torch.vstack([torch.arange(node_count)]*2)))
    new_edge_attr = torch.hstack((edge_attr, torch.zeros(node_count)))
    
    self_edge_labels = torch.tensor([[0,1,0,0,0],[0,0,0,1,0]])
    new_edge_labels = torch.vstack((edge_labels, 
                                    torch.index_select(self_edge_labels, 0, x[:, voxel_label_index].int())))
    
    return new_edge_index, new_edge_attr, new_edge_labels

def train_voxel_embedder(config: dict, model_path: str, weights_out: str, holdout_list: list, resolution: float=1.0, load_weights: str=None) -> None:
    ATOM_LABELS = config['constants']['point_cloud']['HEAVY_ATOM_LABELS'] 
    EDGE_LABELS = config['constants']['point_cloud']['EDGE_LABELS']
    INTERACTION_LABELS = config['constants']['point_cloud']['INTERACTION_LABELS'] 
    CLASS_COUNT = config['constants']['point_cloud']['INTERACTION_COUNTS']
    class_weights = torch.tensor([max(CLASS_COUNT)/x for x in CLASS_COUNT])

    print("Saving weights in %s" % weights_out, flush=True)

    voxel_label_index = ATOM_LABELS.index('VOXEL')

    training_samples_dir = get_path(config, 'training_samples_dir') % resolution
    sample_corpus = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE: ", device, flush=True)

    f_i = 0

    for f in glob.glob(training_samples_dir + "*.pkl"):
        f_i += 1

        # if f_i > 256:
        #     continue

        pdb_id = f.split('/')[-1].split('_')[0]

        if pdb_id in holdout_list:
            print(pdb_id, flush=True)
            continue

        if f_i % 1800 == 0:
            print('Loading samples: %s%%' % int((f_i / 18000)*100), flush=True)

        for graph in pickle.load(open(f, 'rb')):
            graph.edge_index, graph.edge_attr, graph.edge_labels = add_self_loops(graph.x, 
                                                                                  graph.edge_index, 
                                                                                  graph.edge_attr,
                                                                                  graph.edge_labels,
                                                                                  voxel_label_index)
            graph.edge_attr /= 14
            graph.edge_attr = torch.hstack((graph.edge_labels, graph.edge_attr.unsqueeze(1)))
            graph.beta_factor /= 100
            graph.x = torch.hstack((graph.x, graph.beta_factor.unsqueeze(1)))
            node_features_dim = graph.x.size(1)
            edge_features_dim = graph.edge_attr.size(1)
            sample_corpus.append(graph)
 
    sample_loader = DataLoader(sample_corpus, batch_size=32, shuffle=True)
    del sample_corpus

    model = GCN(node_features_dim, 512, len(INTERACTION_LABELS), edge_features_dim).to(device)
    # model = GCN(node_feature_dim, edge_feature_dim, 256, len(INTERACTION_LABELS)).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    sigmoid = torch.nn.Sigmoid()

    batch_idx = 0
    min_loss = 999

    for epoch in range(1000):
        print("EPOCH %s" % epoch, flush=True)
        epoch_loss = []
        reporter_loss = []
        
        for batch in sample_loader:
            optimizer.zero_grad()
            positive_mask = torch.where(torch.sum(batch.y,dim=1) > 0)[0]
            batch = batch.to(device)

            _, out = model(batch)

            out = out[positive_mask]
            y = batch.y[positive_mask].float()

            loss = criterion(out, y)
            epoch_loss.append(loss.item())
            reporter_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            batch_idx += 1

            if batch_idx % 50 == 0:
                reporter_avg_loss = sum(reporter_loss) / len(reporter_loss)
                reporter_loss = []
                print("Average loss: %s" % reporter_avg_loss, flush=True)
                print(' '.join(["%.2f" % x.item() for x in sigmoid(out[0])]), flush=True)
                print(' '.join([str(int(x.item())) for x in y[0]]), flush=True)

        epoch_avg_loss = sum(epoch_loss) / len(epoch_loss)
        print("Epoch loss: %s" % epoch_avg_loss, flush=True)

        if epoch_avg_loss < min_loss:
            min_loss = epoch_avg_loss
            torch.save(model.state_dict(), weights_out) 



