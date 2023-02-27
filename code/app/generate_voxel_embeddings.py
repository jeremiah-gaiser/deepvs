from importlib import import_module
import sys
from code.utils.get_path import get_path
from code.utils.get_distance import get_distance
from copy import deepcopy
import numpy as np
from torch_geometric.utils import subgraph 
from torch_geometric.data import Data
from code.utils.EdgeData import EdgeData
import torch
import pickle
import os

def generate_voxel_embeddings(config: dict, id_batch: list, gcn_model: str, trained_weights: str, resolution: float=1.0, neighbor_count: int=10, 
                              protein_file: str=None, voxel_file: str=None, out_file: str=None, out_dir: str=None) -> None:
    
    embed_ft = get_path(config, 'voxel_embed_ft')
    voxel_ft = get_path(config, 'voxel_ft')
    protein_ft = get_path(config, 'protein_graph_ft')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = None

    GCN = getattr(import_module(config['models']['voxel_embedder'].replace("/", ".") % gcn_model), 'GCN')

    if out_dir is None:
        embed_dir = get_path(config, 'voxel_embed_dir') % resolution
    else:
        if out_dir[-1] != '/':
            out_dir += '/'

        embed_dir = out_dir
        embed_ft = out_dir + embed_ft.split('/')[-1]

    if os.path.exists(embed_dir)==False:
        os.makedirs(embed_dir)

    ATOM_LABELS = config['constants']['point_cloud']['HEAVY_ATOM_LABELS']
    EDGE_LABELS = config['constants']['point_cloud']['EDGE_LABELS']
    INTERACTION_LABELS = config['constants']['point_cloud']['INTERACTION_LABELS']
    voxel_label_index = ATOM_LABELS.index('VOXEL')

    batch_total = len(id_batch)

    for pdb_i, pdb_id in enumerate(id_batch): 
        if config['args']['skip']:
            if os.path.exists(embed_ft % pdb_id):
                continue

        print("%s: %s of %s" % (pdb_id, pdb_i+1, batch_total))

        voxel_file = voxel_ft % (resolution, pdb_id)
        protein_file = protein_ft % pdb_id

        if False in map(os.path.exists, [voxel_file, protein_file]):
            continue

        graph_x = []
        graph_pos = []
        graph_edge_index = [[], []]

        voxel_graph = pickle.load(open(voxel_file, 'rb'))  
        # Data(y=[N, 9], pos=[N, 3], contact_map=[N])
        protein_graph = pickle.load(open(protein_file, 'rb'))  
        # Data(x=[N, 38], pos=[N, 3], b_factor=[N])

        voxel_label_vector = torch.zeros(protein_graph.x.size(1))
        voxel_label_vector[voxel_label_index] = 1

        # for v_i in filtered_voxels:
        for voxel_pos in voxel_graph.pos:
            edge_data = EdgeData(EDGE_LABELS)

            voxel_x = deepcopy(voxel_label_vector)
            voxel_b_factor = torch.zeros(1)

            atom_voxel_distances, atom_indices = torch.sort(torch.cdist(voxel_pos.unsqueeze(0), protein_graph.pos, p=2))
            
            atom_voxel_distances = atom_voxel_distances[0][:neighbor_count]
            atom_indices = atom_indices[0][:neighbor_count]

            atom_x = protein_graph.x[atom_indices]
            atom_pos = protein_graph.pos[atom_indices]
            atom_b_factor = protein_graph.b_factor[atom_indices]

            for atom_i in range(neighbor_count): 
                av_distance = atom_voxel_distances[atom_i].item()
                edge_data.add_edge(0, atom_i+1, av_distance, 'atom-voxel')

                if atom_i == neighbor_count-1:
                    break

                atom_atom_distances = torch.cdist(atom_pos[atom_i].unsqueeze(0), atom_pos[atom_i+1:], p=2)[0]

                for atom_j, d in enumerate(atom_atom_distances):
                    edge_data.add_edge(atom_i+1, atom_i+atom_j+2, d.item(), 'atom-atom')

            g_edge_index, g_edge_attr, g_edge_labels = edge_data.get_data()

            g_pos = torch.vstack((voxel_pos, atom_pos)).float()
            g_x = torch.vstack((voxel_x, atom_x)).float()
            g_b_factor = torch.hstack((voxel_b_factor, atom_b_factor)).float()

            graph = Data(x=g_x, 
                         pos=g_pos, 
                         beta_factor=g_b_factor,
                         edge_index=torch.tensor(g_edge_index, dtype=torch.long),
                         edge_attr=torch.tensor(g_edge_attr, dtype=torch.float),
                         edge_labels=torch.tensor(g_edge_labels, dtype=torch.long))

            graph.edge_attr /= 14
            graph.edge_attr = torch.hstack((graph.edge_labels, graph.edge_attr.unsqueeze(1)))
            graph.beta_factor /= 100
            graph.x = torch.hstack((graph.x, graph.beta_factor.unsqueeze(1)))
            node_features_dim = graph.x.size(1)
            edge_features_dim = graph.edge_attr.size(1)

            if model is None:
                model = GCN(node_features_dim, 512, len(INTERACTION_LABELS), edge_features_dim)
                model.load_state_dict(torch.load(trained_weights, map_location=device))
                model.eval()
                torch.no_grad()

            voxel_mask = torch.where(graph.x[:, voxel_label_index] == 1)[0]
            node_embed, _ = model(graph)
            node_embed = node_embed[voxel_mask]
            node_pos = graph.pos[voxel_mask]

            graph_x.append(node_embed)
            graph_pos.append(node_pos)

        graph_x = torch.vstack(graph_x)
        graph_pos = torch.vstack(graph_pos)
        graph_edge_data = EdgeData(['adj', 'self'])

        for n_i in range(graph_pos.size(0)):
            vv_distances = torch.cdist(graph_pos[n_i].unsqueeze(0), graph_pos[n_i:])[0]

            for n_idx in torch.where(vv_distances <= resolution)[0]:
                n_j = n_i + n_idx.item()
                edge_label = 'adj'

                if n_i == n_j:
                    label = 'self'

                graph_edge_data.add_edge(n_i, n_j, vv_distances[n_idx].item(), edge_label)

        graph_edge_index, graph_edge_attr, graph_edge_labels = graph_edge_data.get_data()

        voxel_graph = Data(x=graph_x, 
                     pos=graph_pos, 
                     edge_index=torch.tensor(graph_edge_index, dtype=torch.long),
                     edge_attr=torch.tensor(graph_edge_attr, dtype=torch.float),
                     edge_labels=torch.tensor(graph_edge_labels, dtype=torch.long))

        pickle.dump(voxel_graph, open(embed_ft % pdb_id, 'wb'))








