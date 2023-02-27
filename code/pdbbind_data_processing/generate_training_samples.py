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

def generate_training_samples(config: dict, id_batch: list, resolution: float=1.0, neighbor_count: int=10) -> None:
    training_samples_dir = get_path(config, 'training_samples_dir') % resolution
    training_sample_ft = get_path(config, 'training_samples_ft')
    voxel_ft = get_path(config, 'voxel_ft')
    protein_ft = get_path(config, 'protein_graph_ft')

    if os.path.exists(training_samples_dir)==False:
        os.makedirs(training_samples_dir)

    voxel_dir = get_path(config, 'voxel_dir') 
    protein_dir = get_path(config, 'protein_graph_dir')

    ATOM_LABELS = config['constants']['point_cloud']['HEAVY_ATOM_LABELS']
    EDGE_LABELS = config['constants']['point_cloud']['EDGE_LABELS']
    voxel_label_index = ATOM_LABELS.index('VOXEL')

    batch_total = len(id_batch)

    if os.path.exists(training_samples_dir)==False:
        os.makedirs(training_samples_dir)

    for pdb_i, pdb_id in enumerate(id_batch): 
        print("%s: %s of %s" % (pdb_id, pdb_i+1, batch_total))

        voxel_file = voxel_ft % (resolution, pdb_id)
        protein_file = protein_ft % pdb_id

        if False in map(os.path.exists, [voxel_file, protein_file]):
            continue

        training_sample_file = training_sample_ft % (resolution, pdb_id)
        graph_list = []

        voxel_graph = pickle.load(open(voxel_file, 'rb'))  
        # Data(y=[N, 9], pos=[N, 3], contact_map=[N])
        protein_graph = pickle.load(open(protein_file, 'rb'))  
        # Data(x=[N, 38], pos=[N, 3], b_factor=[N])

        voxel_label_vector = torch.zeros(protein_graph.x.size(1))
        voxel_label_vector[voxel_label_index] = 1

        atom_y_vector = torch.zeros(voxel_graph.y.size(1))

        # occupied_voxels  = torch.where(voxel_graph.contact_map != -1)[0]
        interacting_voxels  = torch.where(torch.sum(voxel_graph.y, dim=1) > 0)[0]
        # filtered_voxels = torch.sort(torch.cat((occupied_voxels, interacting_voxels)))[0].unique()

        # for v_i in filtered_voxels:
        for v_i in interacting_voxels:
            edge_data = EdgeData(EDGE_LABELS)

            voxel_y = voxel_graph.y[v_i]
            voxel_pos = voxel_graph.pos[v_i]
            voxel_contact = voxel_graph.contact_map[v_i]
            voxel_x = deepcopy(voxel_label_vector)
            voxel_b_factor = torch.zeros(1)

            atom_voxel_distances, atom_indices = torch.sort(torch.cdist(voxel_pos.unsqueeze(0), protein_graph.pos, p=2))
            
            atom_voxel_distances = atom_voxel_distances[0][:neighbor_count]
            atom_indices = atom_indices[0][:neighbor_count]

            atom_x = protein_graph.x[atom_indices]
            atom_pos = protein_graph.pos[atom_indices]
            atom_b_factor = protein_graph.b_factor[atom_indices]

            atom_y = atom_y_vector.repeat(neighbor_count, 1)
            atom_contact = torch.tensor([-1]*neighbor_count)

            for atom_i in range(neighbor_count): 
                av_distance = atom_voxel_distances[atom_i].item()
                edge_data.add_edge(0, atom_i+1, av_distance, 'atom-voxel')

                if atom_i == neighbor_count-1:
                    break

                atom_atom_distances = torch.cdist(atom_pos[atom_i].unsqueeze(0), atom_pos[atom_i+1:], p=2)[0]

                for atom_j, d in enumerate(atom_atom_distances):
                    edge_data.add_edge(atom_i+1, atom_i+atom_j+2, d.item(), 'atom-atom')

            g_edge_index, g_edge_attr, g_edge_labels = edge_data.get_data()

            g_y = torch.vstack((voxel_y, atom_y)).long()
            g_pos = torch.vstack((voxel_pos, atom_pos)).float()
            g_contact = torch.hstack((voxel_contact, atom_contact)).long()
            g_x = torch.vstack((voxel_x, atom_x)).float()
            g_b_factor = torch.hstack((voxel_b_factor, atom_b_factor)).float()

            sample_graph = Data(x=g_x,
                                y=g_y,
                                pos=g_pos, 
                                contact_map=g_contact,
                                beta_factor=g_b_factor,
                                edge_index=torch.tensor(g_edge_index, dtype=torch.long),
                                edge_attr=torch.tensor(g_edge_attr, dtype=torch.float),
                                edge_labels=torch.tensor(g_edge_labels, dtype=torch.long))
            graph_list.append(sample_graph)

        pickle.dump(graph_list, open(training_sample_file, 'wb'))
