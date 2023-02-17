from code.utils.get_path import get_path
import numpy as np
from torch_geometric.utils import subgraph 
from torch_geometric.data import Data
import torch
import pickle
import os

def get_adjacent_nodes(edge_index, node_indices):
    adjacent_node_indices = torch.tensor([])

    for node_index in node_indices:
        incident_edge_indices = torch.where(edge_index[0] == node_index)[0]
        adjacent_node_indices = torch.hstack((adjacent_node_indices, edge_index[1][incident_edge_indices]))

    return torch.unique(adjacent_node_indices).long()

def filter_positive_samples(config: dict, id_batch: list, resolution: float=1.0) -> None:
    positive_samples_dir = get_path(config, 'positive_samples_dir') % resolution
    positive_sample_ft = get_path(config, 'positive_samples_ft')
    full_graph_ft = get_path(config, 'point_cloud_ft')

    ATOM_LABELS = config['constants']['point_cloud']['HEAVY_ATOM_LABELS']
    EDGE_LABELS = config['constants']['point_cloud']['EDGE_LABELS']
    voxel_label_index = ATOM_LABELS.index('VOXEL')
    voxel_voxel_edge_label_index = EDGE_LABELS.index('voxel-voxel')

    batch_total = len(id_batch)

    if os.path.exists(positive_samples_dir)==False:
        os.makedirs(positive_samples_dir)

    for pdb_i, pdb_id in enumerate(id_batch): 
        print("%s: %s of %s" % (pdb_id, pdb_i, batch_total))
        positive_sample_file = positive_sample_ft % (resolution, pdb_id)
        full_graph_file = full_graph_ft % (resolution, pdb_id)
        
        if os.path.exists(full_graph_file)==False:
            continue

        full_graph = pickle.load(open(full_graph_file, 'rb')) 
        last_voxel_index = torch.where(full_graph.x[:, voxel_label_index]==1)[0].size(0)-1

        # Remove any voxel source edges (voxels should not influence any other nodes in the graph)
        filtered_edges = torch.where(full_graph.edge_index[0] > last_voxel_index)[0]

        edge_index = full_graph.edge_index[:, filtered_edges]
        edge_attr = full_graph.edge_attr[filtered_edges]
        edge_labels = full_graph.edge_labels[filtered_edges]

        positive_voxels = torch.where(torch.sum(full_graph.y, dim=1) != 0)[0]
        occupied_voxels = torch.where(full_graph.contact_index != -1)[0]
        voxel_nodes, _ = torch.sort(
                            torch.unique(
                                torch.hstack((positive_voxels, occupied_voxels))))

        atom_nodes = torch.where(full_graph.x[:, voxel_label_index] == 0)[0]
        node_subset, _ = torch.sort(
                            torch.unique(
                                torch.hstack([positive_voxels, atom_nodes])))

        sub_edge_index, sub_edge_attr = subgraph(node_subset, edge_index, edge_attr) 
        _, sub_edge_labels = subgraph(node_subset, edge_index, edge_labels) 

        sub_index_dict = {}
        for atom_i, atom in enumerate(node_subset): 
            sub_index_dict[atom.item()] = atom_i

        sub_edge_index.apply_(lambda x: sub_index_dict[x])

        # pc_data = Data(x=torch.tensor(graph_x, dtype=torch.float),
        #                y=torch.tensor(graph_y, dtype=torch.long),
        #                pos=torch.tensor(graph_pos, dtype=torch.float),
        #                edge_index=torch.tensor(graph_edge_index, dtype=torch.long),
        #                edge_attr=torch.tensor(graph_edge_attr, dtype=torch.float),
        #                edge_labels=torch.tensor(graph_edge_labels, dtype=torch.long),
        #                contact_index=torch.tensor(contact_map, dtype=torch.long))

        sub_graph = Data(x=full_graph.x[node_subset],
                        y=full_graph.y[node_subset],
                        pos=full_graph.pos[node_subset],
                        edge_index=sub_edge_index,
                        edge_attr=sub_edge_attr,
                        edge_labels=sub_edge_labels,
                        contact_index=full_graph.contact_index[node_subset])

        pickle.dump(sub_graph, open(positive_sample_file, 'wb'))

