from code.utils.get_path import get_path
from torch_geometric.utils import subgraph 
import torch
import pickle
import os

def get_adjacent_nodes(edge_index, node_indices):
    adjacent_node_indices = torch.tensor([])

    for node_index in node_indices:
        incident_edge_indices = torch.where(edge_index[0] == node_index)[0]
        adjacent_node_indices = torch.hstack((adjacent_node_indices, edge_index[1][incident_edge_indices]))

    return torch.unique(adjacent_node_indices).long()

def filter_positive_examples(config: dict, id_batch: list, resolution: float=1.0) -> None:
    positive_samples_dir = get_path(config, 'positive_samples_dir') % resolution
    positive_sample_ft = get_path(config, 'positive_samples_ft')
    full_graph_ft = get_path(config, 'point_cloud_ft')

    ATOM_LABELS = config['constants']['point_cloud']['HEAVY_ATOM_LABELS']
    EDGE_LABELS = config['constants']['point_cloud']['EDGE_LABELS']
    voxel_label_index = ATOM_LABELS.index('VOXEL')
    voxel_voxel_edge_label_index = EDGE_LABELS.index('voxel-voxel')

    if os.path.exists(positive_samples_dir)==False:
        os.makedirs(positive_samples_dir)

    for pdb_i, pdb_id in enumerate(id_batch): 
        positive_sample_file = positive_sample_ft % (resolution, pdb_id)
        full_graph_file = full_graph_ft % (resolution, pdb_id)
        
        if os.path.exists(full_graph_file)==False:
            continue

        full_graph = pickle.load(open(full_graph_file, 'rb')) 

        not_voxel_edges = torch.where(full_graph.edge_labels[:, voxel_voxel_edge_label_index] == 0)[0]

        edge_index = full_graph.edge_index[:, not_voxel_edges]
        edge_attr = full_graph.edge_attr[not_voxel_edges]
        edge_labels = full_graph.edge_labels[not_voxel_edges]

        for r_i, row in enumerate(edge_labels):
            print(r_i, row)

        break

        # positive_voxels = torch.where(torch.sum(full_graph.y, dim=1) != 0)[0]
        # atom_nodes = torch.where(full_graph.x[:, voxel_label_index] == 0)[0]
        # node_subset = torch.hstack([positive_voxels, atom_nodes])

        # subgraph_index_dict = get_index_dict(node_subset)

        # sub_edge_index, sub_edge_attr = subgraph(node_subset, edge_index, edge_attr) 

        # sub_x = full_graph.x[node_subset]


        break


