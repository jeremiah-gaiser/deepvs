from code.utils.get_path import get_path
from copy import deepcopy
import os
import pickle
import sys
import numpy as np
from code.pdbbind_data_processing.get_voxel_coords import get_voxel_coords 
from code.utils.pdb_utils import get_pdb_atoms
from code.utils.get_distance import get_distance 

import torch
from torch_geometric.data import Data

interaction_types = []

class EdgeData:
    def __init__(self, feature_labels):
        self.edge_index = np.array([[], []])
        self.edge_attr = np.array([])
        self.edge_labels = np.array([])
        self.feature_labels = feature_labels

    def are_adjacent(self, n1, n2):
        source = self.edge_index[0]
        sink = self.edge_index[1]

        incident_edges = np.where(source == n1) 

        if n2 in sink[incident_edges]:
            return True
        else:
            return False

    def add_edge(self, n1, n2, distance, label):
        tail = np.array([[n1,n2],[n2,n1]]) 
        label_vector = np.zeros(len(self.feature_labels))
        label_vector[self.feature_labels.index(label)] = 1

        self.edge_index = np.hstack((self.edge_index, tail))
        self.edge_attr = np.hstack((self.edge_attr, [distance]*2))

        if self.edge_labels.size == 0:
            self.edge_labels = np.array([label_vector]*2)
        else:
            self.edge_labels = np.vstack((self.edge_labels, [label_vector]*2))

    def get_data(self):
        return self.edge_index, self.edge_attr, self.edge_labels
    
def generate_point_clouds(config: dict, id_batch: list, resolution: float=1.0, neighbor_count: int=10) -> None:
    ligand_graph_ft = get_path(config, 'mol_graph_ft')
    ligand_pdb_ft = get_path(config, 'ligand_pdb_ft')
    protein_pdb_ft = get_path(config, 'pocket_pdb_ft')
    interaction_profile_ft = get_path(config, 'interaction_profile_ft')
    pc_ft = get_path(config, 'point_cloud_ft')
    pc_dir = get_path(config, 'point_cloud_dir') % resolution

    if os.path.exists(pc_dir) == False:
        os.makedirs(pc_dir)

    HEAVY_ATOM_LABELS = config['constants']['point_cloud']['HEAVY_ATOM_LABELS'] 
    INTERACTION_LABELS = config['constants']['point_cloud']['INTERACTION_LABELS'] 
    EDGE_LABELS = config['constants']['point_cloud']['EDGE_LABELS'] 

    batch_total = len(id_batch)

    for pdb_i, pdb_id in enumerate(id_batch):
        try:
            print("generating %s: %s of %s" % (pdb_id, pdb_i+1, batch_total))
            skip = False

            ligand_pdb = ligand_pdb_ft % pdb_id
            protein_pdb = protein_pdb_ft % (pdb_id, pdb_id)
            interaction_file = interaction_profile_ft % pdb_id 
            pc_file = pc_ft % (resolution, pdb_id)

            if config['args']['skip']: 
                if os.path.exists(pc_file):
                    continue

            for f in [ligand_pdb, protein_pdb, interaction_file]:
                if os.path.exists(f) == False:
                    skip = True 

            if skip:
                continue

            interaction_profile = pickle.load(open(interaction_file, 'rb'))

            graph_nodes = []

            edge_data = EdgeData(EDGE_LABELS)
            
            protein_atom_data = get_pdb_atoms(protein_pdb, deprotonate=True)
            voxel_coords = get_voxel_coords(config, protein_pdb, resolution, ligand_file=ligand_pdb)

            for voxel_coord in voxel_coords:
                graph_nodes.append(['VOXEL', 0] + voxel_coord)  

            # Add edges between all adjacent voxel nodes
            for v_i in range(len(graph_nodes)-1):
                node_1 = graph_nodes[v_i]
                for v_j in range(v_i+1, len(graph_nodes)):
                    node_2 = graph_nodes[v_j]
                    voxel_distance = get_distance(node_1[-3:], node_2[-3:])

                    if voxel_distance <= resolution:
                        edge_data.add_edge(v_i, v_j, voxel_distance, 'voxel-voxel')

            # For every voxel point, get N closest protein atoms and add them to graph node list
            for voxel_node_index in range(len(graph_nodes)):
                voxel_node = graph_nodes[voxel_node_index]
                protein_indices = []

                nearest_protein_atoms = sorted(protein_atom_data, key=lambda x: get_distance(x[-3:], voxel_node[-3:]))[:neighbor_count]

                for protein_atom in nearest_protein_atoms:
                    if protein_atom not in graph_nodes:
                        graph_nodes.append(protein_atom)

                    protein_atom_index = graph_nodes.index(protein_atom)
                    protein_indices.append(protein_atom_index)

                    voxel_atom_distance = get_distance(protein_atom[-3:], voxel_node[-3:])
                    edge_data.add_edge(voxel_node_index, protein_atom_index, voxel_atom_distance, 'atom-voxel')

                # if edge does not exist between any two of the neighboring protein atoms, add atom-atom edge
                for a_i in range(len(nearest_protein_atoms)-1):
                    protein_atom_1 = nearest_protein_atoms[a_i]
                    protein_atom_i = graph_nodes.index(protein_atom_1)

                    for a_j in range(a_i+1, len(nearest_protein_atoms)):
                        protein_atom_2 = nearest_protein_atoms[a_j]
                        protein_atom_j = graph_nodes.index(protein_atom_2)

                        if edge_data.are_adjacent(protein_atom_i, protein_atom_j):
                            continue

                        atom_distance = get_distance(protein_atom_1[-3:], protein_atom_2[-3:])

                        edge_data.add_edge(protein_atom_i, protein_atom_j, atom_distance, 'atom-atom')

            graph_y = np.zeros((np.size(graph_nodes, 0), len(INTERACTION_LABELS)))
            contact_map = [-1 for x in graph_y]

            # For every atom in ligand, properly label closest voxel point
            if os.path.exists(ligand_graph_ft % pdb_id):
                ligand_graph = pickle.load(open(ligand_graph_ft % pdb_id, 'rb'))

                for atom_index, coord in enumerate(ligand_graph.pos):
                    neighbor_voxel = sorted(voxel_coords, key=lambda x: get_distance(coord, x))[0]
                    voxel_index = voxel_coords.index(neighbor_voxel)
                    contact_map[voxel_index] = atom_index

            # For every interaction in interaction profile, properly label closest voxel point
            for record in interaction_profile:
                neighbor_voxel = sorted(voxel_coords, key=lambda x: get_distance(record[1], x))[0]
                voxel_index = voxel_coords.index(neighbor_voxel)
                interaction_index = INTERACTION_LABELS.index(record[0])
                graph_y[voxel_index][interaction_index] = 1
                    
            graph_x = []              
            graph_pos = []
            node_onehot = [0 for _ in HEAVY_ATOM_LABELS]

            for node in graph_nodes:
                node_label_onehot = deepcopy(node_onehot)
                node_label_onehot[HEAVY_ATOM_LABELS.index(node[0])] = 1
                node_feature_vector = node_label_onehot + [node[1]]
                graph_x.append(node_feature_vector)
                graph_pos.append(node[-3:])

            graph_edge_index, graph_edge_attr, graph_edge_labels = edge_data.get_data() 

            pc_data = Data(x=torch.tensor(graph_x, dtype=torch.float),
                           y=torch.tensor(graph_y, dtype=torch.long),
                           pos=torch.tensor(graph_pos, dtype=torch.float),
                           edge_index=torch.tensor(graph_edge_index, dtype=torch.long),
                           edge_attr=torch.tensor(graph_edge_attr, dtype=torch.float),
                           edge_labels=torch.tensor(graph_edge_labels, dtype=torch.long),
                           contact_index=torch.tensor(contact_map, dtype=torch.long))

            pickle.dump(pc_data, open(pc_file, 'wb'))
        except Exception as e:
            print(e)