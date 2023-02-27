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
   
def generate_protein_graphs(config: dict, id_batch: list, neighbor_count: int=15) -> None:
    protein_pdb_ft = get_path(config, 'pocket_pdb_ft')
    protein_graph_ft = get_path(config, 'protein_graph_ft')
    voxel_graph_ft = get_path(config, 'voxel_ft')
    protein_graph_dir = get_path(config, 'protein_graph_dir')

    if os.path.exists(protein_graph_dir) == False:
        os.makedirs(protein_graph_dir)

    ATOM_LABELS = config['constants']['point_cloud']['HEAVY_ATOM_LABELS'] 
    label_vector_template = [0 for _ in ATOM_LABELS]

    batch_total = len(id_batch)

    for pdb_i, pdb_id in enumerate(id_batch):
        try:
            print("generating %s: %s of %s" % (pdb_id, pdb_i+1, batch_total))
            skip = False

            protein_pdb = protein_pdb_ft % (pdb_id, pdb_id)
            protein_graph_file = protein_graph_ft % pdb_id
            voxel_graph_file = get_path(config, 'voxel_ft') % ("1.0", pdb_id)

            if config['args']['skip']: 
                if os.path.exists(protein_graph_file):
                    continue

            for f in [protein_pdb, voxel_graph_file]:
                if os.path.exists(f) == False:
                    skip = True 

            if skip:
                continue

            graph_x = []
            graph_pos = []
            graph_beta_factor = []

            protein_atom_data = get_pdb_atoms(protein_pdb, deprotonate=True)
            voxel_graph = pickle.load(open(voxel_graph_file, 'rb'))

            # For every voxel point, get N closest protein atoms and add them to graph node list
            for voxel_coord in voxel_graph.pos:
                nearest_protein_atoms = sorted(protein_atom_data, key=lambda x: get_distance(x[-3:], voxel_coord))[:neighbor_count]

                for protein_atom in nearest_protein_atoms:
                    atom_coords = protein_atom[-3:]

                    if atom_coords not in graph_pos:
                        label_vector = deepcopy(label_vector_template)
                        label_vector[ATOM_LABELS.index(protein_atom[0])] = 1

                        graph_x.append(label_vector)
                        graph_pos.append(atom_coords)
                        graph_beta_factor.append(protein_atom[1])

            prot_graph = Data(x=torch.tensor(graph_x, dtype=torch.float),
                              pos=torch.tensor(graph_pos, dtype=torch.float),
                              b_factor=torch.tensor(graph_beta_factor, dtype=torch.float))

            pickle.dump(prot_graph, open(protein_graph_file, 'wb'))
        except Exception as e:
            print(e)