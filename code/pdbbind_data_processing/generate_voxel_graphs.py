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

def generate_voxel_graphs(config: dict, id_batch: list, resolution: float=1.0, neighbor_count: int=10) -> None:
    ligand_graph_ft = get_path(config, 'mol_graph_ft')
    ligand_pdb_ft = get_path(config, 'ligand_pdb_ft')
    protein_pdb_ft = get_path(config, 'protein_pdb_ft')
    interaction_profile_ft = get_path(config, 'interaction_profile_ft')
    voxel_ft = get_path(config, 'voxel_ft')
    voxel_dir = get_path(config, 'voxel_dir') % resolution

    if os.path.exists(voxel_dir) == False:
        os.makedirs(voxel_dir)

    HEAVY_ATOM_LABELS = config['constants']['point_cloud']['HEAVY_ATOM_LABELS'] 
    INTERACTION_LABELS = config['constants']['point_cloud']['INTERACTION_LABELS'] 

    batch_total = len(id_batch)

    for pdb_i, pdb_id in enumerate(id_batch):
        try:
            print("generating %s: %s of %s" % (pdb_id, pdb_i+1, batch_total))
            skip = False

            ligand_pdb = ligand_pdb_ft % pdb_id
            protein_pdb = protein_pdb_ft % (pdb_id, pdb_id)
            interaction_file = interaction_profile_ft % pdb_id 
            voxel_file = voxel_ft % (resolution, pdb_id)

            if config['args']['skip']: 
                if os.path.exists(voxel_file):
                    continue

            for f in [ligand_pdb, protein_pdb, interaction_file]:
                if os.path.exists(f) == False:
                    skip = True 

            if skip:
                continue

            interaction_profile = pickle.load(open(interaction_file, 'rb'))
            voxel_coords = get_voxel_coords(config, protein_pdb, resolution, ligand_file=ligand_pdb)
            graph_y = np.zeros((len(voxel_coords), len(INTERACTION_LABELS)))
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
            node_onehot = [0 for _ in HEAVY_ATOM_LABELS]
            node_onehot[HEAVY_ATOM_LABELS.index('VOXEL')]=1

            vox_data = Data(y=torch.tensor(graph_y, dtype=torch.long),
                           pos=torch.tensor(voxel_coords, dtype=torch.float),
                           contact_map=torch.tensor(contact_map, dtype=torch.long))

            pickle.dump(vox_data, open(voxel_file, 'wb'))
        except Exception as e:
            print(e)