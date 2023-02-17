import pickle
import os
import re
import numpy as np
import sys
import glob
import yaml
from copy import deepcopy
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import code.utils.mol_gen_utils as mol_gen_utils 
from code.utils.get_path import get_path
from code.utils.get_distance import get_distance 

def generate_mol_graphs(config: dict, id_batch: list, ) -> None:
    INTERACTION_LABELS = config['constants']['point_cloud']['INTERACTION_LABELS'] 
    ATOM_LABELS = config['constants']['molecules']['ATOM_LABELS'] 
    # ['halogenbond', 'hbond_a', 'hbond_d', 'hydroph_interaction', 'pication_c', 'pication_r', 'pistack', 'saltbridge_n', 'saltbridge_p']

    batch_total = len(id_batch)

    mol_graph_ft = get_path(config, 'mol_graph_ft')
    mol_pdb_ft = get_path(config, 'ligand_pdb_ft')
    mol_sdf_ft = get_path(config, 'ligand_sdf_ft')
    ip_ft = get_path(config, 'interaction_profile_ft')

    for t_idx, pdb_id in enumerate(id_batch):
        mol_graph_file = mol_graph_ft % pdb_id

        if config['args']['skip']: 
            if os.path.exists(mol_graph_file):
                continue

        print("%s: %s of %s" % (pdb_id, t_idx, batch_total))

        pdb_file_content = ""
        pdb_H_count = 0
        pdb_heavy_atom_data = []
        mol_pos = []
        heavy_marker = []

        with open(mol_pdb_ft % pdb_id, 'r') as pdb_in:
            for line in pdb_in:
                pdb_file_content += line

                if line[:6].strip() in ['ATOM', 'HETATM']:
                   
                    atom_x, atom_y, atom_z = (float(line[30:38].strip()),
                                              float(line[38:46].strip()),
                                              float(line[46:54].strip()))

                    mol_pos.append([atom_x, atom_y, atom_z])

                    # skip hydrogen atoms in ligand PDB
                    if line[76:78].strip() == 'H':
                        pdb_H_count += 1
                        heavy_marker.append(0)
                        continue

                    pdb_heavy_atom_data.append([line[12:16].strip(),
                                                atom_x,
                                                atom_y,
                                                atom_z])
                    heavy_marker.append(1)

        mol_y = np.zeros((len(heavy_marker), len(INTERACTION_LABELS)))

        # if no ip file, complex is with peptide/nucleic acid, so skip it.
        if os.path.exists(ip_ft % pdb_id) == False:
            continue 

        ip = pickle.load(open(ip_ft % pdb_id, 'rb'))

        for record in ip:
            itype, interaction_xyz = record

            # list of atomic distances from interaction location to atoms in ligand
            # list items correspond in `pdb_data_distances` correspond to list items in `pdb_heavy_atom_data`
            pdb_data_distances = np.array([get_distance(x[-3:], interaction_xyz) for x in pdb_heavy_atom_data])

            # list of atom indices corresponding to 'pdb_heavy_atom_data', sorted by distance to interaction location 
            sorted_pdb_data_indices = np.argsort(pdb_data_distances)

            # pication_r and pistack interactions located in the center of a ring
            # every member of that ring should be labeled with interaction 
            if itype in ['pication_r', 'pistack']:
                min_distance = pdb_data_distances[sorted_pdb_data_indices[0]]

                # iterate through ligand atoms in order of distance to interaction location
                # if difference in shortest distance from current atom tdistance to interaction location is greater thatn 0.5...
                # ...we are out of the range of atoms that are members of the ring
                for a_i, atom_idx in enumerate(sorted_pdb_data_indices):
                    if pdb_data_distances[atom_idx] - min_distance > 0.5:
                        break

                # `a_i` nearest atoms to be labeled 
                interacting_atoms = sorted_pdb_data_indices[:a_i]
            else:
                # not a ring-centered interaction, so only closest atom is to be labeled.  
                interacting_atoms = [sorted_pdb_data_indices[0]]

            # update onehot vector representing interaction type corresponding to ligand atom
            for atom_idx in interacting_atoms:
                mol_y[atom_idx] = mol_gen_utils.one_hot_update(INTERACTION_LABELS, mol_y[atom_idx], [itype])

        try:
            molecule = Chem.rdmolfiles.MolFromPDBBlock(pdb_file_content, removeHs=False)
            g = mol_gen_utils.generate_mol_graph(molecule, mol_y, ATOM_LABELS)
            mol_pos = torch.tensor(mol_pos)
            heavy_marker = torch.tensor(heavy_marker)
            g.heavy = heavy_marker
            g.pos = mol_pos
            pickle.dump(g, open(mol_graph_file, 'wb'))
        except Exception as e:
            print(e, pdb_id)

