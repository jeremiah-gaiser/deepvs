from code.utils.get_path import get_path
import sys
import numpy as np
from code.pdbbind_data_processing.get_voxel_coords import get_voxel_coords 
from code.utils.pdb_utils import get_pdb_atoms

def generate_point_clouds(config: dict, id_batch: list) -> None:
    ligand_pdb_ft = get_path(config, 'ligand_pdb_ft')
    protein_pdb_ft = get_path(config, 'protein_pdb_ft')
    interaction_profile_ft = get_path(config, 'interaction_profile_ft')

    for pdb_id in id_batch:
        ligand_pdb = ligand_pdb_ft % pdb_id
        protein_pdb = protein_pdb_ft % (pdb_id, pdb_id)


        protein_atom_data = np.array(get_pdb_atoms(protein_pdb))
        voxel_coords = np.array(get_voxel_coords(config, protein_pdb, 1.0, ligand_file=ligand_pdb))



