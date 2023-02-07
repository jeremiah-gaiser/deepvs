from code.utils.pdb_utils import get_pdb_atoms
import numpy as np

def get_voxel_coords(config: dict, 
                     pdb_file: str, 
                     resolution: float,
                     ligand_file: str=None,
                     min_coord: list=None, 
                     max_coord: list=None,
                     padding: float=1,
                     shrinkwrap: bool=False) -> list:
    
    def get_diagonal_from_ligand(ligand_file, padding, resolution):
        ligand_atom_coords = np.array(get_pdb_atoms(ligand_file, ligand=True))[:,-3:].astype(float)
        voxel_points = []

        padding_array = np.array([padding]*3)

        max_xyz = np.array([np.max(ligand_atom_coords[:,0]),
                            np.max(ligand_atom_coords[:,1]),
                            np.max(ligand_atom_coords[:,2])]) + padding_array

        min_xyz = np.array([np.min(ligand_atom_coords[:,0]),
                            np.min(ligand_atom_coords[:,1]),
                            np.min(ligand_atom_coords[:,2])]) + padding_array

        for x_val in np.arange(min_xyz[0], max_xyz[0], resolution):
            for y_val in np.arange(min_xyz[1], max_xyz[1], resolution):
                for z_val in np.arange(min_xyz[2], max_xyz[2], resolution):
                    voxel_points.append([x_val, y_val, z_val])

        return voxel_points

    if ligand_file is not None:
        return get_diagonal_from_ligand(ligand_file, padding, resolution)








