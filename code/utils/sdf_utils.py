
def get_sdf_atoms(sdf_file: str, removeHs: bool=False) -> list:
	H_count = 0
	heavy_atom_data = []
	mol_pos = []
	heavy_marker = []
	mol = None

	with Chem.SDMolSupplier(mol_sdf_file, removeHs=removeHs, sanitize=False) as sd_mol_in:
	    for rdkit_mol in sd_mol_in:
	        mol = rdkit_mol

	if mol is None:
	    continue

	conformer = mol.GetConformer()

	for atom in mol.GetAtoms():
	    pos = conformer.GetAtomPosition(atom.GetIdx())
	    atom_x, atom_y, atom_z = (pos.x, pos.y, pos.z)
	    mol_pos.append([atom_x, atom_y, atom_z])

	    if atom.GetSymbol() == 'H':
	        H_count += 1
	        heavy_marker.append(0)
	        continue

	    heavy_atom_data.append([atom.GetSymbol(),
	                            atom_x,
	                            atom_y,
	                            atom_z])
	    heavy_marker.append(1)