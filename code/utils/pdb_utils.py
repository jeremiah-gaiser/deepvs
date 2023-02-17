def get_pdb_atoms(pdb_file: str, ligand: bool=False, deprotonate: bool=False) -> list:
    atom_list = []
    atom_type = "ATOM"

    if ligand:
        atom_type = "HETATM"

    line_ptrs = []

    # BETA FACTOR
    if not ligand:
        line_ptrs.append((60,66))

    # X, Y, and Z coord values
    line_ptrs.extend([(30,38), (38,46), (46,54)])

    with open(pdb_file, 'r') as pdb_in:  
        for line in pdb_in:
            if line[:6].strip() != atom_type:
                continue

            if deprotonate:
                if line[76:78].strip() == 'H':
                    continue
            
            atom_data = [line[12:16].strip()]
            atom_data.extend([float(line[ptr[0]:ptr[1]].strip()) for ptr in line_ptrs])
            atom_list.append(atom_data)

    return atom_list 