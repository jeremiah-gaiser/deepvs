def get_protein_atoms(protein_pdb: str) -> list:
    protein_atom_data = []

    with open(protein_pdb, 'r') as pdb_in:  
        for line in pdb_in:
            if line[:4] != "ATOM":
                continue

            atom_data = [line[12:16].strip()]
            atom_xyz_beta = [float(line[ptr[0]:ptr[1]].strip()) for ptr in ((30,38), (38,46), (46,54), (60,66))]
            atom_data.extend(atom_xyz_beta)
            protein_atom_data.append(atom_data)

    return protein_atom_data



                        






