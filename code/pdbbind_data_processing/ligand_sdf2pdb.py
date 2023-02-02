import sys
import os
import glob

def ligand_sdf2pdb(config: dict, id_batch: list) -> None:
    # PLIP requires ligand and protein structure data in one PDB.
    # So we need to generate PDB versions of SDF ligand files.

    # command line function to convert isdf file to pdb
    obabel_command = "obabel -isdf %s -opdb > %s" 
    root = config['root']

    ligand_pdb_ft = root + config['ligand_pdb_ft']
    ligand_sdf_ft = config['pdbbind_dir'] + config['ligand_sdf_ft']
    output_dir = root + config['ligand_pdb_dir']

    # if output directory doesn't exist, make it
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    input_total = len(id_batch)

    for id_itr, pdb_id in enumerate(id_batch):
        input_sdf = ligand_sdf_ft % (pdb_id, pdb_id)
        output_pdb = ligand_pdb_ft % pdb_id 

        os.system(obabel_command % (input_sdf, output_pdb))

        print("Converted %s: %s of %s" % (pdb_id, id_itr+1, input_total))