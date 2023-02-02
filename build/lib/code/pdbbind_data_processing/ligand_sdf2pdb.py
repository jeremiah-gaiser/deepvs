import sys
import os
import glob
import yaml

# PLIP requires ligand and protein structure data in one PDB.
# So we need to generate PDB versions of SDF ligand files.

def main(): 
  with open('config.yaml', 'r') as config_file:  
    config = yaml.safe_load(config_file)

  # command line function to convert isdf file to pdb
  obabel_command = "obabel -isdf %s -opdb > %s" 

  ligand_file_template = config['ligand_pdb_file_template']

  output_dir = "/".join(ligand_file_template.split('/')[:-1])+"/"

  # if output directory doesn't exist, make it
  if os.path.exists(output_dir) == False:
    os.system("mkdir -p %s" % output_dir)

  input_dir = sorted(glob.glob(config['pdbbind_dir'] + "/*/"))

  # do not count 'index','readme' directories in total
  input_total = len(input_dir)-2

  for dir_idx,input_target_dir in enumerate(input_dir):
    target_id = input_target_dir.split('/')[-2]

    # skip index and readme directories
    if target_id in ['index', 'readme']:
      continue

    input_ligand_file = input_target_dir + target_id + "_ligand.sdf"
    output_ligand_file = ligand_file_template % target_id

    # if ligand PDB doesn't exist, make it
    if os.path.exists(output_ligand_file) == False:
      os.system(obabel_command % (input_ligand_file, output_ligand_file))

    print("Converted %s: %s of %s" % (target_id, dir_idx+1, input_total))