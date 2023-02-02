import sys
import numpy as np
import os
import yaml
import argparse 
from glob import glob

from code.pdbbind_data_processing.ligand_sdf2pdb import ligand_sdf2pdb 
from code.pdbbind_data_processing.get_plip_data import get_plip_data

import code.utils.string_utils as string_utils

parser = argparse.ArgumentParser(
                    prog = 'DeepVS',
                    description = 'Virtual screening for fun and profit.',
                    epilog = 'For those in search of drugs.')

parser.add_argument('action', type=str)
parser.add_argument('-n', '--batch_number', type=int, default=1)
parser.add_argument('-c', '--batch_count', type=int, default=1)

args = parser.parse_args()

root = os.path.dirname(__file__) + "/"
CONFIG_PATH =  root + "config.yaml"
config['root'] = root

with open(CONFIG_PATH, 'r') as config_file:  
    config = yaml.safe_load(config_file) 

training_data_processes = ['ligand_sdf2pdb']
data_processing = False

if script_key in training_data_proceses: 
    data_processing = True

script_dict = {
    'ligand_sdf2pdb': ligand_sdf2pdb
    'get_plip_data': get_plip_data
}

if data_processing:
    pdbbind_ids = []  

    for target in glob(config['pdbbind_dir'] + "*/"):
        pdb_id = string_utils.get_pdb_id(target)

        if pdb_id in ['index', 'readme']:
            continue

        pdbbind_ids.append(pdb_id)

    pdbbind_ids = np.array(sorted(pdbbind_ids))
    id_batch = np.array_split(pdbbind_ids, batch_count)[batch_number-1].tolist()

    script_dict[script_key](config, id_batch)