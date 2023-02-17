import sys
import numpy as np
import os
import yaml
import argparse 
from glob import glob

from code.pdbbind_data_processing.ligand_sdf2pdb import ligand_sdf2pdb 
# from code.pdbbind_data_processing.get_plip_data import get_plip_data
from code.pdbbind_data_processing.generate_point_clouds import generate_point_clouds
from code.pdbbind_data_processing.generate_mol_graphs import generate_mol_graphs
from code.pdbbind_data_processing.filter_positive_samples import filter_positive_samples

import code.utils.string_utils as string_utils
from code.utils.get_path import get_path

parser = argparse.ArgumentParser(
                    prog = 'DeepVS',
                    description = 'Virtual screening for fun and profit.',
                    epilog = 'For those in search of drugs.')

parser.add_argument('action', type=str)
parser.add_argument('-n', '--batch_number', type=int, default=1)
parser.add_argument('-c', '--batch_count', type=int, default=1)
parser.add_argument('-r', '--root', type=str, default='')
parser.add_argument('-s', '--skip', action='store_true') 

args = parser.parse_args()

if args.root == '':
    root = os.path.dirname(__file__) + "/"
else:
    root = args.root

CONFIG_PATH =  root + "config.yaml"

with open(CONFIG_PATH, 'r') as config_file:  
    config = yaml.safe_load(config_file) 

config['paths']['absolute']['directories']['root'] = root

config['args']['skip'] = args.skip

training_data_pipeline = ['ligand_sdf2pdb', 
                          'get_plip_data',
                          'generate_mol_graphs',
                          'generate_point_clouds',
                          'filter_positive_samples']

data_processing = False

if args.action in training_data_pipeline: 
    data_processing = True

script_dict = {
    'ligand_sdf2pdb': ligand_sdf2pdb,
    # 'get_plip_data': get_plip_data,
    'generate_mol_graphs': generate_mol_graphs,
    'generate_point_clouds': generate_point_clouds,
    'filter_positive_samples': filter_positive_samples
}

if data_processing:
    pdbbind_ids = []  

    for directory in config['paths']['root_relative']['directories'].values():
        root_relative_dir = root + directory
        if os.path.exists(root_relative_dir) == False:
            if "%s" not in root_relative_dir:
                os.makedirs(root_relative_dir)

    for target in glob(get_path(config, 'pdbbind_dir') + "*/"):
        pdb_id = string_utils.get_pdb_id(target)

        if pdb_id in ['index', 'readme']:
            continue

        pdbbind_ids.append(pdb_id)

    pdbbind_ids = np.array(sorted(pdbbind_ids))
    id_batch = np.array_split(pdbbind_ids, args.batch_count)[args.batch_number-1].tolist()

    script_dict[args.action](config, id_batch)