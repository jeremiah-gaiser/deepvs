import sys
import random
from glob import glob
from code.utils.get_path import get_path
from copy import deepcopy
import numpy as np
from torch_geometric.data import Data
import torch
import pickle
import os

def partition_corpus(config: dict, in_dir: str, out_dir: str, partitions: int=20) -> None:
    if in_dir[-1] != '/':
        in_dir += '/'

    if out_dir[-1] != '/':
        out_dir += '/'

    partition_ft = out_dir + 'voxel_embeds_%s.pkl'

    embed_files = glob(in_dir + "*.pkl")
    embed_indices = np.arange(len(embed_files))
    np.random.shuffle(embed_indices)
    index_groups = np.array_split(embed_indices, partitions)

    for partition_idx, row in enumerate(index_groups):
        embed_collection = []

        for embed_idx in row: 
            embed_graph_file = embed_files[embed_idx]
            pdb_id = embed_graph_file.split('/')[-1].split('_')[0] 
            embed_graph = pickle.load(open(embed_files[embed_idx], 'rb'))
            embed_graph.pdb_id = pdb_id
            embed_collection.append(embed_graph)

        print(partition_ft % (partition_idx+1))
        pickle.dump(embed_collection, open(partition_ft % (partition_idx+1), 'wb'))
        del embed_collection
