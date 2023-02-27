import pickle
import re
import numpy as np
import sys
import os
import glob
import torch
from torch import nn
import torch_geometric
import random
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from code.utils.get_path import get_path
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from importlib import import_module
from code.utils.get_path import get_path

def train_classifier(config: dict, molecule_model: str, mol_weights_out: str, classifier_model, 
                     classifier_weights_out: str, embed_dir: str, mol_dir: str, holdout_list: list,  load_weights: str=None) -> None:

    moldir = get_path(config, 'mol_graph_dir')
    AC = getattr(import_module(config['models']['active_classifier'].replace("/", ".") % classifier_model), 'AC')
    ME = getattr(import_module(config['models']['mol_embedder'].replace("/", ".") % molecule_model), 'ME')



