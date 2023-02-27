# For every pdb id in random_examples directory, go fetch the all relevant training data graphs 
# in the deepvs corpus.
# Output all data stored in these graphs into flat txt files so they may be be easily parsed in pymol.

import numpy as np
import pickle
import torch
from torch_geometric.data import Data
from glob import glob
import os
import yaml

random_examples_dir = "/xdisk/twheeler/jgaiser/deepvs2/data_check/random_examples/"
out_dir =  "/xdisk/twheeler/jgaiser/deepvs2/data_check/graph_data/"
voxel_graphs_ft = "/xdisk/twheeler/jgaiser/deepvs2/deepvs/data/training_data/graph_data/training_samples/1.0_angstroms/%s_trainings.pkl"
mol_graph_ft = "/xdisk/twheeler/jgaiser/deepvs2/deepvs/data/training_data/graph_data/molecules/%s_mol.pkl"
ip_ft = "/xdisk/twheeler/jgaiser/deepvs2/deepvs/data/training_data/structures/interaction_profiles/%s_ip.pkl"

voxel_out = out_dir + "voxel/%s/%s_voxel_%s.txt"
mol_out = out_dir + "mol/%s/%s_mol.txt"
ip_out = out_dir + "ip/%s/%s_ip.pkl"

config_path = "../../config.yaml"

with open(config_path, 'r') as config_file:  
    config = yaml.safe_load(config_file) 

POCKET_ATOM_LABELS = config['constants']['point_cloud']['HEAVY_ATOM_LABELS']
POCKET_EDGE_LABELS = config['constants']['point_cloud']['EDGE_LABELS']
INTERACTION_TYPES = np.array(config['constants']['point_cloud']['INTERACTION_LABELS'])
MOL_ATOM_LABELS = config['constants']['molecules']['ATOM_LABELS']

for pdb_dir in glob(random_examples_dir + "*/"):
    pdb_id = pdb_dir.split('/')[-2]
    
    if os.path.exists(voxel_graphs_ft % pdb_id)==False:
        continue

    if os.path.exists(mol_graph_ft % pdb_id)==False:
        continue

    for output_dir in [('/'.join(voxel_out.split('/')[:-1])+"/") % pdb_id, 
                       ('/'.join(mol_out.split('/')[:-1])+"/") % pdb_id, 
                       ('/'.join(ip_out.split('/')[:-1])+"/") % pdb_id]:
        if os.path.exists(output_dir)==False:
            os.makedirs(output_dir)

    voxel_graphs = pickle.load(open(voxel_graphs_ft % pdb_id, 'rb'))
    mol_graph = pickle.load(open(mol_graph_ft % pdb_id, 'rb'))
    os.system("cp %s %s" % (ip_ft % pdb_id, ip_out % (pdb_id, pdb_id)))

    for v_i, vg in enumerate(voxel_graphs):
        voxel_txtfile = voxel_out % (pdb_id, pdb_id, v_i)
        voxel_file_content = []

        voxel_file_content.append(["INTERACTION", INTERACTION_TYPES[torch.where(vg.y[0] == 1)[0]]])
        voxel_file_content.append(["CONTACT", vg.contact_map[0].item()])

        for label, pos in zip(vg.x, vg.pos):
            txt_row = ['NODE', POCKET_ATOM_LABELS[torch.where(label==1)[0].item()]]
            txt_row.extend(pos.tolist())
            voxel_file_content.append(txt_row)

        observed_edges = [] 

        with open(voxel_txtfile, 'w') as txt_out:
            for source,sink,label_onehot,attr, in zip(vg.edge_index[0], vg.edge_index[1], vg.edge_labels, vg.edge_attr): 
                source = source.item()
                sink = sink.item()
                node_pair = sorted([source,sink]) 

                if node_pair in observed_edges:
                    continue

                observed_edges.append(node_pair)

                edge_label = POCKET_EDGE_LABELS[torch.where(label_onehot==1)[0].item()] 
                attr = attr.item()

                txt_row = ["EDGE", source, sink, edge_label, attr]
                voxel_file_content.append(txt_row)

            for row in voxel_file_content:
                txt_out.write(" ".join([str(x) for x in row]))
                txt_out.write("\n")
