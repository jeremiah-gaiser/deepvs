import os
import itertools
import sys
import re
import pickle
from plip.structure.preparation import PDBComplex
from code.utils.get_path import get_path

def get_plip_data(config: dict, id_batch: list) -> None:
    # Gather interaction data from all PDBBind complexes
    # Store interactions by location/type in interaction file  
    batch_total = len(id_batch)
    root = config['paths']['absolute']['directories']['root']

    ip_dir = get_path(config, 'interaction_profile_dir')
    ip_ft = get_path(config, 'interaction_profile_ft')
    protein_ft = get_path(config, 'pocket_pdb_ft')
    ligand_ft = get_path(config, 'ligand_pdb_ft')
    temporary_dir = get_path(config, 'temporary_dir')
    pdbbind_dir = get_path(config, 'pdbbind_dir')
    
    def mean(l):
        return sum(l) / len(l)

    # function takes object returned by PDBComplex.analyze() method
    # returns list of list of interactions in format [[TYPE, X, Y, Z]]
    def get_ligand_data(pl_interaction):
        ligand_interaction_data = []

        for interaction in pl_interaction.all_itypes:
            i_type = re.search(r".*\.(\S+)\'\>$", str(type(interaction))).group(1)

            if i_type == 'hbond':
                if interaction.protisdon == True:
                    interaction_record = [i_type+"_a", interaction.a.coords]
                else:
                    interaction_record = [i_type+"_d", interaction.h.coords]

            if i_type == 'hydroph_interaction':
                interaction_record = [i_type, interaction.ligatom.coords] 

            if i_type == 'halogenbond':
                interaction_record = [i_type, interaction.don.orig_x.coords] 

            if i_type == 'pistack':
                interaction_record = [i_type, tuple(interaction.ligandring.center)]

            if i_type == 'saltbridge':
                if interaction.protispos:
                    interaction_record = ['saltbridge_n', tuple(interaction.negative.center)]
                else:
                    interaction_record = ['saltbridge_p', tuple(interaction.positive.center)]

            if i_type == 'pication':
                if interaction.protcharged:
                    interaction_record = [i_type + '_r', tuple(interaction.ring.center)]
                else:
                    interaction_record = [i_type + '_c', tuple(interaction.charge.center)]

            if i_type in ['metal_complex', 'waterbridge']: 
                continue

            ligand_interaction_data.append(interaction_record)

        return ligand_interaction_data 

    def get_interaction_data(pdb_file):
        my_mol = PDBComplex()
        my_mol.load_pdb(pdb_file)
        my_mol.analyze()

        interaction_data = []

        for object_ids, pl_interaction in my_mol.interaction_sets.items():
            plip_profile = get_ligand_data(pl_interaction)
            interaction_data.extend(plip_profile)

        # Remove duplicates
        interaction_data = [x for x,_ in itertools.groupby(sorted(interaction_data))]
        return interaction_data


    def stringify_atom_idx(number, total_width):
        number = str(number)
        padding = total_width - len(number) 
        return " "*padding + number

    for target_count, target_id in enumerate(id_batch):
        protein_pdb =  protein_ft % (target_id, target_id)
        ligand_pdb = ligand_ft % target_id
        complex_pdb = temporary_dir + "%s_complex.pdb" % target_id
        ip_file = ip_ft % target_id

        if os.path.exists(ligand_pdb) == False:
            continue

        # PLIP Requires a pdb complex file containing both protein and ligand
        # store ATOM/HETATM PDB lines here
        complex_pdb_content = ""

        atom_idx = -1

        with open(protein_pdb, 'r') as protein_in:
            for line in protein_in:
                if line[:6].strip() in ['HETATM', 'ATOM']:
                    atom_idx = int(line[6:11].strip())

                if line[:3] == 'END':
                    continue

                complex_pdb_content += line 

        with open(ligand_pdb, 'r') as ligand_in:
            for line in ligand_in:
                if line[:6].strip() not in ['HETATM', 'ATOM']: 
                    continue
                atom_idx += 1
                line = line[:6] + stringify_atom_idx(atom_idx, 5) + line[11:] 
                complex_pdb_content += line

        complex_pdb_content += "END\n"

        with open(complex_pdb, 'w') as complex_out:
            complex_out.write(complex_pdb_content)

        interaction_data = get_interaction_data(complex_pdb) 

        # PLIP does not successfully fetch interaction data for all files
        # Only write interation profile if there is data available
        if len(interaction_data) > 0:
            pickle.dump(interaction_data, open(ip_file, 'wb'))

        os.remove(complex_pdb)

        print("PLIP Processed %s: %s of %s" % (target_id, target_count+1, batch_total))
