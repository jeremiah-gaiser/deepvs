
def get_basename(path):
    # Usual `os` library tools are file/directory specific.
    # Sometimes we need the name of the file or directory.
    basename_idx = -1

    if path[-1] == '/':
        basename_idx = -2

    return path.split('/')[basename_idx]


def get_pdb_id(path):
    # By convention, the PDB ID associated with any file...
    # ...should be the first sequence before an underscore.
    # e.g. 2a5b_pocket_embed.pkl.
    basename = get_basename(path)
    return basename.split('_')[0]