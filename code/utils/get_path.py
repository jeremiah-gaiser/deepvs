def get_path(config_dict, key, prefix=''):

    def gp(path_tree, key, prefix=''):
        for k,v in path_tree.items():
            if k == 'root_relative':
                prefix = path_tree['absolute']['directories']['root']

            if k == 'pdbbind_relative':
                prefix = path_tree['absolute']['directories']['pdbbind_dir']

            if k == key:
                return prefix+v

            if isinstance(v, str):
                continue

            rv = gp(path_tree[k], key, prefix)

            if rv is not None:
                return rv

    return gp(config_dict['paths'], key)
