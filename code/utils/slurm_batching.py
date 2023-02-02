from glob import glob
from string_utils import get_pdb_id 

def get_slurm_batch(input_glob_string: str, batch_number: int, batch_count: int, output_glob_string: str=None) -> list :
    # When running many embarrasingly parallel jobs, we use get_slurm_batch to gather the list...
    # ...of paths to files (the batch) we will be working with. 

    # generate list of all files according to `glob_string`
    root_dir = sorted(glob(glob_string))
    
    batch_size = int(len(root_dir) / batch_count)
    batch_remainder = len(root_dir) % batch_count
    
    if batch_number <= batch_remainder:
        batch_start_offset = batch_number-1
        batch_end_offset = batch_number 
    else:
        batch_start_offset = batch_remainder
        batch_end_offset = batch_remainder

    # batch_numbers are 1-indexed, so substract one from batch_number value...
    # ...when calculating slice indices.
    batch_start_index = (batch_number - 1) * batch_size + batch_start_offset
    batch_end_index = (batch_number - 1) * batch_size + batch_size + batch_end_offset
    
    batch = root_dir[batch_start_index:batch_end_index]

    # If `output_glob_string` is specified, this indicates that we don't want to re-generate...
    # ...any files that are already present. 
    if output_glob_string is not None:
        trimmed_batch = []
        completed_ids = [get_pdb_id(x) for x in sorted(glob(output_glob_string))]

        for file_path in batch:
            if get_pdb_id(file_path) not in completed_files:
                trimmed_batch.append(file_path)

        return trimmed_batch 

    return batch 
