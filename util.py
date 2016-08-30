import os.path
import subprocess

import pandas as pd

from ceam import config

STATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cen_functions_scripts')

def stata_wrapper(do_file_name, out_file_name, *args):
    cache_path = config.get('input_data', 'intermediary_data_cache_path')
    path = os.path.join(cache_path, 'gbd_to_microsim_unprocessed_data', out_file_name)
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path))
        except FileExistsError:
            # Directory already exists, which is fine
            pass

        dofile = os.path.join(STATA_PATH, do_file_name)
        cmd = ["stata", "-b", "do", dofile] + [str(a) for a in args] + [path]
        subprocess.call(cmd)
    return pd.read_csv(path)
