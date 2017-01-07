import os.path
import subprocess

import pandas as pd

from getpass import getuser

from ceam import config


PYTHON_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cen_functions_scripts')

def get_cache_directory():
    return config.get('input_data', 'intermediary_data_cache_path').format(username=getuser())

def python_wrapper(py_file_name, out_file_name, *args):
    cache_path = get_cache_directory()
    path = os.path.join(cache_path, 'gbd_to_microsim_unprocessed_data', out_file_name)
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path))
        except FileExistsError:
            # Directory already exists, which is fine
            pass

        pyfile = os.path.join(PYTHON_PATH, py_file_name)
        cmd = '/ihme/code/central_comp/new_miniconda/envs/gbd_env/bin/python {} {}'.format(pyfile, ' '.join([str(a) for a in args] + ['"'+path+'"']))
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    return pd.read_csv(path)

def stata_wrapper(do_file_name, out_file_name, *args):
    cache_path = get_cache_directory()
    path = os.path.join(cache_path, 'gbd_to_microsim_unprocessed_data', out_file_name)
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path))
        except FileExistsError:
            # Directory already exists, which is fine
            pass

        dofile = os.path.join(STATA_PATH, do_file_name)
        cmd = 'stata -q "{}" {}, nostop'.format(dofile, ' '.join([str(a) for a in args] + ['"'+path+'"']))
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    return pd.read_csv(path)
