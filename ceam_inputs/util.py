import os.path
import subprocess

import pandas as pd

from getpass import getuser

from ceam import config



STATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cen_functions_scripts')

def get_cache_directory():
    return config.get('input_data', 'intermediary_data_cache_path').format(username=getuser())
