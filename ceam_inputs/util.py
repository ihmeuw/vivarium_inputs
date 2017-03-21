import os.path
import subprocess

import pandas as pd

from getpass import getuser

from ceam import config



STATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cen_functions_scripts')

def get_cache_directory():
    return config.get('input_data', 'intermediary_data_cache_path').format(username=getuser())

def gbd_year_range():
    year_start = round_to_gbd_year(config.getint('simulation_parameters', 'year_start'))
    year_end = round_to_gbd_year(config.getint('simulation_parameters', 'year_end'), down=False)
    if year_end == year_start:
        year_end += 5
    return year_start, year_end

# FIXME: Need to handle GBD 2013!!
def round_to_gbd_year(year, down=True):
    rounded_year = int(year/5)*5
    if not down and rounded_year != year:
        rounded_year += 5
    if year >= 2015:
        rounded_year = year
    return rounded_year
