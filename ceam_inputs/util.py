import os.path

from getpass import getuser
from vivarium.configuration import build_simulation_configuration

STATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cen_functions_scripts')


def get_input_config(override_config=None):
    # This will grab the config in this users home directory as well as setting some defaults.
    input_config = build_simulation_configuration()
    inputs_config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'gbd_config.yaml')
    input_config.update(inputs_config_path, layer='base', source=inputs_config_path)
    input_config.update(override_config)
    return input_config


def get_cache_directory(config):
    return config.input_data.intermediary_data_cache_path.format(username=getuser())
