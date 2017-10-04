import os.path

from getpass import getuser

STATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cen_functions_scripts')


def get_input_config(override_config=None):
    # This will grab the config in this users home directory as well as setting some defaults.
    if override_config is None:
        from vivarium.config_tree import ConfigTree
        override_config = ConfigTree()
    _inputs_config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'gbd_config.yaml')
    override_config.update(_inputs_config_path, layer='base', source=_inputs_config_path)
    return override_config


def get_cache_directory(config):
    return config.input_data.intermediary_data_cache_path.format(username=getuser())
