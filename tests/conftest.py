import pytest

from ceam_inputs.util import get_input_config


@pytest.fixture(scope='module')
def base_config():
    base_config = get_input_config()
    # Remove user overrides but keep custom cache locations if any
    base_config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                       'input_data.auxiliary_data_folder'])
    base_config.simulation_parameters.year_start = 1990
    base_config.simulation_parameters.year_end = 2010
    base_config.simulation_parameters.time_step = 30.5
    return base_config
