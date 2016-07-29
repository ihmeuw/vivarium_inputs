from ceam import config

from ceam.gbd_data.gbd_ms_functions import load_data_from_cache, get_modelable_entity_draws

def get_excess_mortality(modelable_entity_id):
    return load_data_from_cache(get_modelable_entity_draws, 'rate', location_id=config.getint('simulation_parameters', 'location_id'), year_start=config.getint('simulation_parameters', 'year_start'), year_end=config.getint('simulation_parameters', 'year_end'), measure=9, me_id=modelable_entity_id)

def get_incidence(modelable_entity_id):
    return load_data_from_cache(get_modelable_entity_draws, 'rate', location_id=config.getint('simulation_parameters', 'location_id'), year_start=config.getint('simulation_parameters', 'year_start'), year_end=config.getint('simulation_parameters', 'year_end'), measure=6, me_id=modelable_entity_id)
