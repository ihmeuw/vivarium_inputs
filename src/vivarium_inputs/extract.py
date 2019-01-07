from gbd_mapping.sequela import Sequela
import pandas as pd

from .globals import gbd
import vivarium_inputs.validation.raw as validation


def get_sequela_prevalence(entity: Sequela, location_id: int) -> pd.DataFrame:
    validation.check_metadata(entity, 'prevalence')
    data = gbd.get_como_draws(entity_id=entity.gbd_id, location_id=location_id, entity_type='sequela')
    data = data[data.measure_id == 5]
    validation.validate_raw_data(data, entity, 'prevalence', location_id)
    return data


def get_sequela_incidence(entity: Sequela, location_id: int) -> pd.DataFrame:
    validation.check_metadata(entity, 'incidence')
    data = gbd.get_como_draws(entity_id=entity.gbd_id, location_id=location_id, entity_type='sequela')
    data = data[data.measure_id == 6]
    validation.validate_raw_data(data, entity, 'incidence', location_id)
    return data


def get_sequela_disability_weight(entity: Sequela, location_id: int) -> pd.DataFrame:
    validation.check_metadata(entity, 'disability_weight')
    disability_weights = gbd.get_auxiliary_data('disability_weight', 'sequela', 'all')
    data = disability_weights.loc[disability_weights.healthstate_id == entity.healthstate.gbd_id, :]
    validation.validate_raw_data(data, entity, 'disability_weight', location_id)
    return data


def get_population_structure(location_id: int) -> pd.DataFrame:
    data = gbd.get_population(location_id)
    validation.validate_raw_data(data, 'population', 'structure', location_id)
    return data

