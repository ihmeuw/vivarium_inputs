import pandas as pd

from vivarium_gbd_access import gbd
from gbd_mapping.sequela import Sequela
from vivarium_inputs import extract
from vivarium_inputs.utilities import normalize, reshape, remove_ids, DataNotExistError


def get_sequela_prevalence(entity: Sequela, location: int) -> pd.DataFrame:
    data = extract.get_sequela_prevalence(entity, location)
    data = normalize(data, location, 0)
    data = reshape(data)
    data = remove_ids(data)
    return data


def get_sequela_incidence(entity: Sequela, location: int) -> pd.DataFrame:
    data = extract.get_sequela_incidence(entity, location)
    age_groups = gbd.get_age_group_id()
    data = data[data.age_group_id.isin(age_groups)]
    data = normalize(data, location, 0)
    data = reshape(data)
    return data


def get_sequela_birth_prevalence(entity: Sequela, location: int) -> pd.DataFrame:
    if not entity.birth_prevalence_exists:
        raise DataNotExistError(f'{entity.name} does not have data for birth prevalence')
    else:
        data = extract.get_sequela_incidence(entity, location)
        data = data[data.age_group_id == 164]
        data.drop('age_group_id', axis=1, inplace=True)
        data = normalize(data, location)
        data = reshape(data, to_keep=('year_id', 'sex_id', 'location_id'))
        return data


def get_sequela_disability_weight(entity: Sequela, location: int) -> pd.DataFrame:
    data = extract.get_sequela_disability_weight(entity, location)
    data = normalize(data, location)
    data = reshape(data)
    return data

