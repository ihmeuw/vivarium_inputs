import pandas as pd

from gbd_mapping.sequela import Sequela
from vivarium_inputs import extract
from vivarium_inputs.utilities import normalize, reshape, remove_ids


def get_sequela_prevalence(entity: Sequela, location: int) -> pd.DataFrame:
    data = extract.get_sequela_prevalence(entity, location)
    data = normalize(data, 0, location)
    data = reshape(data)
    data = remove_ids(data)
    return data


def get_sequela_incidence(entity: Sequela, location: int) -> pd.DataFrame:
    data = extract.get_sequela_incidence(entity, location)
    data = normalize(data, 0, location)
    data = reshape(data)
    data = remove_ids(data)
    return data


def get_sequela_disability_weight(entity: Sequela, _) -> pd.DataFrame:
    data = extract.get_sequela_disability_weight(entity, _)
    data = normalize(data, _, __)
    data = reshape(data)
    data = remove_ids(data)
    return data

