import pandas as pd
import math

from vivarium_inputs.core import get_location_id
from vivarium_inputs.utilities import normalize_for_simulation, get_age_group_bins_from_age_group_id, gbd, forecasting
from gbd_mapping import causes, covariates, etiologies
from vivarium_public_health.dataset_manager import EntityKey

NUM_DRAWS = 1000
FERTILE_AGE_GROUP_IDS = list(range(7, 15 + 1))  # need for calc live births by sex
BASE_COLUMNS = ['year_start', 'year_end', 'age_group_start', 'age_group_end', 'draw', 'sex']


def load_forecast(entity_key: EntityKey, location: str):
    entity_data = {
        "cause": {
            "mapping": causes,
            "getter": get_cause_data,
            "measures": ["cause_specific_mortality"]
        },
        "etiology": {
            "mapping": etiologies,
            "getter": get_etiology_data,
            "measures": ["incidence", "mortality"],
        },
        "population": {
            "mapping": {'': None},
            "getter": get_population_data,
            "measures": ["structure"],
        },
        "covariate": {
            "mapping": covariates,
            "getter": get_covariate_data,
            "measures": ["estimate"]
        },
    }
    mapping, getter, measures = entity_data[entity_key.type].values()
    entity = mapping[entity_key.name]
    data = getter(entity, entity_key.measure, get_location_id(location))
    data['location'] = location
    return data


def get_cause_data(cause, measure, location_id):
    data = forecasting.get_entity_measure(cause, measure, location_id)
    data = standardize_data(data, 0)
    value_column = 'value'
    data = normalize_forecasting(data, value_column)
    return data[BASE_COLUMNS + [value_column]]


def get_etiology_data(etiology, measure, location_id):
    data = forecasting.get_entity_measure(etiology, measure, location_id)
    data = standardize_data(data, 0)
    value_column = 'value'
    data = normalize_forecasting(data, value_column)
    return data[BASE_COLUMNS + [value_column]]


def get_population_data(_, measure, location_id):
    if measure == 'structure':
        data = forecasting.get_population(location_id)
        value_column = 'population'
        data = normalize_forecasting(data, value_column)
        return data[BASE_COLUMNS + [value_column]]
    else:
        raise ValueError(f"Only population.structure is supported from forecasting. You requested {measure}.")


def get_covariate_data(covariate, measure, location_id):
    if measure != 'estimate':
        raise ValueError(f"The only measure that can be retrieved for covariates is estimate. You requested {measure}.")
    value_column = 'mean_value'
    if covariate.name == 'live_births_by_sex':  # we have to calculate
        data = _get_live_births_by_sex(location_id)
    else:
        data = forecasting.get_entity_measure(covariate, measure, location_id)
        data = standardize_data(data, 0)
        data = normalize_forecasting(data, value_column)
    return data


def _get_live_births_by_sex(location_id):
    """Forecasting didn't save live_births_by_sex so have to calc from population
    and age specific fertility rate"""
    pop = forecasting.get_population(location_id)
    asfr = forecasting.get_entity_measure(covariates.age_specific_fertility_rate, 'estimate', location_id)

    # calculation of live births by sex from pop & asfr from Martin Pletcher

    fertile_pop = pop[((pop.age_group_id.isin(FERTILE_AGE_GROUP_IDS)) & (pop.sex_id == gbd.FEMALE))]
    data = asfr.merge(fertile_pop, on=['age_group_id', 'draw', 'year_id', 'sex_id', 'location_id', 'scenario'])
    data['live_births'] = data['asfr'] * data['population_agg']
    data = data.groupby(['draw', 'year_id', 'location_id'])[['live_births']].sum().reset_index()
    # normalize first because it would drop sex_id = 3 and duplicate for male and female but we need both for use in
    # vph FertilityCrudeBirthRate
    data = normalize_forecasting(data, 'mean_value')
    data['sex'] = 'Both'
    return data


def normalize_forecasting(data: pd.DataFrame, value_column='value') -> pd.DataFrame:
    assert not data.empty

    data = normalize_for_simulation(rename_value_columns(data, value_column))

    if "age_group_id" in data:
        if (data["age_group_id"] == 22).all():  # drop age if data is for all ages
            data = data.drop("age_group_id", "columns")
        else:
            # drop any age group ids that don't map to bins we use from gbd (e.g., 1 which is <5 or 158 which is <20)
            data = data[data.age_group_id.isin(gbd.get_age_bins().age_group_id)]
            data = get_age_group_bins_from_age_group_id(data)

    # not filtering on year as in vivarium_inputs.data_artifact.utilities.normalize b/c will drop future data

    if 'scenario' in data:
        data = data.drop("scenario", "columns")

    # make sure there are at least NUM_DRAWS draws
    return replicate_data(data)


def rename_value_columns(data: pd.DataFrame, value_column: str='value') -> pd.DataFrame:
    if not ('value' in data or 'mean_value' in data):
        # we need to rename the value column to match vivarium_inputs convention
        col = set(data.columns) - {'year_id', 'sex_id', 'age_group_id', 'draw', 'scenario', 'location_id'}
        if len(col) > 1:
            raise ValueError(f"You have multiple value columns in your data.")
        data = data.rename(columns={col.pop(): value_column})
    return data


def standardize_data(data: pd.DataFrame, fill_value: int) -> pd.DataFrame:
    # because forecasting data is already in long format, we need a custom standardize method

    # age_groups that we expect to exist for each entity
    whole_age_groups = gbd.get_age_group_id() if set(data.age_group_id) != {22} else [22]
    sex_id = data.sex_id.unique()
    year_id = data.year_id.unique()
    location_id = data.location_id.unique()

    index_cols = ['year_id', 'location_id', 'sex_id', 'age_group_id']
    expected_list = [year_id, location_id, sex_id, whole_age_groups]

    if 'draw' in data:
        index_cols += ['draw']
        expected_list += [data.draw.unique()]

    value_cols = {c for c in data.dropna(axis=1).columns if c not in index_cols}
    data = data.set_index(index_cols)

    # expected indexes to be in the data
    expected = pd.MultiIndex.from_product(expected_list, names=index_cols)

    new_data = data.copy()
    missing = expected.difference(data.index)

    # assign dtype=float to prevent the artifact error with mixed dtypes
    to_add = pd.DataFrame({column: fill_value for column in value_cols}, index=missing, dtype=float)

    new_data = new_data.append(to_add).sort_index()

    return new_data.reset_index()


def replicate_data(data):
    """If data has fewer than NUM_DRAWS draws, duplicate to have the full set.
    Assumes draws in data are sequential and start at 0
    """
    if 'draw' not in data:  # for things with only 1 draw, draw isn't included as a col
        data['draw'] = 0

    full_data = data.copy()
    og_draws = data.draw.unique()
    n_draws = len(og_draws)

    if n_draws < NUM_DRAWS:

        for i in range(1, math.ceil(NUM_DRAWS/n_draws)):

            max_draw = max(og_draws)
            if i == math.ceil(NUM_DRAWS/n_draws)-1 and NUM_DRAWS % n_draws > 0:
                max_draw = NUM_DRAWS % n_draws - 1

            draw_map = pd.Series(range(i*n_draws, i*n_draws + n_draws), index=og_draws)

            new_data = data[data.draw <= max_draw].copy()
            new_data['draw'] = new_data.draw.apply(lambda x: draw_map[x])

            full_data = full_data.append(new_data)

    return full_data


