import pandas as pd

from vivarium_inputs.utilities import normalize_for_simulation, get_age_group_bins_from_age_group_id, gbd, forecasting
from vivarium_inputs.core import get_location_name
from gbd_mapping.base_template import ModelableEntity


def get_forecast(measure: str, location_id: int, entity: ModelableEntity=None) -> pd.DataFrame:
    past, future = forecasting.get_forecasting_data(measure, location_id, entity)

    # asfr and population require special names for value columns
    if measure == 'asfr':
        value_column = 'mean_value'
        # forecasting data doesn't have fertility rate for all ages (in gbd it's 0 for young ages) so we need to fill it
        past = standardize_data(past, 0)
        future = standardize_data(future, 0)
    elif measure == 'cause_specific_mortality':
        value_column = 'value'
        past = standardize_data(past, 0)
        future = standardize_data(future, 0)
    elif measure == 'population':
        value_column = 'population'
    else:
        value_column = 'value'

    past = normalize_forecasting(past, value_column=value_column)
    future = normalize_forecasting(future, value_column=value_column)

    data = combine_past_future(past, future)

    data['location'] = get_location_name(location_id)
    data = data.drop("location_id", "columns")

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
    return data


def rename_value_columns(data: pd.DataFrame, value_column: str='value') -> pd.DataFrame:
    if not ('value' in data or 'mean_value' in data):
        # we need to rename the value column to match vivarium_inputs convention
        col = set(data.columns) - {'year_id', 'sex_id', 'age_group_id', 'draw', 'scenario', 'location_id'}
        if len(col) > 1:
            raise ValueError(f"You have multiple value columns in your data.")
        data = data.rename(columns={col.pop(): value_column})
    return data


def combine_past_future(past: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    # if we have draws for a forecast where we don't have draws for past (population) - replicate past for all draws
    if 'draw' in future and 'draw' not in past:
        draws = pd.DataFrame({'draw': future.draw.unique(), 'key': 1})
        past = past.assign(key=1).merge(draws, how='outer').drop('key', 'columns')

    last_past_year = past.year_start.max()
    # any 'past' years in the future dataset are predicted past so we don't want to use them
    future = future[future.year_start > last_past_year]
    return past.append(future).reset_index(drop=True)


def standardize_data(data: pd.DataFrame, fill_value: int) -> pd.DataFrame:
    # because forecasting data is already in long format, we need a custom standardize method

    # age_groups that we expect to exist for each entity
    whole_age_groups = gbd.get_age_group_id()
    sex_id = data.sex_id.unique()
    year_id = data.year_id.unique()
    location_id = data.location_id.unique()
    draw = data.draw.unique()

    index_cols = ['year_id', 'location_id', 'sex_id', 'age_group_id', 'draw']

    value_cols = {c for c in data.dropna(axis=1).columns if c not in index_cols}
    data = data.set_index(index_cols)

    # expected indexes to be in the data
    expected = pd.MultiIndex.from_product([year_id, location_id, sex_id, whole_age_groups, draw], names=index_cols)

    new_data = data.copy()
    missing = expected.difference(data.index)

    # assign dtype=float to prevent the artifact error with mixed dtypes
    to_add = pd.DataFrame({column: fill_value for column in value_cols}, index=missing, dtype=float)

    new_data = new_data.append(to_add).sort_index()

    return new_data.reset_index()
