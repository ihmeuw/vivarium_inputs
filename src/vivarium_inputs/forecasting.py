import pandas as pd

from vivarium_inputs.utilities import normalize_for_simulation, get_age_group_bins_from_age_group_id, gbd
from vivarium_gbd_access.forecasting import get_forecasting_data
from vivarium_inputs.core import get_location_name
from gbd_mapping.base_template import ModelableEntity


def get_forecast(measure: str, location_id: int, entity: ModelableEntity=None) -> pd.DataFrame:
    past, future = get_forecasting_data(measure, location_id, entity)

    # asfr and population require special names for value columns
    if measure == 'asfr':
        value_column = 'mean_value'
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

    if not ('value' in data or 'mean_value' in data):
        # we need to rename the value column to match vivarium_inputs convention
        col = set(data.columns) - {'year_id', 'sex_id', 'age_group_id', 'draw', 'scenario', 'location_id'}
        if len(col) > 1:
            raise ValueError(f"You have multiple value columns in your data.")
        data = data.rename(columns={col.pop(): value_column})

    data = normalize_for_simulation(data)

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


def combine_past_future(past: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    # if we have draws for a forecast measure where we don't have draws for past - replicate past for all draws
    if 'draw' in future and 'draw' not in past:  # population
        draws = pd.DataFrame({'draw': future.draw.unique(), 'key': 1})
        past = past.assign(key=1).merge(draws, how='outer').drop('key', 'columns')

    last_past_year = past.year_start.max()
    future = future[future.year_start > last_past_year]
    return past.append(future).reset_index(drop=True)



# just patching in forecasting data to replace existing gbd data - we have to work out the full mapping between
# entity keys and forecasting files at some point

# from vivarium_public_health.dataset_manager.artifact import Artifact
# from vivarium_inputs.forecasting import get_forecasting_data
# from gbd_mapping.cause import causes
#
# art = Artifact('/home/kate/vivarium_artifacts/forecasting_test.hdf')
#
# keys = ['cause.all_causes.cause_specific_mortality', 'population.structure',
#         'covariate.age_specific_fertility_rate.estimate', 'cause.peripheral_artery_disease.cause_specific_mortality']
# measures = ['death', 'population', 'asfr', 'cause_specific_mortality']
# entities = [None, None, None, causes.peripheral_artery_disease]
#
# for key, measure, entity in zip(keys, measures, entities):
#     print(f'Working on measure {measure}')
#     gbd = art.load(key)
#     art.write(f'{key}_gbd', gbd)
#
#     forecast = get_forecasting_data(measure, 102, entity)
#     art.replace(key, forecast)
