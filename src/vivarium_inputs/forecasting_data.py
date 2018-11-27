import xarray as xr
import pandas as pd

from vivarium_inputs.utilities import normalize_for_simulation, get_age_group_bins_from_age_group_id, gbd
from vivarium_inputs.core import get_location_name
from fbd_core.file_interface.file_interface import FBDPath
from gbd_mapping.cause_template import Cause
from gbd_mapping.base_template import ModelableEntity
import db_queries

# for now, use BAU scenario 0
FORECASTING_SCENARIO = 0

# we have to get these from forecasting
FORECASTING_VERSIONS = {'asfr': {'future': '/5/future/asfr/20180620_no_int_shift_logit/asfr.nc',
                                 'past': '/5/past/asfr/20180619_va83/asfr.nc'},
                        'population': {'future': '/4/future/population/20180729_fixed_mediation_squeezed_agg/population_agg.nc',
                                       'past': '/4/past/population/20180803_best_with_aggs/population.nc'},
                        'death': {'future': '/4/future/death/20180729_fixed_mediation_squeezed_agg/_all.nc',
                                  'past': '/4/past/death/20181008_fhs_aggregate/_all.nc'},
                        'cause_specific_mortality': {'future': '/4/future/death/20180729_fixed_mediation_squeezed_agg/',
                                                     'past': '/4/past/death/20180330_new_draws/'}
                        }


def get_forecasting_data(measure: str, location_id: int, entity: ModelableEntity=None) -> pd.DataFrame:

    if measure not in FORECASTING_VERSIONS:
        raise NotImplementedError(f"You requested forecasting data for {measure} but we don't currently have recorded "
                                  f"versions for that measure.")

    measure_handlers = {
        'population': get_forecasting_population,
        'death': get_forecasting_death,
        'asfr': get_forecasting_asfr,
        'cause_specific_mortality': get_forecasting_cause_specific_mortality
    }

    data = measure_handlers[measure](entity, location_id)
    data['location'] = get_location_name(location_id)
    data = data.drop("location_id", "columns")

    return data


def get_forecasting_population(_, location_id: int) -> pd.DataFrame:
    # read in netcdf files
    future = xr.open_dataarray(get_version('population', 'future'))\
                               .sel(location_id=location_id, scenario=FORECASTING_SCENARIO)
    # we have to give the past DataArray a name in order to be able to transform into a df
    past = xr.open_dataarray(get_version('population', 'past'))\
        .sel(location_id=location_id).rename('population')

    # transform into data frames and normalize
    future = normalize_forecasting(future.to_dataframe().reset_index(), value_column='population')
    past = normalize_forecasting(past.to_dataframe().reset_index(), value_column='population')

    return combine_past_future(past, future)


def get_forecasting_death(_, location_id: int) -> pd.DataFrame:
    # read in netcdf files
    future = xr.open_dataarray(get_version('death', 'future'))\
        .sel(location_id=location_id, scenario=FORECASTING_SCENARIO)
    # we have to give the past DataArray a name in order to be able to transform into a df
    past = xr.open_dataarray(get_version('death', 'past')).sel(location_id=location_id).rename('death')

    # transform into data frames and normalize

    # future data for deaths has extra columns with constant values that we don't need
    future = normalize_forecasting(future.to_dataframe().reset_index())

    past = past.to_dataframe().reset_index().drop(['rei', 'acause', 'metric_id'], 'columns')
    past = normalize_forecasting(past)

    return combine_past_future(past, future)


def get_forecasting_asfr(_, location_id: int) -> pd.DataFrame:
    # read in netcdf files
    future = xr.open_dataarray(get_version('asfr', 'future'))\
        .sel(location_id=location_id, scenario=FORECASTING_SCENARIO)
    # we have to give the past DataArray a name in order to be able to transform into a df
    past = xr.open_dataarray(get_version('asfr', 'past')).sel(location_id=location_id).rename('asfr')

    # transform into data frames and normalize
    future = normalize_forecasting(future.to_dataframe().reset_index(), value_column='mean_value')
    past = past.to_dataframe().reset_index()
    past['sex_id'] = 2  # past doesn't have sex_id but future does - plus fertility component in vph filters on sex
    past = normalize_forecasting(past, value_column='mean_value')

    return combine_past_future(past, future)


def get_forecasting_cause_specific_mortality(cause: Cause, location_id: int) -> pd.DataFrame:
    """ Note that forecasting uses acauses for file names so we need to map cause ids to acause names"""
    acause = get_acause_name(cause)
    # read in netcdf files
    future = xr.open_dataarray(f"{get_version('cause_specific_mortality', 'future')}/{acause}.nc")\
        .sel(location_id=location_id, scenario=FORECASTING_SCENARIO)
    past = xr.open_dataset(f"{get_version('cause_specific_mortality', 'past')}/{acause}.nc")\
        .sel(location_id=location_id)

    future = normalize_forecasting(future.to_dataframe().reset_index())
    past = normalize_forecasting(past.to_dataframe().reset_index().drop(['mean', 'acause'], 'columns'))

    return combine_past_future(past, future)


def get_acause_name(cause: Cause):
    df = db_queries.get_cause_metadata(cause_set_id=6, gbd_round_id=4)
    df = df[df.cause_id == cause.gbd_id]
    return df.acause.values[0]


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


def get_version(measure, future_or_past):
    versions = FORECASTING_VERSIONS[measure]
    return str(FBDPath(versions[future_or_past]))


# just patching in forecasting data to replace existing gbd data - we have to work out the full mapping between
# entity keys and forecasting files at some point

# from vivarium_public_health.dataset_manager.artifact import Artifact
# from vivarium_inputs.forecasting_data import get_forecasting_data
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
