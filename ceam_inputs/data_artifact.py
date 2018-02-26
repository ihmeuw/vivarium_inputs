import datetime
import json
import os
from collections import defaultdict
from multiprocessing import cpu_count
from random import shuffle

import numpy as np
import xarray as xr
import pandas as pd

from pathos.multiprocessing import ProcessPool as Pool


from .gbd import get_estimation_years, GBD_ROUND_ID
from .gbd_mapping import causes, risk_factors, sequelae
from ceam_inputs import core
from ceam_inputs.utilities import standardize_dimensions, normalize_for_simulation, standardize_dimensions, get_age_group_midpoint_from_age_group_id

import logging
_log = logging.getLogger(__name__)

def split_entity_path(path):
    entity_path_components = path.split('.')
    if len(entity_path_components) == 2:
        entity_type, entity_name = entity_path_components
    else:
        entity_type = entity_path_components[0]
        entity_name = None
    return entity_type, entity_name

class EntityConfig:
    def __init__(self, entity_type, name, locations, year_start, year_end, modeled_causes, dimensions, entity=None, pool=None):
        self.type = entity_type
        self.name = name
        self.locations = locations
        self.year_start = year_start
        self.year_end = year_end
        self.modeled_causes = modeled_causes
        self.dimensions = dimensions
        self.entity = entity
        self.pool = pool


def normalize(data, dimensions, fills):
    data = normalize_for_simulation(data)
    data = standardize_dimensions(data, dimensions, fills)
    data = get_age_group_midpoint_from_age_group_id(data)
    draw_columns = [c for c in data.columns if 'draw_' in c]
    index_columns = [c for c in data.columns if 'draw_' not in c]
    data = pd.melt(data, id_vars=index_columns, value_vars=draw_columns, var_name='draw')
    data['draw'] = data.draw.str.partition('_')[2].astype(int)
    return data


def load_cause(entity_config):
    measures = ['death', 'prevalence', 'incidence', 'cause_specific_mortality', 'excess_mortality']
    result = core.get_draws([causes[entity_config.name]], measures, entity_config.locations).drop('cause_id', axis=1)

    #Try remission in isolation since it is missing for some models
    try:
        measures = ['remission']
        result = result.append(core.get_draws([causes[entity_config.name]], measures, entity_config.locations).drop('cause_id', axis=1))
    except core.InvalidQueryError:
        pass
    fills = {'death': 0.0, 'prevalence': 0.0, 'incidence': 0.0, 'cause_specific_mortality': 0.0, 'excess_mortality': 0.0, 'remission': 0.0}

    result = normalize(result, entity_config.dimensions, fills)
    dims = ['year', 'sex', 'measure', 'age', 'age_group_start', 'age_group_end', 'location_id', 'draw', 'cause_id']
    result['cause_id'] = causes[entity_config.name].gbd_id
    result = result.set_index(dims)

    return (result,  {})

def load_risk_factor(entity_config):
    risk = risk_factors[entity_config.name]
    result = []

    rrs = core.get_draws([risk], ['relative_risk'], entity_config.locations).drop('risk_id', axis=1)
    normalized = []
    for key, group in rrs.groupby(['parameter', 'cause_id']):
        group = group.drop(['cause_id', 'parameter'], axis=1)
        group = normalize(group, entity_config.dimensions, {'relative_risk': 0.0})
        group['parameter'] = key[0]
        group['cause_id'] = key[1]
        dims = ['year', 'sex', 'measure', 'age', 'age_group_start', 'age_group_end', 'location_id', 'draw', 'cause_id', 'parameter']
        normalized.append(group.set_index(dims))
    rrs = pd.concat(normalized)
    result.append(rrs)

    mfs = core.get_draws([risk], ['mediation_factor'], entity_config.locations).drop('risk_id', axis=1)
    if not mfs.empty:
        # Not all risks have mediation factors
        index_columns = [c for c in mfs.columns if 'draw_' not in c]
        draw_columns = [c for c in mfs.columns if 'draw_' in c]
        mfs = pd.melt(mfs, id_vars=index_columns, value_vars=draw_columns, var_name='draw')
        mfs['draw'] = mfs.draw.str.partition('_')[2].astype(int)
        mfs = mfs.set_index(['measure', 'draw', 'cause_id'])
        result.append(mfs)

    pafs = core.get_draws([risk], ['population_attributable_fraction'], entity_config.locations).drop('risk_id', axis=1)
    normalized = []
    for key, group in pafs.groupby(['cause_id']):
        group = group.drop(['cause_id'], axis=1)
        group = normalize(group, entity_config.dimensions, {'population_attributable_fraction': 0.0})
        group['cause_id'] = key
        dims = ['year', 'sex', 'measure', 'age', 'age_group_start', 'age_group_end', 'location_id', 'draw', 'cause_id']
        normalized.append(group.set_index(dims))
    pafs = pd.concat(normalized)
    result.append(pafs)

    exposures = core.get_draws([risk], ['exposure'], entity_config.locations).drop('risk_id', axis=1)
    normalized = []
    for key, group in exposures.groupby(['parameter']):
        group = group.drop(['parameter'], axis=1)
        group = normalize(group, entity_config.dimensions, {'exposure': 0.0})
        group['parameter'] = key
        dims = ['year', 'sex', 'measure', 'age', 'age_group_start', 'age_group_end', 'location_id', 'draw', 'parameter']
        normalized.append(group.set_index(dims))
    exposures = pd.concat(normalized)
    result.append(exposures)

    if risk.exposure_parameters is not None:
        exposure_stds = core.get_draws([risk], ['exposure_standard_deviation'], entity_config.locations).drop('risk_id', axis=1)
    import pdb; pdb.set_trace()


    result = xr.merge(result)
    result.coords['risk_id'] = risk.gbd_id
    result = result.expand_dims('risk_id')

    return (result,  {})

def load_sequela(entity_config):
    sequela = sequelae[entity_config.name]
    measures = ['prevalence', 'incidence']
    fills = {'prevalence': 0.0, 'incidence': 0.0}
    result = core.get_draws([sequela], measures, entity_config.locations).drop('sequela_id', axis=1)
    result = normalize(result, entity_config.dimensions, fills)

    #TODO cleanup code duplication around this special case (disability weights are constant on all dimensions except draw)
    weights = core.get_draws([sequela], ['disability_weight'], entity_config.locations).drop('sequela_id', axis=1)
    index_columns = [c for c in weights.columns if 'draw_' not in c]
    draw_columns = [c for c in weights.columns if 'draw_' in c]
    weights = pd.melt(weights, id_vars=index_columns, value_vars=draw_columns, var_name='draw')
    weights['draw'] = weights.draw.str.partition('_')[2].astype(int)
    weights = weights.set_index(['measure', 'draw']).to_xarray()

    return (xr.merge([result, weights]),  {})


def load_population(entity_config):
    pop = core.get_populations(entity_config.locations)
    pop = normalize_for_simulation(pop)
    pop = get_age_group_midpoint_from_age_group_id(pop)
    pop = pop.set_index(['location_id', 'year', 'sex', 'age', 'age_group_start', 'age_group_end']).to_xarray()

    return (pop, {})

LOADERS = {
        'cause': load_cause,
        'risk_factor': load_risk_factor,
        'sequela': load_sequela,
        'population': load_population,
        }

def _entities_by_type(entities):
    entity_by_type = defaultdict(set)
    for entity_path in entities:
        entity_type, entity_name = split_entity_path(entity_path)
        entity_by_type[entity_type].add(entity_name)
    return entity_by_type

class ArtifactBuilder:
    def __init__(self, config):
        self.config = config
        self.entities = set()
        self.constructors = {}

    def save(self, path, locations, year_range=None, parallelism=None):
        locations = locations

        if year_range is not None:
            year_start, year_end = year_range
        else:
            estimation_years = get_estimation_years(GBD_ROUND_ID)
            year_start = min(estimation_years)
            year_end = max(estimation_years)

        entity_by_type = _entities_by_type(self.entities)

        age_bins = core.get_age_bins()
        dimensions = pd.MultiIndex.from_product([range(year_start, year_end+1), ['Male', 'Female'], age_bins.age_group_id, locations], names=['year', 'sex', 'age_group_id', 'location_id']) 

        if parallelism is None:
            parallelism = cpu_count()
        finished_count = 0
        with Pool(parallelism) as pool:
            es = list(entity_by_type.items())
            shuffle(es)
            for entity_type, entities in es:
                scalar_measures = {}
                measures = {}
                for entity_name in entities:
                    finished_count += 1
                    _log.info(f'Loading data for {entity_type}.{entity_name} ({finished_count} of {len(self.entities)})')
                    entity_config = EntityConfig(entity_type=entity_type,
                                                 name=entity_name,
                                                 locations=locations,
                                                 year_start=year_start,
                                                 year_end=year_end,
                                                 dimensions = dimensions,
                                                 modeled_causes=entity_by_type['cause'],
                                                 pool=pool)
                    array, scalars = LOADERS[entity_type](entity_config)
                    if array is not None:
                        for variable in array.data_vars:
                            var_data = array[variable]
                            if 'year' in var_data:
                                mask = (var_data.year >= year_start) & (var_data.year <= year_end)
                                array[variable] = var_data.where(mask, drop=True)

                                array[variable] = var_data.where(mask, drop=True)
                        array = array.assign_coords(entity=entity_name if entity_name else entity_type).expand_dims('entity')
                        for measure in array.data_vars:
                            if measure in measures:
                                measures[measure] = xr.concat([measures[measure], array[measure]], dim='entity')
                            else:
                                measures[measure] = array[measure]
                    if scalars is not None:
                        scalar_measures[entity_name] = scalars

                if measures or scalar_measures:
                    data = _pack_measures(measures)
                    _dump(data, scalar_measures, entity_type, path)

    def data_container(self, entity_path: str):
        self.entities.add(entity_path)
        print(entity_path)

def _pack_measures(measures):
    data = xr.Dataset(measures)
    if data.sizes:
        for variable in data.variables:
            if data[variable].dtype == np.dtype('object'):
                # NOTE: There could be a bug that would produce variables with object type which
                # aren't actually strings in which case this is doing the wrong thing.
                mlen = max([len(r) for r in data[variable].values])
                data[variable] = data[variable].astype('<U{}'.format(mlen))
        for dimension in data.dims.keys():
            data = data.dropna(dimension, how='all')

    else:
        data = xr.Dataset()
    return data


def _dump(data, scalar_measures, entity_type, path):
    data.attrs['title'] = 'Input data for CEAM model'
    data.attrs['institution'] = 'Institute for Health Metrics and Evaluation'
    data.attrs['source'] = 'Global Burden of Disease estimates plus other data created by the IHME Cost Effectiveness team'
    data.attrs['history'] = 'Snapshot of GBD and auxiliary data taken by CEAM on {}'.format(datetime.datetime.now().date().isoformat())

    data.attrs['scalar_measures'] = json.dumps(scalar_measures)

    encoding = {var: {'zlib':True, 'complevel':5} for var in data.data_vars}
    if os.path.exists(path):
        mode = 'a'
    else:
        mode = 'w'

    for c in data.coords:
        data[c] = np.atleast_1d(data[c])

    data.to_netcdf(path, encoding=encoding, group='/{}'.format(entity_type), mode=mode)
