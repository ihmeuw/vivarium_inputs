import datetime
import json
import os
from collections import defaultdict
from multiprocessing import cpu_count
from random import shuffle

import numpy as np
import xarray as xr
import pandas as pd

from multiprocessing import Pool


from .gbd import get_estimation_years, get_covariate_estimates, GBD_ROUND_ID
from .gbd_mapping import causes, risk_factors, sequelae, healthcare_entities, treatment_technologies, coverage_gaps, etiologies, covariates
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
    #data = standardize_dimensions(data, dimensions, fills)
    if 'age_group_id' in data:
        data = get_age_group_midpoint_from_age_group_id(data)
    draw_columns = [c for c in data.columns if 'draw_' in c]
    index_columns = [c for c in data.columns if 'draw_' not in c]
    data = pd.melt(data, id_vars=index_columns, value_vars=draw_columns, var_name='draw')
    data['draw'] = data.draw.str.partition('_')[2].astype(int)
    return data


def load_cause(entity_config, writer):
    measures = ['death', 'prevalence', 'incidence', 'cause_specific_mortality', 'excess_mortality']
    result = core.get_draws([causes[entity_config.name]], measures, entity_config.locations)
    fills = {'death': 0.0, 'prevalence': 0.0, 'incidence': 0.0, 'cause_specific_mortality': 0.0, 'excess_mortality': 0.0, 'remission': 0.0}
    result = normalize(result, entity_config.dimensions, fills)
    for k,g in result.groupby('measure'):
        writer(k, g)
    del result


    try:
        measures = ['remission']
        result = core.get_draws([causes[entity_config.name]], measures, entity_config.locations)
        #result = normalize(result, entity_config.dimensions, {'remission': 0.0})
        result['cause_id'] = causes[entity_config.name].gbd_id
        writer('remission', result)
    except core.InvalidQueryError:
        pass

def load_risk_factor(entity_config, writer):
    if entity_config.name == 'correlations':
        #TODO: weird special case but this groups it with the other risk data which  I think makes sense
        correlations = core.get_risk_correlation_matrix(entity_config.locations)
        writer('correlations', correlations)
        return

    risk = risk_factors[entity_config.name]

    rrs = core.get_draws([risk], ['relative_risk'], entity_config.locations)
    normalized = []
    for key, group in rrs.groupby(['parameter', 'cause_id']):
        group = group.drop(['cause_id', 'parameter'], axis=1)
        group = normalize(group, entity_config.dimensions, {'relative_risk': 0.0})
        group['parameter'] = key[0]
        group['cause_id'] = key[1]
        dims = ['year', 'sex', 'measure', 'age', 'age_group_start', 'age_group_end', 'location_id', 'draw', 'cause_id', 'parameter']
        normalized.append(group.set_index(dims))
    writer('relative_risk', pd.concat(normalized))
    del normalized

    mfs = core.get_draws([risk], ['mediation_factor'], entity_config.locations)
    if not mfs.empty:
        # Not all risks have mediation factors
        index_columns = [c for c in mfs.columns if 'draw_' not in c]
        draw_columns = [c for c in mfs.columns if 'draw_' in c]
        mfs = pd.melt(mfs, id_vars=index_columns, value_vars=draw_columns, var_name='draw')
        mfs['draw'] = mfs.draw.str.partition('_')[2].astype(int)
        writer('mediation_factor', mfs)
        del mfs

    pafs = core.get_draws([risk], ['population_attributable_fraction'], entity_config.locations)
    normalized = []
    for key, group in pafs.groupby(['cause_id']):
        group = group.drop(['cause_id'], axis=1)
        group = normalize(group, entity_config.dimensions, {'population_attributable_fraction': 0.0})
        group['cause_id'] = key
        dims = ['year', 'sex', 'measure', 'age', 'age_group_start', 'age_group_end', 'location_id', 'draw', 'cause_id']
        normalized.append(group.set_index(dims))
    writer('population_attributable_fraction', pd.concat(normalized))
    del normalized

    exposures = core.get_draws([risk], ['exposure'], entity_config.locations)
    normalized = []
    for key, group in exposures.groupby(['parameter']):
        group = group.drop(['parameter'], axis=1)
        group = normalize(group, entity_config.dimensions, {'exposure': 0.0})
        group['parameter'] = key
        dims = ['year', 'sex', 'measure', 'age', 'age_group_start', 'age_group_end', 'location_id', 'draw', 'parameter']
        normalized.append(group.set_index(dims))
    writer('exposure', pd.concat(normalized))
    del normalized

    if risk.exposure_parameters is not None:
        exposure_stds = core.get_draws([risk], ['exposure_standard_deviation'], entity_config.locations)
        exposure_stds = normalize(exposure_stds, entity_config.dimensions, {})
        writer('exposure_stardard_deviation', exposure_stds)

def load_sequela(entity_config, writer):
    sequela = sequelae[entity_config.name]
    measures = ['prevalence', 'incidence']
    fills = {'prevalence': 0.0, 'incidence': 0.0}
    result = core.get_draws([sequela], measures, entity_config.locations).drop('sequela_id', axis=1)
    result = normalize(result, entity_config.dimensions, fills)
    result['sequela_id'] = sequela.gbd_id
    for k,g in result.groupby('measure'):
        writer(k, g)
    del result


    weights = core.get_draws([sequela], ['disability_weight'], entity_config.locations)
    index_columns = [c for c in weights.columns if 'draw_' not in c]
    draw_columns = [c for c in weights.columns if 'draw_' in c]
    weights = pd.melt(weights, id_vars=index_columns, value_vars=draw_columns, var_name='draw')
    weights['draw'] = weights.draw.str.partition('_')[2].astype(int)
    writer('disability_weight', weights)

def load_healthcare_entity(entity_config, writer):
    healthcare_entity = healthcare_entities[entity_config.name]

    cost = core.get_draws([healthcare_entity], ['cost'], entity_config.locations)
    cost = normalize(cost, entity_config.dimensions, {})
    writer('cost', cost)

    annual_visits = core.get_draws([healthcare_entity], ['annual_visits'], entity_config.locations)
    annual_viannual_visits = normalize(annual_visits, entity_config.dimensions, {})
    writer('annual_visits', annual_visits)


def load_treatment_technology(entity_config, writer):
    treatment_technology = treatment_technologies[entity_config.name]

    if treatment_technology.protection:
        try:
            protection = core.get_draws([treatment_technology], ['protection'], entity_config.locations)
            protection = normalize(protection, entity_config.dimensions, {})
            writer('protection', protection)
        except core.DataMissingError:
            pass

    if treatment_technology.relative_risk:
        relative_risk = core.get_draws([treatment_technology], ['relative_risk'], entity_config.locations)
        relative_risk = normalize(relative_risk, entity_config.dimensions, {})
        writer('relative_risk', relative_risk)

    if treatment_technology.exposure:
        try:
            exposure = core.get_draws([treatment_technology], ['exposure'], entity_config.locations)
            exposure = normalize(exposure, entity_config.dimensions, {})
            writer('exposure', exposure)
        except core.DataMissingError:
            pass

    if treatment_technology.cost:
        cost = core.get_draws([treatment_technology], ['cost'], entity_config.locations)
        cost = normalize(cost, entity_config.dimensions, {})
        writer('cost', cost)

def load_coverage_gap(entity_config, writer):
    entity = coverage_gaps[entity_config.name]

    try:
        exposure = core.get_draws([entity], ['exposure'], entity_config.locations)
        exposure = normalize(exposure, entity_config.dimensions, {})
        writer('exposure', exposure)
    except core.InvalidQueryError:
        pass

    mediation_factor = core.get_draws([entity], ['mediation_factor'], entity_config.locations)
    if not mediation_factor.empty:
        #TODO: This should probably be an exception. It looks like James was in the middle of doing better
        # error handling in ceam_inputs.core but hasn't finished yet
        mediation_factor = normalize(mediation_factor, entity_config.dimensions, {})
        writer('mediation_factor', mediation_factor)

    relative_risk = core.get_draws([entity], ['relative_risk'], entity_config.locations)
    relative_risk = normalize(relative_risk, entity_config.dimensions, {})
    writer('relative_risk', relative_risk)


    population_attributable_fraction = core.get_draws([entity], ['population_attributable_fraction'], entity_config.locations)
    population_attributable_fraction = normalize(population_attributable_fraction, entity_config.dimensions, {})
    writer('population_attributable_fraction', population_attributable_fraction)

def load_etiology(entity_config, writer):
    entity = etiologies[entity_config.name]

    population_attributable_fraction = core.get_draws([entity], ['population_attributable_fraction'], entity_config.locations)
    population_attributable_fraction = normalize(population_attributable_fraction, entity_config.dimensions, {})
    writer('population_attributable_fraction', population_attributable_fraction)


def load_population(entity_config, writer):
    pop = core.get_populations(entity_config.locations)
    pop = normalize_for_simulation(pop)
    pop = get_age_group_midpoint_from_age_group_id(pop)
    writer('structure', pop)

    bins = core.get_age_bins()[['age_group_years_start', 'age_group_years_end', 'age_group_name']]
    bins = bins.rename(columns={'age_group_years_start': 'age_group_start', 'age_group_years_end': 'age_group_end'})
    writer('age_bins', bins)

    writer('theoretical_minimum_risk_life_expectancy', core.get_theoretical_minimum_risk_life_expectancy())

def load_covariate(entity_config, writer):
    entity = covariates[entity_config.name]
    estimate = get_covariate_estimates([entity.gbd_id], entity_config.locations)

    if entity is covariates.age_specific_fertility_rate:
        estimate = estimate[['location_id', 'mean_value', 'lower_value', 'upper_value', 'age_group_id', 'sex_id', 'year_id']]
        estimate = get_age_group_midpoint_from_age_group_id(estimate)
        estimate = normalize_for_simulation(estimate)
    elif entity in (covariates.live_births_by_sex, covariates.dtp3_coverage_proportion):
        estimate = estimate[['location_id', 'mean_value', 'lower_value', 'upper_value', 'sex_id', 'year_id']]
        estimate = normalize_for_simulation(estimate)
    writer('estimate', estimate)

def load_subregions(entity_config, writer):
    writer('sub_region_ids', core.get_subregions(entity_config.locations))

LOADERS = {
        'cause': load_cause,
        'risk_factor': load_risk_factor,
        'sequela': load_sequela,
        'population': load_population,
        'healthcare_entity': load_healthcare_entity,
        'treatment_technology': load_treatment_technology,
        'coverage_gap': load_coverage_gap,
        'etiology': load_etiology,
        'covariate': load_covariate,
        'subregions': load_subregions,
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
            jobs = []
            for entity_type, entities in es:
                scalar_measures = {}
                measures = {}
                for entity_name in entities:
                    finished_count += 1
                    entity_config = EntityConfig(entity_type=entity_type,
                                                  name=entity_name,
                                                  locations=locations,
                                                  year_start=year_start,
                                                  year_end=year_end,
                                                  dimensions = dimensions,
                                                  modeled_causes=entity_by_type['cause'],
                                                  pool=None)
                    if parallelism > 1:
                        jobs.append(pool.apply_async(worker, (entity_config, path, LOADERS[entity_type])))
                    else:
                        worker(entity_config, path, LOADERS[entity_type])
            result = [j.get() for j in jobs]

    def data_container(self, entity_path: str):
        self.entities.add(entity_path)
        print(entity_path)

def worker(entity_config, path, loader):
    _log.info(f'Loading data for {entity_config.type}.{entity_config.name}')
    def writer(measure, data):
        if isinstance(data, pd.DataFrame):
                if 'year' in data:
                    data = data.loc[(data.year >= entity_config.year_start) & (data.year <=entity_config.year_end)]
        _dump(data, entity_config.type, entity_config.name, measure, path)
    loader(entity_config, writer)

def _dump(data, entity_type, entity_name, measure, output_dir):
    path_components = [output_dir, entity_type]
    if entity_name:
        path_components.append(entity_name)

    os.makedirs(os.path.join(*path_components), exist_ok=True)
    path = os.path.join(*(path_components + [measure]))
    if isinstance(data, pd.DataFrame):
        data.to_hdf(path + '.hdf', 'data', format="table", compression=0)
    else:
        with open(path + '.json', 'w') as f:
            json.dump(data, f)
