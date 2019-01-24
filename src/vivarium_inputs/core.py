from collections import namedtuple
from typing import Union
import itertools

from gbd_mapping import Cause, RiskFactor, Sequela, Covariate
import pandas as pd
import numpy as np

from vivarium_gbd_access import gbd
from vivarium_inputs import utilities, extract
from vivarium_inputs.globals import InvalidQueryError, DEMOGRAPHIC_COLUMNS, DRAW_COLUMNS


def get_data(entity, measure: str, location: str):
    measure_handlers = {
        # Cause-like measures
        'incidence': (get_incidence, ('cause', 'sequela')),
        'prevalence': (get_prevalence, ('cause', 'sequela')),
        'birth_prevalence': (get_birth_prevalence, ('cause', 'sequela')),
        'disability_weight': (get_disability_weight, ('cause', 'sequela')),
        'remission': (get_remission, ('cause',)),
        'cause_specific_mortality': (get_cause_specific_mortality, ('cause',)),
        'excess_mortality': (get_excess_mortality, ('cause',)),
        'case_fatality': (get_case_fatality, ('cause',)),
        # Risk-like measures
        'exposure': (get_exposure, ('risk_factor', 'coverage_gap', 'alternative_risk_factor',)),
        'exposure_standard_deviation': (get_exposure_standard_deviation, ('risk_factor', 'alternative_risk_factor')),
        'exposure_distribution_weights': (get_exposure_distribution_weights, ('risk_factor', 'alternative_risk_factor')),
        'relative_risk': (get_relative_risk, ('risk_factor', 'coverage_gap')),
        'population_attributable_fraction': (get_population_attributable_fraction, ('risk_factor', 'coverage_gap', 'etiology')),
        'mediation_factors': (get_mediation_factors, ('risk_factor',)),
        # Covariate measures
        'estimate': (get_estimate, ('covariate',)),
        # Health system measures
        'cost': (get_cost, ('healthcare_entity', 'health_technology')),
        'utilization': (get_utilization, ('healthcare_entity',)),
        # Population measures
        'structure': (get_structure, ('population',)),
        'theoretical_minimum_risk_life_expectancy': (get_theoretical_minimum_risk_life_expectancy, ('population',)),
        'age_bins': (get_age_bins, ('population',)),
        'demographic_dimensions': (get_demographic_dimensions, ('population',))
    }

    if measure not in measure_handlers:
        raise InvalidQueryError(f'No functions available to pull data for measure {measure}.')

    handler, entity_types = measure_handlers[measure]

    if entity.kind not in entity_types:
        raise InvalidQueryError(f'{measure.capitalize()} not available for {entity.kind}.')

    location_id = utilities.get_location_id(location)
    data = handler(entity, location_id)
    return data


def get_incidence(entity: Union[Cause, Sequela], location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'incidence', location_id)
    data = utilities.normalize(data, fill_value=0)
    data = utilities.reshape(data).set_index(list(DEMOGRAPHIC_COLUMNS) + ['draw'])
    prevalence = get_prevalence(entity, location_id).set_index(list(DEMOGRAPHIC_COLUMNS) + ['draw'])
    # Convert from "True incidence" to the incidence rate among susceptibles
    data /= 1 - prevalence
    return data.fillna(0).reset_index()


def get_prevalence(entity: Union[Cause, Sequela], location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'prevalence', location_id)
    data = utilities.normalize(data, fill_value=0)
    data = utilities.reshape(data)
    return data


def get_birth_prevalence(entity: Union[Cause, Sequela], location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'birth_prevalence', location_id)
    data = data.drop('age_group_id', 'columns')
    data = utilities.normalize(data, fill_value=0)
    data = utilities.reshape(data, to_keep=('year_id', 'sex_id', 'location_id'))
    return data


def get_disability_weight(entity: Union[Cause, Sequela], location_id: int) -> pd.DataFrame:
    if entity.kind == 'cause':
        if entity.sequelae:
            partial_weights = []
            for sequela in entity.sequelae:
                prevalence = get_prevalence(sequela, location_id).set_index(list(DEMOGRAPHIC_COLUMNS) + ['draw'])
                disability = get_disability_weight(sequela, location_id)
                disability['location_id'] = location_id
                disability = disability.set_index(list(DEMOGRAPHIC_COLUMNS) + ['draw'])
                partial_weights.append(prevalence*disability)
            data = sum(partial_weights).reset_index()
        else:  # assume disability weight is zero if no sequela
            sex_id = [1, 2]
            age_group_id = gbd.get_age_group_id()
            location_id = [location_id]
            year_id = range(1990, 2018)
            draw = range(1000)
            value = [0.0]
            data = pd.DataFrame(data=list(itertools.product(sex_id, age_group_id, location_id, year_id, draw, value)),
                                columns=["sex_id", "age_group_id", "location_id", "year_id", "draw", "value"])
    else:  # entity.kind == 'sequela'
        data = extract.extract_data(entity, 'disability_weight', location_id)
        data = utilities.normalize(data)
        data = utilities.reshape(data)
    return data


def get_remission(entity: Cause, location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'remission', location_id)
    data = utilities.normalize(data, fill_value=0)
    data = utilities.reshape(data)
    return data


def get_cause_specific_mortality(entity: Cause, location_id: int) -> pd.DataFrame:
    deaths = _get_deaths(entity, location_id)
    pop = get_structure(namedtuple('Population', 'kind')('population'), location_id)
    data = deaths.merge(pop, on=DEMOGRAPHIC_COLUMNS)
    data['value'] = data['value_x'] / data['value_y']
    return data.drop(['value_x', 'value_y'], 'columns')


def get_excess_mortality(entity: Cause, location_id: int) -> pd.DataFrame:
    csmr = get_cause_specific_mortality(entity, location_id).set_index(list(DEMOGRAPHIC_COLUMNS) + ['draw'])
    prevalence = get_prevalence(entity, location_id).set_index(list(DEMOGRAPHIC_COLUMNS) + ['draw'])
    data = (csmr / prevalence).fillna(0)
    data = data.replace([np.inf, -np.inf], 0)
    return data.reset_index()


def get_case_fatality(entity: Cause, location_id: int):
    raise NotImplementedError()


def _get_deaths(entity: Cause, location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'deaths', location_id)
    data = utilities.normalize(data, fill_value=0)
    data = utilities.reshape(data)
    return data


def get_exposure(entity, location_id):
    if entity.kind == 'risk_factor' and entity.distribution == 'dichotomous':
        data = extract.extract_data(entity, 'exposure', location_id)
        cat1 = data[data.parameter == 'cat1']
        cat1 = utilities.normalize(cat1, fill_value=0)
        cat2 = cat1.copy()
        cat2['parameter'] = 'cat2'
        cat2[list(DRAW_COLUMNS)] = 1 - cat2[list(DRAW_COLUMNS)]
        data = pd.concat([cat1, cat2], ignore_index=True, sort=True)
        data = utilities.reshape(data, to_keep=list(DEMOGRAPHIC_COLUMNS) + ['parameter'])
    elif entity.kind == 'coverage_gap':
        data = extract.extract_data(entity, 'exposure', location_id)
        cat1 = data[data.parameter == 'cat1']
        cat1 = utilities.normalize(cat1, fill_value=0)
        cat2 = data[data.parameter == 'cat2']
        cat2 = utilities.normalize(cat2, fill_value=1)
        data = pd.concat([cat1, cat2], ignore_index=True)
        data = utilities.reshape(data, to_keep=list(DEMOGRAPHIC_COLUMNS) + ['parameter'])
    elif entity.kind == 'alternative_risk_factor':
        data = extract.extract_data(entity, 'exposure', location_id)
        # FIXME: me_id is usually nan which breaks year interpolation. This is a stupid hack.
        data = data.drop('modelable_entity_id', 'columns')
        data = utilities.normalize(data, fill_value=0)
        data = utilities.reshape(data)
    else:
        raise NotImplementedError()
    return data


def get_exposure_standard_deviation(entity, location_id):
    data = extract.extract_data(entity, 'exposure_standard_deviation', location_id)
    # FIXME: me_id is usually nan which breaks year interpolation. This is a stupid hack.
    data = data.drop('modelable_entity_id', 'columns')
    data = utilities.normalize(data, fill_value=0)
    data = utilities.reshape(data)
    return data


def get_exposure_distribution_weights(entity, location_id):
    data = extract.extract_data(entity, 'exposure_distribution_weights', location_id)
    data = utilities.normalize(data, fill_value=0)
    key_cols = ['location_id', 'sex_id', 'age_group_id', 'year_id']
    distribution_cols = ['exp', 'gamma', 'invgamma', 'llogis', 'gumbel', 'invweibull', 'weibull',
                         'lnorm', 'norm', 'glnorm', 'betasr', 'mgamma', 'mgumbel']
    data = pd.melt(data, id_vars=key_cols, value_vars=distribution_cols, var_name='parameter')
    return data


def get_relative_risk(entity, location_id):
    if entity.kind == 'risk_factor':
        data = extract.extract_data(entity, 'relative_risk', location_id)
        data = utilities.convert_affected_entity(data, 'cause_id')
        data['affected_measure'] = 'incidence_rate'
        data = utilities.normalize(data, fill_value=1)
        data = utilities.reshape(data, to_keep=list(DEMOGRAPHIC_COLUMNS)
                                               + ['affected_entity', 'affected_measure', 'parameter'])
    elif entity.kind == 'coverage_gap':
        data = extract.extract_data(entity, 'relative_risk', location_id)
        data = utilities.convert_affected_entity(data, 'rei_id')
        data['affected_measure'] = 'exposure_parameters'

        # coverage gap might have very weird year_id, so drop it.
        data.drop('year_id', axis=1, inplace=True)
        data = utilities.normalize(data, fill_value=1)
        data = utilities.reshape(data, to_keep=list(DEMOGRAPHIC_COLUMNS)
                                               + ['affected_entity', 'affected_measure', 'parameter'])

    else:
        raise NotImplementedError()
    return data


def get_population_attributable_fraction(entity, location_id):
    if entity.kind in ['risk_factor', 'etiology']:
        data = extract.extract_data(entity, 'population_attributable_fraction', location_id)
        data = utilities.convert_affected_entity(data, 'cause_id')
        data['affected_measure'] = 'incidence_rate'
        data = utilities.normalize(data, fill_value=0)
        data = utilities.reshape(data, to_keep=DEMOGRAPHIC_COLUMNS + ('affected_entity', 'affected_measure',))
    elif entity.kind == 'coverage_gap':
        e = get_exposure(entity, location_id).drop('location_id', axis=1)
        rrs = get_relative_risk(entity, location_id).drop('location_id', axis=1)
        # For rr we know that we have all annual data but for e, who knows what they put as years
        rrs = rrs[rrs.year_id.isin(e.year_id.unique())]
        affected_entities = rrs.affected_entity.unique()
        pafs = []
        for affected_entity in affected_entities:
            paf = utilities.compute_categorical_paf(rrs, e, affected_entity)
            pafs.append(paf)
        data = pd.concat(pafs)
        data['location_id'] = location_id
    else:
        raise NotImplementedError()
    return data


def get_mediation_factors(entity, location_id):
    if entity.kind == 'risk_factor':
        raise NotImplementedError()


def get_estimate(entity, location_id):
    data = extract.extract_data(entity, 'estimate', location_id)

    key_columns = ['location_id', 'year_id']
    if entity.by_age:
        key_columns.append('age_group_id')
    if entity.by_sex:
        key_columns.append('sex_id')

    data = pd.melt(data, id_vars=key_columns,
                   value_vars=['mean_value', 'upper_value', 'lower_value'], var_name='parameter')
    data = utilities.normalize(data)
    return data


def get_cost(entity, location_id):
    if entity.kind == 'healthcare_entity':
        raise NotImplementedError()
    elif entity.kind == 'health_technology':
        raise NotImplementedError()


def get_utilization(entity, location_id):
    if entity.kind == 'healthcare_entity':
        raise NotImplementedError()


def get_structure(entity, location_id):
    data = extract.extract_data(entity, 'structure', location_id)
    data = data.drop('run_id', 'columns').rename(columns={'population': 'value'})
    data = utilities.normalize(data)
    return data


def get_theoretical_minimum_risk_life_expectancy(entity, location_id):
    data = extract.extract_data(entity, 'theoretical_minimum_risk_life_expectancy', location_id)
    data = data.rename(columns={'age': 'age_group_start', 'life_expectancy': 'value'})
    data['age_group_end'] = data.age_group_start.shift(-1).fillna(125.)
    return data


def get_age_bins(entity, location_id):
    age_bins = utilities.get_age_bins()[['age_group_name', 'age_group_start', 'age_group_end']]
    return age_bins


def get_demographic_dimensions(entity, location_id):
    demographic_dimensions = utilities.get_demographic_dimensions(location_id)
    demographic_dimensions = utilities.normalize(demographic_dimensions)
    return demographic_dimensions
