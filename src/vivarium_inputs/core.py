from collections import namedtuple
from typing import Union

from gbd_mapping import Cause, RiskFactor, Sequela, Covariate
import pandas as pd

from vivarium_inputs import utilities, extract
from .globals import InvalidQueryError, DEMOGRAPHIC_COLUMNS


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
        'exposure': (get_exposure, ('risk_factor', 'coverage_gap')),
        'exposure_standard_deviation': (get_exposure_standard_deviation, ('risk_factor',)),
        'exposure_distribution_weights': (get_exposure_distribution_weights, ('risk_factor',)),
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
    }

    if measure not in measure_handlers:
        raise InvalidQueryError(f'No functions available to pull data for measure {measure}')

    handler, entity_types = measure_handlers[measure]

    if entity.kind not in entity_types:
        raise InvalidQueryError(f'{measure.capitalize()} not available for {entity.kind}')

    location_id = utilities.get_location_id(location)
    data = handler(entity, location_id)
    return data


def get_incidence(entity: Union[Cause, Sequela], location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'incidence', location_id)
    data = utilities.normalize(data, location_id, fill_value=0)
    data = utilities.reshape(data).set_index(list(DEMOGRAPHIC_COLUMNS) + ['draw'])
    prevalence = get_prevalence(entity, location_id).set_index(list(DEMOGRAPHIC_COLUMNS) + ['draw'])
    # Convert from "True incidence" to the incidence rate among susceptibles
    data /= 1 - prevalence
    return data.fillna(0).reset_index()


def get_prevalence(entity: Union[Cause, Sequela], location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'prevalence', location_id)
    data = utilities.normalize(data, location_id, fill_value=0)
    data = utilities.reshape(data)
    return data


def get_birth_prevalence(entity: Union[Cause, Sequela], location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'birth_prevalence', location_id)
    data = data.drop('age_group_id', 'columns')
    data = utilities.normalize(data, location_id, fill_value=0)
    data = utilities.reshape(data, to_keep=('year_id', 'sex_id', 'location_id'))
    return data


def get_disability_weight(entity: Union[Cause, Sequela], location_id: int) -> pd.DataFrame:
    if entity.kind == 'cause':
        partial_weights = []
        for sequela in entity.sequelae:
            p = get_prevalence(sequela, location_id).set_index(list(DEMOGRAPHIC_COLUMNS) + ['draw'])
            d = get_disability_weight(sequela, location_id).set_index(list(DEMOGRAPHIC_COLUMNS) + ['draw'])
            partial_weights.append(p*d)
        data = sum(partial_weights).reset_index()
    else:  # entity.kind == 'sequela'
        data = extract.extract_data(entity, 'disability_weight', location_id)
        data = utilities.normalize(data, location_id)
        data = utilities.reshape(data)
    return data


def get_remission(entity: Cause, location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'remission', location_id)
    data = utilities.normalize(data, location_id, fill_value=0)
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
    data = csmr / prevalence
    return data.reset_index()


def get_case_fatality(entity: Cause, location_id: int):
    raise NotImplementedError()


def _get_deaths(entity: Cause, location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'deaths', location_id)
    data = utilities.normalize(data, location_id, fill_value=0)
    data = utilities.reshape(data)
    return data


def get_exposure(entity, location_id):
    if entity.kind == 'risk_factor':
        data = extract.extract_data(entity, 'exposure', location_id)
        data = utilities.normalize(data, location_id)
        data = utilities.reshape(data, to_keep=list(DEMOGRAPHIC_COLUMNS) + ['parameter'])
    else:
        raise NotImplementedError()
    return data


def get_exposure_standard_deviation(entity, location_id):
    if entity.kind == 'risk_factor':
        raise NotImplementedError()


def get_exposure_distribution_weights(entity, location_id):
    if entity.kind == 'risk_factor':
        raise NotImplementedError()


def get_relative_risk(entity, location_id):
    if entity.kind == 'risk_factor':

    else:
        raise NotImplementedError()


def get_population_attributable_fraction(entity, location_id):
    if entity.kind == 'risk_factor':
        raise NotImplementedError()
    elif entity.kind == 'coverage_gap':
        raise NotImplementedError()
    elif entity.kind == 'etiology':
        data = extract.extract_data(entity, 'population_attributable_fraction', location_id)
        data.drop(['rei_id', 'measure_id', 'metric_id'], axis=1, inplace=True)
        data = utilities.normalize(data, location_id, fill_value=0)
        data['affected_measure'] = 'incidence_rate'
        data = utilities.reshape(data, to_keep=DEMOGRAPHIC_COLUMNS + ('cause_id', 'affected_measure',))
        return data



def get_mediation_factors(entity, location_id):
    if entity.kind == 'risk_factor':
        raise NotImplementedError()


def get_estimate(entity, location_id):
    if entity.kind == 'covariate':
        data = extract.extract_data(entity, 'estimate', location_id)
        data = pd.melt(data, id_vars=DEMOGRAPHIC_COLUMNS,
                       value_vars=['mean_value', 'upper_value', 'lower_value'], var_name='parameter')
        data = utilities.normalize(data, location_id)
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
    data = utilities.normalize(data, location_id)
    return data


def get_theoretical_minimum_risk_life_expectancy(entity, location_id):
    data = extract.extract_data(entity, 'theoretical_minimum_risk_life_expectancy', location_id)
    data = data.rename(columns={'age': 'age_group_start', 'life_expectancy': 'value'})
    data['age_group_end'] = data.age_group_start.shift(-1).fillna(125.)
    return data
