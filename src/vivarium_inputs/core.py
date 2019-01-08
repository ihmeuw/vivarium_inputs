
import pandas as pd

from vivarium_inputs import utilities, extract
from .globals import InvalidQueryError, gbd, DEMOGRAPHIC_COLUMNS



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
        'exposure': (get_exposure, ('risk', 'coverage_gap')),
        'exposure_standard_deviation': (get_exposure_standard_deviation, ('risk',)),
        'exposure_distribution_weights': (get_exposure_distribution_weights, ('risk',)),
        'relative_risk': (get_relative_risk, ('risk', 'coverage_gap')),
        'population_attributable_fraction': (get_population_attributable_fraction, ('risk', 'coverage_gap', 'etiology')),
        'mediation_factors': (get_mediation_factors, ('risk',)),
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


def get_incidence(entity, location_id):
    if entity.kind == 'cause':
        raise NotImplementedError()
    elif entity.kind == 'sequela':
        data = extract.extract_data(entity, 'incidence', location_id)
        data = utilities.normalize(data, location_id, fill_value=0)
        data = utilities.reshape(data)
        return data


def get_prevalence(entity, location_id):
    if entity.kind == 'cause':
        raise NotImplementedError()
    elif entity.kind == 'sequela':
        data = extract.extract_data(entity, 'prevalence', location_id)
        data = utilities.normalize(data, location_id, fill_value=0)
        data = utilities.reshape(data)
        return data


def get_birth_prevalence(entity, location_id):
    birth_prevalence_age_group = 164
    if entity.kind == 'cause':
        raise NotImplementedError()
    elif entity.kind == 'sequela':
        data = extract.extract_data(entity, 'birth_prevalence', location_id)
        data = data[data.age_group_id == birth_prevalence_age_group]
        data.drop('age_group_id', axis=1, inplace=True)
        data = utilities.normalize(data, location_id, fill_value=0)
        data = utilities.reshape(data, to_keep=('year_id', 'sex_id', 'location_id'))
        return data


def get_disability_weight(entity, location_id):
    if entity.kind == 'cause':
        raise NotImplementedError()
    elif entity.kind == 'sequela':
        data = extract.extract_data(entity, 'disability_weight', location_id)
        data = utilities.normalize(data, location_id)
        data = utilities.reshape(data)
        return data


def get_remission(entity, location_id):
    if entity.kind == 'cause':
        raise NotImplementedError()


def get_cause_specific_mortality(entity, location_id):
    if entity.kind == 'cause':
        raise NotImplementedError()


def get_excess_mortality(entity, location_id):
    if entity.kind == 'cause':
        raise NotImplementedError()


def get_case_fatality(entity, location_id):
    if entity.kind == 'cause':
        raise NotImplementedError()


def get_exposure(entity, location_id):
    if entity.kind == 'risk':
        raise NotImplementedError()
    elif entity.kind == 'coverage_gap':
        raise NotImplementedError()


def get_exposure_standard_deviation(entity, location_id):
    if entity.kind == 'risk':
        raise NotImplementedError()


def get_exposure_distribution_weights(entity, location_id):
    if entity.kind == 'risk':
        raise NotImplementedError()


def get_relative_risk(entity, location_id):
    if entity.kind == 'risk':
        raise NotImplementedError()
    elif entity.kind == 'coverage_gap':
        raise NotImplementedError()


def get_population_attributable_fraction(entity, location_id):
    if entity.kind == 'risk':
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
    if entity.kind == 'risk':
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
    if entity.kind == 'population':
        data = extract.extract_data(entity, 'structure', location_id)
        data = data.drop('run_id', 'columns').rename(columns={'population': 'value'})
        data = utilities.normalize(data, location_id)
        return data


def get_theoretical_minimum_risk_life_expectancy(entity, location_id):
    if entity.kind == 'population':
        data = extract.extract_data(entity, 'theoretical_minimum_risk_life_expectancy', location_id)
        data = data.rename(columns={'age': 'age_group_start', 'life_expectancy': 'value'})
        data['age_group_end'] = data.age_group_start.shift(-1).fillna(125.)
        return data
