from collections import namedtuple
from typing import Union, NamedTuple

from gbd_mapping import Cause, Sequela, RiskFactor, CoverageGap, Etiology, Covariate, causes
import pandas as pd
import numpy as np

from vivarium_inputs import utilities, extract
from vivarium_inputs.mapping_extension import AlternativeRiskFactor, HealthcareEntity, HealthTechnology

from .globals import InvalidQueryError, DEMOGRAPHIC_COLUMNS, MEASURES

POP = NamedTuple("Population", [('kind', str)])


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
        'population_attributable_fraction': (get_population_attributable_fraction, ('risk_factor', 'etiology')),
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
    if entity.kind == 'cause':
        restrictions_entity = entity
    else:  # sequela
        cause = [c for c in causes if c.sequelae and entity in c.sequelae][0]
        restrictions_entity = cause

    data = utilities.filter_data_by_restrictions(data, restrictions_entity, 'yld')
    data = utilities.normalize(data, fill_value=0)
    data = utilities.reshape(data).set_index(DEMOGRAPHIC_COLUMNS + ['draw'])
    prevalence = get_prevalence(entity, location_id).set_index(DEMOGRAPHIC_COLUMNS + ['draw'])
    # Convert from "True incidence" to the incidence rate among susceptibles
    data /= 1 - prevalence
    return data.fillna(0).reset_index()


def get_prevalence(entity: Union[Cause, Sequela], location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'prevalence', location_id)
    if entity.kind == 'cause':
        restrictions_entity = entity
    else:  # sequela
        cause = [c for c in causes if c.sequelae and entity in c.sequelae][0]
        restrictions_entity = cause

    data = utilities.filter_data_by_restrictions(data, restrictions_entity, 'yld')
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
        data = utilities.get_demographic_dimensions(location_id, draws=True)
        data['value'] = 0.0
        data = data.set_index(DEMOGRAPHIC_COLUMNS + ['draw'])
        if entity.sequelae:
            for sequela in entity.sequelae:
                prevalence = get_prevalence(sequela, location_id).set_index(DEMOGRAPHIC_COLUMNS + ['draw'])
                disability = get_disability_weight(sequela, location_id)
                disability['location_id'] = location_id
                disability = disability.set_index(DEMOGRAPHIC_COLUMNS + ['draw'])
                data += prevalence * disability
        data = data.reset_index()
    else:  # entity.kind == 'sequela'
        data = extract.extract_data(entity, 'disability_weight', location_id)
        data = utilities.normalize(data)
        data = utilities.reshape(data)

    return data


def get_remission(entity: Cause, location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'remission', location_id)

    data = utilities.filter_data_by_restrictions(data, entity, 'yld')
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
    csmr = get_cause_specific_mortality(entity, location_id).set_index(DEMOGRAPHIC_COLUMNS + ['draw'])
    prevalence = get_prevalence(entity, location_id).set_index(DEMOGRAPHIC_COLUMNS + ['draw'])
    data = (csmr / prevalence).fillna(0)
    data = data.replace([np.inf, -np.inf], 0)
    return data.reset_index()


def get_case_fatality(entity: Cause, location_id: int):
    raise NotImplementedError()


def _get_deaths(entity: Cause, location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'deaths', location_id)
    data = utilities.filter_data_by_restrictions(data, entity, 'yll')
    data = utilities.normalize(data, fill_value=0)
    data = utilities.reshape(data)
    return data


def get_exposure(entity: Union[RiskFactor, AlternativeRiskFactor, CoverageGap], location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'exposure', location_id)
    data = data.drop('modelable_entity_id', 'columns')
    data = data.groupby('parameter').apply(lambda df: utilities.normalize(df, fill_value=0))

    if entity.kind == 'risk_factor':
        data = utilities.filter_data_by_restrictions(data, entity, 'outer')

    data = utilities.normalize(data, fill_value=0)
    data = utilities.reshape(data, to_keep=DEMOGRAPHIC_COLUMNS + ['parameter'])
    return data


def get_exposure_standard_deviation(entity: Union[RiskFactor, AlternativeRiskFactor], location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'exposure_standard_deviation', location_id)
    data = data.drop('modelable_entity_id', 'columns')

    if entity.kind == 'risk_factor':
        data = utilities.filter_data_by_restrictions(data, entity, 'outer')

    data = utilities.normalize(data, fill_value=0)
    data = utilities.reshape(data)
    return data


def get_exposure_distribution_weights(entity: Union[RiskFactor, AlternativeRiskFactor], location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'exposure_distribution_weights', location_id)
    data = utilities.normalize(data, fill_value=0)
    distribution_cols = ['exp', 'gamma', 'invgamma', 'llogis', 'gumbel', 'invweibull', 'weibull',
                         'lnorm', 'norm', 'glnorm', 'betasr', 'mgamma', 'mgumbel']
    id_cols = ['rei_id', 'location_id', 'sex_id', 'year_id', 'age_group_id', 'measure']
    data = pd.melt(data, id_vars=id_cols, value_vars=distribution_cols, var_name='parameter')
    return data


def get_relative_risk(entity: Union[RiskFactor, CoverageGap], location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'relative_risk', location_id)
    if entity.kind == 'risk_factor':
        data = utilities.filter_data_by_restrictions(data, entity, 'inner')

        data = utilities.convert_affected_entity(data, 'cause_id')
        morbidity = data.morbidity == 1
        mortality = data.mortality == 1
        data.loc[morbidity & mortality, 'affected_measure'] = 'incidence_rate'
        data.loc[morbidity & ~mortality, 'affected_measure'] = 'incidence_rate'
        data.loc[~morbidity & mortality, 'affected_measure'] = 'excess_mortality'
    else:  # coverage_gap
        data = utilities.convert_affected_entity(data, 'rei_id')
        data['affected_measure'] = 'exposure_parameters'

    result = []
    for affected_entity in data.affected_entity.unique():
        df = data[data.affected_entity == affected_entity]
        df = df.groupby('parameter').apply(lambda d: utilities.normalize(d, fill_value=1))
        result.append(df)
    data = pd.concat(result)
    data = utilities.reshape(data, to_keep=DEMOGRAPHIC_COLUMNS + ['affected_entity', 'affected_measure', 'parameter'])

    if entity.distribution in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']:
        tmrel_cat = sorted(list(entity.categories.to_dict()), key=lambda x: int(x[3:]))[-1]
        if np.allclose(data.loc[data.parameter == tmrel_cat, 'value'], 1.0):
            data.loc[data.parameter == tmrel_cat, 'value'] = 1.0

    return data


def get_population_attributable_fraction(entity: Union[RiskFactor, Etiology], location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'population_attributable_fraction', location_id)

    if entity.kind == 'risk_factor':
        restriction_entity = entity
    else:  # etiology
        cause = [c for c in causes if c.etiologies and entity in c.etiologies][0]
        restriction_entity = cause

    data = utilities.filter_data_by_restrictions(data, restriction_entity, 'inner')

    data = utilities.convert_affected_entity(data, 'cause_id')
    data.loc[data['measure_id'] == MEASURES['YLLs'], 'affected_measure'] = 'excess_mortality'
    data.loc[data['measure_id'] == MEASURES['YLDs'], 'affected_measure'] = 'incidence_rate'
    data = data.groupby('measure_id').apply(lambda df: utilities.normalize(df, fill_value=0))
    data = utilities.reshape(data, to_keep=DEMOGRAPHIC_COLUMNS + ['affected_entity', 'affected_measure'])
    return data


def get_mediation_factors(entity, location_id):
    raise NotImplementedError()


def get_estimate(entity: Covariate, location_id: int) -> pd.DataFrame:
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


def get_cost(entity: Union[HealthcareEntity, HealthTechnology], location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'cost', location_id)
    data = utilities.normalize(data, fill_value=0)
    data = utilities.reshape(data, to_keep=DEMOGRAPHIC_COLUMNS + [entity.kind])
    return data


def get_utilization(entity: HealthcareEntity, location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'utilization', location_id)
    data = utilities.normalize(data, fill_value=0)
    data = utilities.reshape(data)
    return data


def get_structure(entity: POP, location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'structure', location_id)
    data = data.drop('run_id', 'columns').rename(columns={'population': 'value'})
    data = utilities.normalize(data)
    return data


def get_theoretical_minimum_risk_life_expectancy(entity: POP, location_id: int) -> pd.DataFrame:
    data = extract.extract_data(entity, 'theoretical_minimum_risk_life_expectancy', location_id)
    data = data.rename(columns={'age': 'age_group_start', 'life_expectancy': 'value'})
    data['age_group_end'] = data.age_group_start.shift(-1).fillna(125.)
    return data


def get_age_bins(entity: POP, location_id: int) -> pd.DataFrame:
    age_bins = utilities.get_age_bins()[['age_group_name', 'age_group_start', 'age_group_end']]
    return age_bins


def get_demographic_dimensions(entity: POP, location_id: int) -> pd.DataFrame:
    demographic_dimensions = utilities.get_demographic_dimensions(location_id)
    demographic_dimensions = utilities.normalize(demographic_dimensions)
    return demographic_dimensions

