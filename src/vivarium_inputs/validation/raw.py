from typing import List
import warnings

import pandas as pd

from vivarium_inputs.globals import (DRAW_COLUMNS, DEMOGRAPHIC_COLUMNS, METRICS,
                                     DataAbnormalError, InvalidQueryError, gbd)


def check_metadata(entity, measure):
    metadata_checkers = {
        'sequela': _check_sequela_metadata,
        'cause': _check_cause_metadata,
        'risk_factor': _check_risk_factor_metadata,
        'etiology': _check_etiology_metadata,
        'covariate': _check_covariate_metadata,
        'coverage_gap': _check_coverage_gap_metadata,
        'health_technology': _check_health_technology_metadata,
        'healthcare_entity': _check_healthcare_entity_metadata,
        'population': _check_population_metadata,
    }

    metadata_checkers[entity.kind](entity, measure)


def validate_raw_data(data, entity, measure, location_id):
    validators = {
        # Cause-like measures
        'incidence': _validate_incidence,
        'prevalence': _validate_prevalence,
        'disability_weight': _validate_disability_weight,
        'remission': _validate_remission,
        'deaths': _validate_deaths,
        # Risk-like measures
        'exposure': _validate_exposure,
        'exposure_standard_deviation': _validate_exposure_standard_deviation,
        'exposure_distribution_weights': _validate_exposure_distribution_weights,
        'relative_risk': _validate_relative_risk,
        'population_attributable_fraction': _validate_population_attributable_fraction,
        'mediation_factors': _validate_mediation_factors,
        # Covariate measures
        'estimate': _validate_estimate,
        # Health system measures
        'cost': _validate_cost,
        'utilization': _validate_utilization,
        # Population measures
        'structure': _validate_structure,
        'theoretical_minimum_risk_life_expectancy': _validate_theoretical_minimum_risk_life_expectancy,
    }

    if measure not in validators:
        raise NotImplementedError()

    validators[measure](data, entity, location_id)


def _check_sequela_metadata(entity, measure):
    if measure in ['incidence', 'prevalence']:
        if not entity[f'{measure}_exists']:
            raise InvalidQueryError(f'{entity.name} does not have {measure} data')
        if not entity[f'{measure}_in_range']:
            warnings.warn(f'{entity.name} has {measure} but its range is abnormal')
    else:  # measure == 'disability_weight
        if not entity.healthstate[f'{measure}_exist']:
            raise InvalidQueryError(f'{entity.name} does not have {measure} data')


def _check_cause_metadata(entity, measure):
    # TODO: Incorporate mapping checks
    pass


def _check_risk_factor_metadata(entity, measure):
    # TODO: Incorporate mapping checks
    pass


def _check_etiology_metadata(entity, measure):
    mapping_measure = 'paf_yld'
    if not entity[f'{mapping_measure}_exists']:
        raise InvalidQueryError(f'{entity.name} does not have {measure} data')
    if not entity[f'{mapping_measure}_in_range']:
        warnings.warn(f'{entity.name} has {measure} but its range is abnormal')


def _check_covariate_metadata(entity, measure):
    if not entity.data_exist:
        raise InvalidQueryError(f'{entity.name} does not have estimate data')

    restrictions = ['sex', 'age']
    for restriction in restrictions:
        if not entity[f'{restriction}_restriction_violated']:
            warnings.warn(f'{entity.name} has {measure} but {restriction}_restriction is violated')


def _check_coverage_gap_metadata(entity, measure):
    raise NotImplementedError()


def _check_health_technology_metadata(entity, measure):
    raise NotImplementedError()


def _check_healthcare_entity_metadata(entity, measure):
    raise NotImplementedError()


def _check_population_metadata(entity, measure):
    pass


def _validate_incidence(data, entity, location_id):
    if data.metric_id.unique() != METRICS['Rate']:
        raise DataAbnormalError('incidence should have only rate (metric_id 3)')
    expected_columns = ('measure_id', 'metric_id', f'{entity.kind}_id') + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')


def _validate_prevalence(data, entity, location_id):
    if data.metric_id.unique() != METRICS['Rate']:
        raise DataAbnormalError('prevalence should have only rate (metric_id 3)')
    expected_columns = ('measure_id', 'metric_id', f'{entity.kind}_id') + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')


def _validate_disability_weight(data, entity, location_id):
    expected_columns = ('location_id', 'age_group_id', 'sex_id', 'measure',
                        'healthstate', 'healthstate_id') + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)


def _validate_remission(data, entity, location_id):
    expected_columns = ('measure_id', 'metric_id', 'model_version_id',
                        'modelable_entity_id') + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'binned')


def _validate_deaths(data, entity, location_id):
    expected_columns = ('measure_id', f'{entity.kind}_id', 'metric_id') + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')


def _validate_exposure(data, entity, location_id):
    expected_columns = ('rei_id', 'modelable_entity_id', 'parameter',
                        'measure_id', 'metric_id') + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'binned')


def _validate_exposure_standard_deviation(data, entity, location_id):
    raise NotImplementedError()


def _validate_exposure_distribution_weights(data, entity, location_id):
    raise NotImplementedError()


def _validate_relative_risk(data, entity, location_id):
    expected_columns = ('rei_id', 'modelable_entity_id', 'cause_id', 'mortality',
                        'morbidity', 'metric_id', 'parameter') + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'binned')


def _validate_population_attributable_fraction(data, entity, location_id):
    expected_columns = ('metric_id', 'measure_id', 'rei_id', 'cause_id') + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')


def _validate_mediation_factors(data, entity, location_id):
    raise NotImplementedError()


def _validate_estimate(data, entity, location_id):
    expected_columns = ['model_version_id', 'covariate_id', 'covariate_name_short', 'location_id',
                        'location_name', 'year_id', 'age_group_id', 'age_group_name', 'sex_id',
                        'sex', 'mean_value', 'lower_value', 'upper_value']
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')


def _validate_cost(data, entity, location_id):
    raise NotImplementedError()


def _validate_utilization(data, entity, location_id):
    raise NotImplementedError()


def _validate_structure(data, entity, location_id):
    expected_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id', 'population', 'run_id']
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')


def _validate_theoretical_minimum_risk_life_expectancy(data, entity, location_id):
    pass


def check_years(df: pd.DataFrame, year_type: str):
    years = {'annual': list(range(1990, 2018)), 'binned': gbd.get_estimation_years()}
    expected_years = years[year_type]
    if set(df.year_id.unique()) < set(expected_years):
        raise DataAbnormalError(f'Data has missing years: {set(expected_years).difference(set(df.year_id.unique()))}')
    # if is it annual, we expect to have extra years from some cases like codcorrect/covariate
    if year_type == 'binned' and set(df.year_id.unique()) > set(expected_years):
        raise DataAbnormalError(f'Data has extra years: {set(df.year_id.unique()).difference(set(expected_years))}')


def check_columns(expected_cols: List, existing_cols: List):
    if set(existing_cols) < set(expected_cols):
        raise DataAbnormalError(f'{set(expected_cols).difference(set(existing_cols))} columns are missing')
    elif set(existing_cols) > set(expected_cols):
        raise DataAbnormalError(f'Data returned extra columns: {set(existing_cols).difference(set(expected_cols))}')
