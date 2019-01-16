from typing import List
import warnings

import pandas as pd
import numpy as np

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
    if measure in ['incidence', 'prevalence', 'birth_prevalence']:
        exists = entity[f'{measure}_exists']
        if not exists:
            warnings.warn(f'{measure.capitalize()} data for sequela {entity.name} may not exist for all locations.')
        if exists and not entity[f'{measure}_in_range']:
            warnings.warn(f'{measure.capitalize()} for sequela {entity.name} may be outside the normal range.')
    else:  # measure == 'disability_weight
        if not entity.healthstate[f'{measure}_exists']:  # this is not location specific so we can actually throw error
            raise InvalidQueryError(f'Sequela {entity.name} does not have {measure} data.')


def _check_cause_metadata(entity, measure):
    exists = entity[f'{measure}_exists']
    if not exists:
        warnings.warn(f'{measure.capitalize()} data for cause {entity.name} may not exist for all locations.')
    if exists and not entity[f'{measure}_in_range']:
        warnings.warn(f"{measure.capitalize()} for cause {entity.name} may be outside the normal range.")

    violated_restrictions = [r.replace('_', ' ').replace(' violated', '') for r in entity.restrictions.violated if measure in r]
    if violated_restrictions:
        warnings.warn(f'Cause {entity.name} {measure} data may violate the following restrictions: '
                      f'{", ".join(violated_restrictions)}.')

    consistent = entity[f"{measure}_consistent"]
    children = "subcauses" if measure == "deaths" else "sequela"

    if consistent is not None and not consistent:
        warnings.warn(f"{measure.capitalize()} data for cause {entity.name} may not exist for {children} in all "
                      f"locations. {children.capitalize()} models may not be consistent with models for this cause.")

    if consistent and not entity[f"{measure}_aggregates"]:
        warnings.warn(f"{children.capitalize()} {measure} data for cause {entity.name} may not correctly "
                      f"aggregate up to the cause level in all locations. {children.capitalize()} models may not "
                      f"be consistent with models for this cause.")


def _check_risk_factor_metadata(entity, measure):
    mapping_names = {'relative_risk': 'rr',
                     'population_attributable_fraction': 'paf',
                     'exposure': 'exposure',
                     'exposure_standard_deviation': 'exposure_sd'}

    if measure == 'population_attributable_fraction':
        paf_types = np.array(['yll', 'yld'])
        missing_pafs = paf_types[[not entity.paf_yll_exists, not entity.paf_yld_exists]]
        if missing_pafs.size:
            warnings.warn(f'{measure.capitalize()} data for {", ".join(missing_pafs)} for risk factor {entity.name}'
                          f'may not exist for all locations.')

        abnormal_range = paf_types[[entity.paf_yll_exists and not entity.paf_yll_in_range,
                                    entity.paf_yld_exists and not entity.paf_yld_in_range]]
        if abnormal_range.size:
            warnings.warn(f'{measure.capitalize()} data for {", ".join(abnormal_range)} for risk factor {entity.name} '
                          f'may be outside expected range [0, 1].')
    else:
        exists = entity[f'{mapping_names[measure]}_exists']
        if exists is not None and not exists:
            warnings.warn(f'{measure.capitalize()} data for risk factor {entity.name} may not exist for all locations.')

        if measure == 'relative_risk' and exists and not entity.rr_in_range:
            warnings.warn(f'{measure.capitalize()} data for risk factor {entity.name} may be outside '
                          f'expected range >1.')

        if measure == 'exposure' and exists and entity.exposure_year_type in ('mix', 'incomplete'):
            warnings.warn(f'{measure.capitalize()} data for risk factor {entity.name} may contain unexpected '
                          f'or missing years.')

    violated_restrictions = [r.replace('_', ' ').replace(' violated', '') for r in entity.restrictions.violated if mapping_names[measure] in r]
    if violated_restrictions:
        warnings.warn(f'Risk factor {entity.name} {measure} data may violate the following restrictions: '
                      f'{", ".join(violated_restrictions)}.')


def _check_etiology_metadata(entity, measure):
    paf_types = np.array(['yll', 'yld'])
    missing_pafs = paf_types[[not entity.paf_yll_exists, not entity.paf_yld_exists]]
    if missing_pafs.size:
        warnings.warn(f'{measure.capitalize()} data for {", ".join(missing_pafs)} for etiology {entity.name}'
                      f'may not exist for all locations.')

    abnormal_range = paf_types[[entity.paf_yll_exists and not entity.paf_yll_in_range,
                                entity.paf_yld_exists and not entity.paf_yld_in_range]]
    if abnormal_range.size:
        warnings.warn(f'{measure.capitalize()} data for {", ".join(abnormal_range)} for etiology {entity.name} '
                      f'may be outside expected range [0, 1].')


def _check_covariate_metadata(entity, measure):
    if not entity.mean_value_exists:
        warnings.warn(f'{measure.capitalize()} data for covariate {entity.name} may not contain'
                      f'mean values for all locations.')

    if not entity.uncertainty_exists:
        warnings.warn(f'{measure.capitalize()} data for covariate {entity.name} may not contain '
                      f'uncertainty values for all locations.')

    violated_restrictions = [f'by {r}' for r in ['sex', 'age'] if entity[f'by_{r}_violated']]
    if violated_restrictions:
        warnings.warn(f'Covariate {entity.name} may violate the following '
                      f'restrictions: {", ".join(violated_restrictions)}.')


def _check_coverage_gap_metadata(entity, measure):
    pass


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
    check_location(data, location_id)


def _validate_prevalence(data, entity, location_id):
    if data.metric_id.unique() != METRICS['Rate']:
        raise DataAbnormalError('prevalence should have only rate (metric_id 3)')
    expected_columns = ('measure_id', 'metric_id', f'{entity.kind}_id') + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')
    check_location(data, location_id)


def _validate_disability_weight(data, entity, location_id):
    expected_columns = ('location_id', 'age_group_id', 'sex_id', 'measure',
                        'healthstate', 'healthstate_id') + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)
    check_location(data, location_id)


def _validate_remission(data, entity, location_id):
    expected_columns = ('measure_id', 'metric_id', 'model_version_id',
                        'modelable_entity_id') + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'binned')
    check_location(data, location_id)


def _validate_deaths(data, entity, location_id):
    expected_columns = ('measure_id', f'{entity.kind}_id', 'metric_id') + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')
    check_location(data, location_id)


def _validate_exposure(data, entity, location_id):

    expected_columns = ('rei_id', 'modelable_entity_id', 'parameter',
                        'measure_id', 'metric_id') + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)
    # we can't check years for coverage_gaps, since it's not consistent.
    if entity.kind == 'risk_factor':
        check_years(data, 'binned')
    check_location(data, location_id)


def _validate_exposure_standard_deviation(data, entity, location_id):
    raise NotImplementedError()


def _validate_exposure_distribution_weights(data, entity, location_id):
    raise NotImplementedError()


def _validate_relative_risk(data, entity, location_id):
    expected_columns = ('rei_id', 'modelable_entity_id', 'cause_id', 'mortality',
                        'morbidity', 'metric_id', 'parameter') + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'binned')
    check_location(data, location_id)


def _validate_population_attributable_fraction(data, entity, location_id):
    expected_columns = ('metric_id', 'measure_id', 'rei_id', 'cause_id') + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')
    check_location(data, location_id)


def _validate_mediation_factors(data, entity, location_id):
    raise NotImplementedError()


def _validate_estimate(data, entity, location_id):
    expected_columns = ['model_version_id', 'covariate_id', 'covariate_name_short', 'location_id',
                        'location_name', 'year_id', 'age_group_id', 'age_group_name', 'sex_id',
                        'sex', 'mean_value', 'lower_value', 'upper_value']
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')
    check_location(data, location_id)


def _validate_cost(data, entity, location_id):
    raise NotImplementedError()


def _validate_utilization(data, entity, location_id):
    raise NotImplementedError()


def _validate_structure(data, entity, location_id):
    expected_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id', 'population', 'run_id']
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')
    check_location(data, location_id)


def _validate_theoretical_minimum_risk_life_expectancy(data, entity, location_id):
    pass


def check_years(df: pd.DataFrame, year_type: str):
    years = {'annual': list(range(1990, 2018)), 'binned': gbd.get_estimation_years()}
    expected_years = years[year_type]
    if set(df.year_id.unique()) < set(expected_years):
        raise DataAbnormalError(f'Data has missing years: {set(expected_years).difference(set(df.year_id.unique()))}.')
    # if is it annual, we expect to have extra years from some cases like codcorrect/covariate
    if year_type == 'binned' and set(df.year_id.unique()) > set(expected_years):
        raise DataAbnormalError(f'Data has extra years: {set(df.year_id.unique()).difference(set(expected_years))}.')


def check_location(data: pd.DataFrame, location_id: str):
    if data.empty:
        raise InvalidQueryError(f'Data does not have location_id {location_id}.')
    if len(data['location_id'].unique()) > 1:
        raise DataAbnormalError(f'Data has extra location ids.')
    data_location_id = data['location_id'].unique()[0]
    if data_location_id not in [1, location_id]:
        raise DataAbnormalError(f'Data called for {location_id} has a location id {data_location_id}.')


def check_columns(expected_cols: List, existing_cols: List):
    if set(existing_cols) < set(expected_cols):
        raise DataAbnormalError(f'Data is missing columns: {set(expected_cols).difference(set(existing_cols))}.')
    elif set(existing_cols) > set(expected_cols):
        raise DataAbnormalError(f'Data returned extra columns: {set(existing_cols).difference(set(expected_cols))}.')
