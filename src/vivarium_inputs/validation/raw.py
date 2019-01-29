from typing import NamedTuple, Union
import warnings

import pandas as pd
import numpy as np

from gbd_mapping import (ModelableEntity, Cause, Sequela, RiskFactor,
                         Etiology, Covariate, CoverageGap)

from vivarium_inputs.globals import (DRAW_COLUMNS, DEMOGRAPHIC_COLUMNS,
                                     DataAbnormalError, InvalidQueryError, gbd)
from vivarium_inputs.mapping_extension import AlternativeRiskFactor, HealthcareEntity, HealthTechnology

from vivarium_inputs.validation.utilities import (check_years, check_location, check_columns, check_data_exist,
                                                  check_age_group_ids, check_sex_ids, check_age_restrictions,
                                                  check_value_columns_boundary, check_sex_restrictions,
                                                  check_measure_id, check_metric_id)


MAX_INCIDENCE = 10
MAX_REMISSION = 365/3
MAX_CATEG_REL_RISK = 15
MAX_CONT_REL_RISK = 5
MAX_UTILIZATION = 20
MAX_LIFE_EXP = 90


def check_metadata(entity: Union[ModelableEntity, NamedTuple], measure: str):
    metadata_checkers = {
        'sequela': check_sequela_metadata,
        'cause': check_cause_metadata,
        'risk_factor': check_risk_factor_metadata,
        'etiology': check_etiology_metadata,
        'covariate': check_covariate_metadata,
        'coverage_gap': check_coverage_gap_metadata,
        'health_technology': check_health_technology_metadata,
        'healthcare_entity': check_healthcare_entity_metadata,
        'population': check_population_metadata,
        'alternative_risk_factor': check_alternative_risk_factor_metadata,
    }

    metadata_checkers[entity.kind](entity, measure)


def validate_raw_data(data, entity, measure, location_id):
    validators = {
        # Cause-like measures
        'incidence': _validate_incidence,
        'prevalence': _validate_prevalence,
        'birth_prevalence': _validate_birth_prevalence,
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


def check_sequela_metadata(entity: Sequela, measure: str):
    if measure in ['incidence', 'prevalence', 'birth_prevalence']:
        _check_exists_in_range(entity, measure)
    else:  # measure == 'disability_weight
        if not entity.healthstate[f'{measure}_exists']:
            # throw warning so won't break if pulled for a cause where not all sequelae may be missing dws
            warnings.warn(f'Sequela {entity.name} does not have {measure} data.')


def check_cause_metadata(entity: Cause, measure: str):
    _check_exists_in_range(entity, measure)

    _warn_violated_restrictions(entity, measure)

    if measure != 'remission':
        consistent = entity[f"{measure}_consistent"]
        children = "subcauses" if measure == "deaths" else "sequela"

        if consistent is not None and not consistent:
            warnings.warn(f"{measure.capitalize()} data for cause {entity.name} may not exist for {children} in all "
                          f"locations. {children.capitalize()} models may not be consistent with models for this cause.")

        if consistent and not entity[f"{measure}_aggregates"]:
            warnings.warn(f"{children.capitalize()} {measure} data for cause {entity.name} may not correctly "
                          f"aggregate up to the cause level in all locations. {children.capitalize()} models may not "
                          f"be consistent with models for this cause.")


def check_risk_factor_metadata(entity: Union[AlternativeRiskFactor, RiskFactor], measure: str):
    if measure in ('exposure_distribution_weights', 'mediation_factors'):
        # we don't have any applicable metadata to check
        return

    if entity.distribution == 'custom':
        raise NotImplementedError('We do not currently support risk factors with custom distributions.')

    if measure == 'population_attributable_fraction':
        _check_paf_types(entity)
    else:
        _check_exists_in_range(entity, measure)

        if measure == 'exposure' and entity.exposure_year_type in ('mix', 'incomplete'):
            warnings.warn(f'{measure.capitalize()} data for risk factor {entity.name} may contain unexpected '
                          f'or missing years.')

    _warn_violated_restrictions(entity, measure)


def check_alternative_risk_factor_metadata(entity: AlternativeRiskFactor, measure: str):
    pass


def check_etiology_metadata(entity: Etiology, measure: str):
    _check_paf_types(entity)


def check_covariate_metadata(entity: Covariate, measure: str):
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


def check_coverage_gap_metadata(entity: CoverageGap, measure: str):
    pass


def check_health_technology_metadata(entity: HealthTechnology, measure: str):
    if measure == 'cost':
        warnings.warn(f'Cost data for {entity.kind} {entity.name} does not vary by year.')


def check_healthcare_entity_metadata(entity: HealthcareEntity, measure: str):
    if measure == 'cost':
        warnings.warn(f'Cost data for {entity.kind} {entity.name} is constantly extrapolated outside of '
                      f'years [1990, 2017].')


def check_population_metadata(entity: NamedTuple, measure: str):
    pass


def _validate_incidence(data: pd.DataFrame, entity: Union[Cause, Sequela], location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', 'metric_id', f'{entity.kind}_id'] + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['incidence'])
    check_metric_id(data, 'rate')

    check_years(data, 'annual')
    check_location(data, location_id)

    if entity.kind == 'cause':
        check_age_group_ids(data, entity.restrictions.yld_age_group_id_start, entity.restrictions.yld_age_group_id_end)
    else:   # sequelae don't have restrictions
        check_age_group_ids(data)

    # como should return all sexes regardless of restrictions
    check_sex_ids(data, male_expected=True, female_expected=True)

    if entity.kind == 'cause':
        check_age_restrictions(data, entity.restrictions.yld_age_group_id_start,
                               entity.restrictions.yld_age_group_id_end)
        check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only)

    check_value_columns_boundary(data, 0, 'lower', inclusive=True, error=True)
    check_value_columns_boundary(data, MAX_INCIDENCE, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=False)


def _validate_prevalence(data: pd.DataFrame, entity: Union[Cause, Sequela], location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', 'metric_id', f'{entity.kind}_id'] + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['prevalence'])
    check_metric_id(data, 'rate')

    check_years(data, 'annual')
    check_location(data, location_id)

    if entity.kind == 'cause':
        check_age_group_ids(data, entity.restrictions.yld_age_group_id_start, entity.restrictions.yld_age_group_id_end)
    else:   # sequelae don't have restrictions
        check_age_group_ids(data)

    # como should all sexes regardless of restrictions
    check_sex_ids(data, male_expected=True, female_expected=True)

    if entity.kind == 'cause':
        check_age_restrictions(data, entity.restrictions.yld_age_group_id_start,
                               entity.restrictions.yld_age_group_id_end)
        check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=True)
    check_value_columns_boundary(data, 1, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=True)


def _validate_birth_prevalence(data: pd.DataFrame, entity: Union[Cause, Sequela], location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', 'metric_id', f'{entity.kind}_id'] + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['incidence'])
    check_metric_id(data, 'rate')

    check_years(data, 'annual')
    check_location(data, location_id)

    birth_age_group_id = 164
    if data.age_group_id.unique() != birth_age_group_id:
        raise DataAbnormalError(f'Birth prevalence data for {entity.kind} {entity.name} includes age groups beyond '
                                f'the expected birth age group (id {birth_age_group_id}).')

    # como should return all sexes regardless of restrictions
    check_sex_ids(data, male_expected=True, female_expected=True)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=True)
    check_value_columns_boundary(data, 1, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=True)

    if entity.kind == 'cause':
        check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only)


def _validate_disability_weight(data: pd.DataFrame, entity: Sequela, location_id: int):
    check_data_exist(data, zeros_missing=False)

    expected_columns = ['location_id', 'age_group_id', 'sex_id', 'measure',
                        'healthstate', 'healthstate_id'] + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_location(data, location_id)

    all_ages_age_group_id = 22
    if set(data.age_group_id) != {all_ages_age_group_id}:
        raise DataAbnormalError(f'Disability weight data for {entity.kind} {entity.name} includes age groups beyond '
                                f'the expected all ages age group (id {all_ages_age_group_id}).')

    check_sex_ids(data, male_expected=False, female_expected=False, combined_expected=True)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=True)
    check_value_columns_boundary(data, 1, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=True)


def _validate_remission(data: pd.DataFrame, entity: Cause, location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', 'metric_id', 'model_version_id',
                        'modelable_entity_id'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['remission'])
    check_metric_id(data, 'rate')

    check_years(data, 'binned')
    check_location(data, location_id)

    restrictions = entity.restrictions

    check_age_group_ids(data, restrictions.yll_age_group_id_start, restrictions.yll_age_group_id_end)

    male_expected = restrictions.male_only or (not restrictions.male_only and not restrictions.female_only)
    female_expected = restrictions.female_only or (not restrictions.male_only and not restrictions.female_only)
    check_sex_ids(data, male_expected, female_expected)

    check_age_restrictions(data, restrictions.yld_age_group_id_start, restrictions.yld_age_group_id_end)
    check_sex_restrictions(data, restrictions.male_only, restrictions.female_only)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=True)
    check_value_columns_boundary(data, MAX_REMISSION, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=False)


def _validate_deaths(data: pd.DataFrame, entity: Cause, location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', f'{entity.kind}_id', 'metric_id'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['deaths'])
    check_metric_id(data, 'number')

    check_years(data, 'annual')
    check_location(data, location_id)

    restrictions = entity.restrictions

    check_age_group_ids(data, restrictions.yll_age_group_id_start, restrictions.yll_age_group_id_end)

    male_expected = restrictions.male_only or (not restrictions.male_only and not restrictions.female_only)
    female_expected = restrictions.female_only or (not restrictions.male_only and not restrictions.female_only)
    check_sex_ids(data, male_expected, female_expected)

    check_age_restrictions(data, restrictions.yll_age_group_id_start, restrictions.yll_age_group_id_end)
    check_sex_restrictions(data, restrictions.male_only, restrictions.female_only)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=True)
    pop = gbd.get_population(location_id)
    idx_cols = ['age_group_id', 'year_id', 'sex_id']
    pop = pop[(pop.year_id.isin(data.year_id.unique())) & (pop.sex_id != gbd.COMBINED[0])].set_index(idx_cols).population
    check_value_columns_boundary(data.set_index(idx_cols), pop, 'upper',
                                 value_columns=DRAW_COLUMNS, inclusive=True, error=False)


def _validate_exposure(data: pd.DataFrame, entity: Union[RiskFactor, CoverageGap], location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['rei_id', 'modelable_entity_id', 'parameter',
                        'measure_id', 'metric_id'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data,  ['prevalence', 'proportion', 'continuous'])
    check_metric_id(data, 'rate')

    if not check_years(data, 'annual', error=False) and not check_years(data, 'binned', error=False):
        raise DataAbnormalError(f'Exposure data for {entity.kind} {entity.name} contains a year range '
                                f'that is neither annual nor binned.')
    check_location(data, location_id)

    cats = data.groupby('parameter')

    if entity.kind == 'risk_factor':
        restrictions = entity.restrictions
        age_start = min(restrictions.yld_age_group_id_start, restrictions.yll_age_group_id_start)
        age_end = max(restrictions.yld_age_group_id_end, restrictions.yll_age_group_id_end)
        male_expected = restrictions.male_only or (not restrictions.male_only and not restrictions.female_only)
        female_expected = restrictions.female_only or (not restrictions.male_only and not restrictions.female_only)

        cats.apply(check_age_group_ids, age_start, age_end)
        cats.apply(check_sex_ids, male_expected, female_expected)

        # we only have metadata about tmred for risk factors
        if entity.distribution in ('ensemble', 'lognormal', 'normal'):  # continuous
            if entity.tmred.inverted:
                check_value_columns_boundary(data, entity.tmred.max, 'upper',
                                             value_columns=DRAW_COLUMNS, inclusive=True, error=False)
            else:
                check_value_columns_boundary(data, entity.tmred.min, 'lower',
                                             value_columns=DRAW_COLUMNS, inclusive=True, error=False)
    else:
        cats.apply(check_age_group_ids, None, None)
        cats.apply(check_sex_ids, True, True)

    if entity.distribution in ('dichotomous', 'ordered_polytomous', 'unordered_polytomous'):  # categorical
        check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=True)
        check_value_columns_boundary(data, 1, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=True)

        g = data.groupby(DEMOGRAPHIC_COLUMNS)[DRAW_COLUMNS].sum()
        if not np.allclose(g, 1.0):
            raise DataAbnormalError(f'Exposure data for {entity.kind} {entity.name} '
                                    f'does not sum to 1 across all categories.')


def _validate_exposure_standard_deviation(data, entity, location_id):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['rei_id', 'modelable_entity_id', 'measure_id',
                        'metric_id'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data,  ['continuous'])
    check_metric_id(data, 'rate')

    if not check_years(data, 'annual', error=False) and not check_years(data, 'binned', error=False):
        raise DataAbnormalError(f'Exposure standard deviation data for {entity.kind} {entity.name} contains '
                                f'a year range that is neither annual nor binned.')
    check_location(data, location_id)

    if entity.kind == 'risk_factor':
        age_start = min(entity.restrictions.yld_age_group_id_start, entity.restrictions.yll_age_group_id_start)
        age_end = max(entity.restrictions.yld_age_group_id_end, entity.restrictions.yll_age_group_id_end)

        check_age_group_ids(data, age_start, age_end)
        check_sex_ids(data, True, True)

        check_age_restrictions(data, age_start, age_end)
        check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only)
    else:
        check_age_group_ids(data, None, None)
        check_sex_ids(data, True, True)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=True)


def _validate_exposure_distribution_weights(data, entity, location_id):
    key_cols = ['rei_id', 'location_id', 'sex_id', 'age_group_id', 'measure']
    distribution_cols = ['exp', 'gamma', 'invgamma', 'llogis', 'gumbel', 'invweibull', 'weibull',
                         'lnorm', 'norm', 'glnorm', 'betasr', 'mgamma', 'mgumbel']

    check_data_exist(data, zeros_missing=True, value_columns=distribution_cols)

    check_columns(key_cols + distribution_cols, data.columns)

    if set(data.measure) != {'ensemble_distribution_weight'}:
        raise DataAbnormalError(f'Exposure distribution weight data for {entity.kind} {entity.name} '
                                f'contains abnormal measure values.')

    check_location(data, location_id)

    all_ages_age_group_id = 22
    if set(data.age_group_id) != {all_ages_age_group_id}:
        raise DataAbnormalError(f'Exposure distribution weight data for {entity.kind} {entity.name} includes '
                                f'age groups beyond the expected all ages age group (id {all_ages_age_group_id}.')

    check_sex_ids(data, male_expected=False, female_expected=False, combined_expected=True)

    check_value_columns_boundary(data, 0, 'lower', value_columns=distribution_cols, inclusive=True, error=True)
    check_value_columns_boundary(data, 1, 'upper', value_columns=distribution_cols, inclusive=True, error=True)

    if not np.allclose(data[distribution_cols].sum(axis=1), 1.0):
        raise DataAbnormalError(f'Distribution weights for {entity.kind} {entity.name} do not sum to 1.')


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


def _validate_cost(data: pd.DataFrame, entity: Union[HealthcareEntity, HealthTechnology], location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure', entity.kind] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    if set(data.measure) != {'cost'}:
        raise DataAbnormalError(f'Cost data for {entity.kind} {entity.name} contains '
                                f'measures beyond the expected cost.')

    check_years(data, 'annual')
    check_location(data, location_id)

    all_ages_age_group_id = 22
    if set(data.age_group_id) != {all_ages_age_group_id}:
        raise DataAbnormalError(f'Cost data for {entity.kind} {entity.name} includes age groups beyond '
                                f'the expected all ages age group (id {all_ages_age_group_id}).')

    check_sex_ids(data, male_expected=False, female_expected=False, combined_expected=True)
    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=True)


def _validate_utilization(data, entity, location_id):
    raise NotImplementedError()


def _validate_structure(data, entity, location_id):
    expected_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id', 'population', 'run_id']
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')
    check_location(data, location_id)


def _validate_theoretical_minimum_risk_life_expectancy(data, entity, location_id):
    pass


############################
# CHECK METADATA UTILITIES #
############################

def _check_exists_in_range(entity: Union[Sequela, Cause, RiskFactor], measure: str):
    exists = entity[f'{measure}_exists']
    if exists is None:
        raise InvalidQueryError(f'{measure.capitalize()} data is not expected to exist '
                                f'for {entity.kind} {entity.name}.')
    if not exists:
        warnings.warn(f'{measure.capitalize()} data for {entity.kind} {entity.name} may not exist for all locations.')
    if f'{measure}_in_range' in entity.__slots__ and exists and not entity[f'{measure}_in_range']:
        warnings.warn(f'{measure.capitalize()} for {entity.kind} {entity.name} may be outside the normal range.')


def _warn_violated_restrictions(entity, measure):
    violated_restrictions = [r.replace(f'by_{measure}', '').replace(measure, '').replace('_', ' ').replace(' violated', '')
                             for r in entity.restrictions.violated if measure in r]
    if violated_restrictions:
        warnings.warn(f'{entity.kind.capitalize()} {entity.name} {measure} data may violate the '
                      f'following restrictions: {", ".join(violated_restrictions)}.')


def _check_paf_types(entity):
    paf_types = np.array(['yll', 'yld'])
    missing_pafs = paf_types[[not entity.population_attributable_fraction_yll_exists,
                              not entity.population_attributable_fraction_yld_exists]]
    if missing_pafs.size:
        warnings.warn(f'Population attributable fraction data for {", ".join(missing_pafs)} for '
                      f'{entity.kind} {entity.name} may not exist for all locations.')

    abnormal_range = paf_types[[entity.population_attributable_fraction_yll_exists
                                and not entity.population_attributable_fraction_yll_in_range,
                                entity.population_attributable_fraction_yld_exists
                                and not entity.population_attributable_fraction_yld_in_range]]
    if abnormal_range.size:
        warnings.warn(f'Population attributable fraction data for {", ".join(abnormal_range)} for '
                      f'{entity.kind} {entity.name} may be outside expected range [0, 1].')

