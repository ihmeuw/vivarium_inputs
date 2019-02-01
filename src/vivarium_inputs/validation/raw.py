from typing import NamedTuple, Union
import warnings

import pandas as pd
import numpy as np

from gbd_mapping import (ModelableEntity, Cause, Sequela, RiskFactor,
                         Etiology, Covariate, CoverageGap, causes)

from vivarium_inputs.globals import (DRAW_COLUMNS, DEMOGRAPHIC_COLUMNS,
                                     DataAbnormalError, InvalidQueryError, gbd)
from vivarium_inputs.mapping_extension import AlternativeRiskFactor, HealthcareEntity, HealthTechnology

from vivarium_inputs.validation.utilities import (check_years, check_and_replace_location, check_columns, check_data_exist,
                                                  check_age_group_ids, check_sex_ids, check_age_restrictions,
                                                  check_value_columns_boundary, check_sex_restrictions,
                                                  check_measure_id, check_metric_id, get_restriction_age_boundary,
                                                  get_restriction_age_ids)


MAX_INCIDENCE = 10
MAX_REMISSION = 365/3
MAX_CATEG_REL_RISK = 20
MAX_CONT_REL_RISK = 5
MAX_UTILIZATION = 50
MAX_LIFE_EXP = 90
MAX_POP = 100_000_000

ALL_AGES_AGE_GROUP_ID = 22
AGE_STANDARDIZED_AGE_GROUP_ID = 27


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
    if entity.restrictions.yll_only:
        raise NotImplementedError(f"{entity.name} is YLL only cause and we currently do not have a model to support it.")

    _check_cause_age_restrictions(entity)
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
        warnings.warn(f'2017 cost data for {entity.kind} {entity.name} is duplicated from 2016 data, and all data '
                      f'before 1995 is backfilled from 1995 data.')


def check_population_metadata(entity: NamedTuple, measure: str):
    pass


def _validate_incidence(data: pd.DataFrame, entity: Union[Cause, Sequela], location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', 'metric_id', f'{entity.kind}_id'] + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['Incidence'])
    check_metric_id(data, 'rate')

    check_years(data, 'annual')
    data = check_and_replace_location(data, location_id)

    if entity.kind == 'cause':
        restrictions = entity.restrictions
    else:  # sequela
        cause = [c for c in causes if c.sequelae and entity in c.sequelae][0]
        restrictions = cause.restrictions

    check_age_group_ids(data, restrictions.yld_age_group_id_start, restrictions.yld_age_group_id_end)
    # como should return all sexes regardless of restrictions
    check_sex_ids(data, male_expected=True, female_expected=True)

    check_age_restrictions(data, restrictions.yld_age_group_id_start, restrictions.yld_age_group_id_end)
    check_sex_restrictions(data, restrictions.male_only, restrictions.female_only)

    check_value_columns_boundary(data, 0, 'lower', inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, MAX_INCIDENCE, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=None)


def _validate_prevalence(data: pd.DataFrame, entity: Union[Cause, Sequela], location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', 'metric_id', f'{entity.kind}_id'] + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['Prevalence'])
    check_metric_id(data, 'rate')

    check_years(data, 'annual')
    data = check_and_replace_location(data, location_id)

    if entity.kind == 'cause':
        restrictions = entity.restrictions
    else:  # sequela
        cause = [c for c in causes if c.sequelae and entity in c.sequelae][0]
        restrictions = cause.restrictions

    check_age_group_ids(data, restrictions.yld_age_group_id_start, restrictions.yld_age_group_id_end)
    # como should return all sexes regardless of restrictions
    check_sex_ids(data, male_expected=True, female_expected=True)

    check_age_restrictions(data, restrictions.yld_age_group_id_start, restrictions.yld_age_group_id_end)
    check_sex_restrictions(data, restrictions.male_only, restrictions.female_only)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, 1, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)


def _validate_birth_prevalence(data: pd.DataFrame, entity: Union[Cause, Sequela], location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', 'metric_id', f'{entity.kind}_id'] + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['Incidence'])
    check_metric_id(data, 'rate')

    check_years(data, 'annual')
    data = check_and_replace_location(data, location_id)

    birth_age_group_id = 164
    if data.age_group_id.unique() != birth_age_group_id:
        raise DataAbnormalError(f'Birth prevalence data for {entity.kind} {entity.name} includes age groups beyond '
                                f'the expected birth age group (id {birth_age_group_id}).')

    # como should return all sexes regardless of restrictions
    check_sex_ids(data, male_expected=True, female_expected=True)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, 1, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)

    if entity.kind == 'cause':
        check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only)


def _validate_disability_weight(data: pd.DataFrame, entity: Sequela, location_id: int):
    check_data_exist(data, zeros_missing=False)

    expected_columns = ['location_id', 'age_group_id', 'sex_id', 'measure',
                        'healthstate', 'healthstate_id'] + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    data = check_and_replace_location(data, location_id)

    if set(data.age_group_id) != {ALL_AGES_AGE_GROUP_ID}:
        raise DataAbnormalError(f'Disability weight data for {entity.kind} {entity.name} includes age groups beyond '
                                f'the expected all ages age group (id {ALL_AGES_AGE_GROUP_ID}).')

    check_sex_ids(data, male_expected=False, female_expected=False, combined_expected=True)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, 1, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)


def _validate_remission(data: pd.DataFrame, entity: Cause, location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', 'metric_id', 'model_version_id',
                        'modelable_entity_id'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['Remission'])
    check_metric_id(data, 'rate')

    check_years(data, 'binned')
    data = check_and_replace_location(data, location_id)

    restrictions = entity.restrictions

    check_age_group_ids(data, restrictions.yld_age_group_id_start, restrictions.yld_age_group_id_end)

    male_expected = restrictions.male_only or (not restrictions.male_only and not restrictions.female_only)
    female_expected = restrictions.female_only or (not restrictions.male_only and not restrictions.female_only)
    check_sex_ids(data, male_expected, female_expected)

    check_age_restrictions(data, restrictions.yld_age_group_id_start, restrictions.yld_age_group_id_end)
    check_sex_restrictions(data, restrictions.male_only, restrictions.female_only)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, MAX_REMISSION, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=None)


def _validate_deaths(data: pd.DataFrame, entity: Cause, location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', f'{entity.kind}_id', 'metric_id'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['Deaths'])
    check_metric_id(data, 'number')

    check_years(data, 'annual')
    data = check_and_replace_location(data, location_id)

    restrictions = entity.restrictions

    check_age_group_ids(data, restrictions.yll_age_group_id_start, restrictions.yll_age_group_id_end)

    male_expected = restrictions.male_only or (not restrictions.male_only and not restrictions.female_only)
    female_expected = restrictions.female_only or (not restrictions.male_only and not restrictions.female_only)
    check_sex_ids(data, male_expected, female_expected)

    check_age_restrictions(data, restrictions.yll_age_group_id_start, restrictions.yll_age_group_id_end)
    check_sex_restrictions(data, restrictions.male_only, restrictions.female_only)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)
    pop = gbd.get_population(location_id)
    idx_cols = ['age_group_id', 'year_id', 'sex_id']
    pop = pop[(pop.age_group_id.isin(data.age_group_id.unique())) & (pop.year_id.isin(data.year_id.unique())) & (
               pop.sex_id != gbd.COMBINED[0])].set_index(idx_cols).population
    check_value_columns_boundary(data.set_index(idx_cols), pop, 'upper',
                                 value_columns=DRAW_COLUMNS, inclusive=True, error=None)


def _validate_exposure(data: pd.DataFrame, entity: Union[RiskFactor, CoverageGap, AlternativeRiskFactor],
                       location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['rei_id', 'modelable_entity_id', 'parameter',
                        'measure_id', 'metric_id'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data,  ['Prevalence', 'Proportion', 'Continuous'])
    check_metric_id(data, 'rate')

    if not check_years(data, 'annual', error=False) and not check_years(data, 'binned', error=False):
        raise DataAbnormalError(f'Exposure data for {entity.kind} {entity.name} contains a year range '
                                f'that is neither annual nor binned.')
    data = check_and_replace_location(data, location_id)

    cats = data.groupby('parameter')

    if entity.kind == 'risk_factor':
        restrictions = entity.restrictions
        age_start = get_restriction_age_boundary(entity, 'start')
        age_end = get_restriction_age_boundary(entity, 'end')
        male_expected = restrictions.male_only or (not restrictions.male_only and not restrictions.female_only)
        female_expected = restrictions.female_only or (not restrictions.male_only and not restrictions.female_only)

        cats.apply(check_age_group_ids, age_start, age_end)
        cats.apply(check_sex_ids, male_expected, female_expected)

        cats.apply(check_age_restrictions, age_start, age_end)
        cats.apply(check_sex_restrictions, entity.restrictions.male_only, entity.restrictions.female_only)

        # we only have metadata about tmred for risk factors
        if entity.distribution in ('ensemble', 'lognormal', 'normal'):  # continuous
            if entity.tmred.inverted:
                check_value_columns_boundary(data, entity.tmred.max, 'upper',
                                             value_columns=DRAW_COLUMNS, inclusive=True, error=None)
            else:
                check_value_columns_boundary(data, entity.tmred.min, 'lower',
                                             value_columns=DRAW_COLUMNS, inclusive=True, error=None)
    else:
        cats.apply(check_age_group_ids, None, None)
        cats.apply(check_sex_ids, True, True)

    if entity.distribution in ('dichotomous', 'ordered_polytomous', 'unordered_polytomous'):  # categorical
        check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)
        check_value_columns_boundary(data, 1, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)

        g = data.groupby(DEMOGRAPHIC_COLUMNS)[DRAW_COLUMNS].sum()
        if not np.allclose(g, 1.0):
            raise DataAbnormalError(f'Exposure data for {entity.kind} {entity.name} '
                                    f'does not sum to 1 across all categories.')


def _validate_exposure_standard_deviation(data: pd.DataFrame, entity: Union[RiskFactor, AlternativeRiskFactor],
                                          location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['rei_id', 'modelable_entity_id', 'measure_id',
                        'metric_id'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data,  ['Continuous'])
    check_metric_id(data, 'rate')

    if not check_years(data, 'annual', error=False) and not check_years(data, 'binned', error=False):
        raise DataAbnormalError(f'Exposure standard deviation data for {entity.kind} {entity.name} contains '
                                f'a year range that is neither annual nor binned.')
    data = check_and_replace_location(data, location_id)

    if entity.kind == 'risk_factor':
        age_start = get_restriction_age_boundary(entity, 'start')
        age_end = get_restriction_age_boundary(entity, 'end')

        check_age_group_ids(data, age_start, age_end)
        check_sex_ids(data, True, True)

        check_age_restrictions(data, age_start, age_end)
        check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only)
    else:
        check_age_group_ids(data, None, None)
        check_sex_ids(data, True, True)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)


def _validate_exposure_distribution_weights(data: pd.DataFrame, entity: Union[RiskFactor, AlternativeRiskFactor],
                                            location_id: int):
    key_cols = ['rei_id', 'location_id', 'sex_id', 'age_group_id', 'measure']
    distribution_cols = ['exp', 'gamma', 'invgamma', 'llogis', 'gumbel', 'invweibull', 'weibull',
                         'lnorm', 'norm', 'glnorm', 'betasr', 'mgamma', 'mgumbel']

    check_data_exist(data, zeros_missing=True, value_columns=distribution_cols)

    check_columns(key_cols + distribution_cols, data.columns)

    if set(data.measure) != {'ensemble_distribution_weight'}:
        raise DataAbnormalError(f'Exposure distribution weight data for {entity.kind} {entity.name} '
                                f'contains abnormal measure values.')

    data = check_and_replace_location(data, location_id)

    if set(data.age_group_id) != {ALL_AGES_AGE_GROUP_ID}:
        raise DataAbnormalError(f'Exposure distribution weight data for {entity.kind} {entity.name} includes '
                                f'age groups beyond the expected all ages age group (id {ALL_AGES_AGE_GROUP_ID}.')

    check_sex_ids(data, male_expected=False, female_expected=False, combined_expected=True)

    check_value_columns_boundary(data, 0, 'lower', value_columns=distribution_cols,
                                 inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, 1, 'upper', value_columns=distribution_cols,
                                 inclusive=True, error=DataAbnormalError)

    if not np.allclose(data[distribution_cols].sum(axis=1), 1.0):
        raise DataAbnormalError(f'Distribution weights for {entity.kind} {entity.name} do not sum to 1.')


def _validate_relative_risk(data: pd.DataFrame, entity: Union[RiskFactor, CoverageGap], location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['rei_id', 'modelable_entity_id', 'cause_id', 'mortality',
                        'morbidity', 'metric_id', 'parameter'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_metric_id(data, 'rate')

    check_years(data, 'binned')
    data = check_and_replace_location(data, location_id)

    cats = data.groupby('parameter')

    if entity.kind == 'risk_factor':
        restrictions = entity.restrictions
        age_start = get_restriction_age_boundary(entity, 'start')
        age_end = get_restriction_age_boundary(entity, 'end')
        male_expected = restrictions.male_only or (not restrictions.male_only and not restrictions.female_only)
        female_expected = restrictions.female_only or (not restrictions.male_only and not restrictions.female_only)

        cats.apply(check_age_group_ids, age_start, age_end)
        cats.apply(check_sex_ids, male_expected, female_expected)

        cats.apply(check_age_restrictions, age_start, age_end)
        cats.apply(check_sex_restrictions, entity.restrictions.male_only, entity.restrictions.female_only)

    else:  # coverage gap
        cats.apply(check_age_group_ids, None, None)
        cats.apply(check_sex_ids, True, True)

    check_value_columns_boundary(data, 1, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)

    max_val = MAX_CATEG_REL_RISK if entity.distribution in ('ensemble', 'lognormal', 'normal') else MAX_CONT_REL_RISK
    check_value_columns_boundary(data, max_val, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=None)

    for c_id in data.cause_id.unique():
        cause = [c for c in causes if c.gbd_id == c_id][0]
        check_mort_morb_flags(data, cause.restrictions.yld_only, cause.restrictions.yll_only)


def _validate_population_attributable_fraction(data: pd.DataFrame, entity: Union[RiskFactor, Etiology],
                                               location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['metric_id', 'measure_id', 'rei_id', 'cause_id'] + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['YLLs', 'YLDs'], single_only=False)
    check_metric_id(data, 'percent')

    check_years(data, 'annual')
    data = check_and_replace_location(data, location_id)

    if entity.kind == 'risk_factor':
        restrictions_entity = entity
    else:  # etiology
        restrictions_entity = [c for c in causes if c.etiologies and entity in c.etiologies][0]

    restrictions = restrictions_entity.restrictions
    age_start = get_restriction_age_boundary(restrictions_entity, 'start')
    age_end = get_restriction_age_boundary(restrictions_entity, 'end')
    male_expected = restrictions.male_only or (not restrictions.male_only and not restrictions.female_only)
    female_expected = restrictions.female_only or (not restrictions.male_only and not restrictions.female_only)

    check_age_group_ids(data, age_start, age_end)
    check_sex_ids(data, male_expected, female_expected)

    check_age_restrictions(data, age_start, age_end)
    check_sex_restrictions(data, restrictions.male_only, restrictions.female_only)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, 1, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)

    for c_id in data.cause_id:
        cause = [c for c in causes if c.gbd_id == c_id][0]
        if cause.restrictions.yld_only and (data.measure_id == 'YLLs').any():
            raise DataAbnormalError(f'Paf data for {entity.kind} {entity.name} affecting {cause.name} contains yll '
                                    f'values despite the affected entity being restricted to yld only.')
        if cause.restrictions.yll_only and (data.measure_id == 'YLDs').any():
            raise DataAbnormalError(f'Paf data for {entity.kind} {entity.name} affecting {cause.name} contains yld '
                                    f'values despite the affected entity being restricted to yll only.')


def _validate_mediation_factors(data, entity, location_id):
    raise NotImplementedError()


def _validate_estimate(data: pd.DataFrame, entity: Covariate, location_id: int):
    value_columns = ['mean_value', 'upper_value', 'lower_value']

    check_data_exist(data, zeros_missing=False, value_columns=value_columns)

    expected_columns = ['model_version_id', 'covariate_id', 'covariate_name_short', 'location_id',
                        'location_name', 'year_id', 'age_group_id', 'age_group_name', 'sex_id',
                        'sex'] + value_columns
    check_columns(expected_columns, data.columns)

    check_years(data, 'annual')
    data = check_and_replace_location(data, location_id)

    if entity.by_age:
        check_age_group_ids(data, None, None)
    else:
        if not set(data.age_group_id).issubset({ALL_AGES_AGE_GROUP_ID, AGE_STANDARDIZED_AGE_GROUP_ID}):
            raise DataAbnormalError(f'Estimate data for {entity.kind} {entity.name} is not supposed to be by age, '
                                    f'but contains age groups beyond all ages and age standardized.')

    check_sex_ids(data, male_expected=entity.by_sex, female_expected=entity.by_sex,
                  combined_expected=(not entity.by_sex))

    _check_covariate_age_restriction(data, entity.by_age)
    _check_covariate_sex_restriction(data, entity.by_sex)


def _validate_cost(data: pd.DataFrame, entity: Union[HealthcareEntity, HealthTechnology], location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure', entity.kind] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    if set(data.measure) != {'cost'}:
        raise DataAbnormalError(f'Cost data for {entity.kind} {entity.name} contains '
                                f'measures beyond the expected cost.')

    check_years(data, 'annual')
    data = check_and_replace_location(data, location_id)

    if set(data.age_group_id) != {ALL_AGES_AGE_GROUP_ID}:
        raise DataAbnormalError(f'Cost data for {entity.kind} {entity.name} includes age groups beyond '
                                f'the expected all ages age group (id {ALL_AGES_AGE_GROUP_ID}).')

    check_sex_ids(data, male_expected=False, female_expected=False, combined_expected=True)
    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)


def _validate_utilization(data: pd.DataFrame, entity: HealthcareEntity, location_id: int):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', 'metric_id', 'model_version_id',
                        'modelable_entity_id'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['Continuous'])
    check_metric_id(data, 'rate')

    check_years(data, 'binned')
    data = check_and_replace_location(data, location_id)

    check_age_group_ids(data, None, None)
    check_sex_ids(data, male_expected=True, female_expected=True, combined_expected=False)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, MAX_UTILIZATION, 'upper', value_columns=DRAW_COLUMNS,
                                 inclusive=True, error=None)


def _validate_structure(data: pd.DataFrame, entity: NamedTuple, location_id: int):
    check_data_exist(data, zeros_missing=True, value_columns=['population'])

    expected_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id', 'population', 'run_id']
    check_columns(expected_columns, data.columns)

    check_years(data, 'annual')
    data = check_and_replace_location(data, location_id)

    check_age_group_ids(data, None, None)
    check_sex_ids(data, male_expected=True, female_expected=True, combined_expected=True)

    check_value_columns_boundary(data, 0, 'lower', value_columns=['population'],
                                 inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, MAX_POP, 'upper', value_columns=['population'],
                                 inclusive=True, error=DataAbnormalError)


def _validate_theoretical_minimum_risk_life_expectancy(data: pd.DataFrame, entity: NamedTuple, location_id: int):
    check_data_exist(data, zeros_missing=True, value_columns=['life_expectancy'])

    expected_columns = ['age', 'life_expectancy']
    check_columns(expected_columns, data.columns)

    min_age, max_age = 0, 110
    if data.age.min() > min_age or data.age.max() < max_age:
        raise DataAbnormalError('Data does not contain life expectancy values for ages [0, 110].')

    check_value_columns_boundary(data, 0, 'lower', value_columns=['life_expectancy'],
                                 inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, MAX_LIFE_EXP, 'upper', value_columns=['life_expectancy'],
                                 inclusive=True, error=DataAbnormalError)


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


############################
# RAW VALIDATION UTILITIES #
############################

def check_mort_morb_flags(data: pd.DataFrame, yld_only: bool, yll_only: bool):
    """ Verify that no unexpected values or combinations of mortality and
    morbidity flags are found in `data`, given the restrictions of the
    affected entity.

    Parameters
    ----------
    data
        Dataframe containing mortality and morbidity flags for relative risk
        data of `entity` affecting `cause`.
    yld_only
        Boolean indicating whether the affected cause is restricted
        to yld_only.
    yll_only
        Boolean indicating whether the affected cause is restricted
        to yll_only.

    Raises
    -------
    DataAbnormalError
        If any unexpected values or combinations of mortality and morbidity
        flags are found.

    """
    valid_morb_mort_values = {0, 1}
    for m in ['morbidity', 'mortality']:
        if not set(data[m]).issubset(valid_morb_mort_values):
            raise DataAbnormalError(f'Data contains values for {m} outside the expected {valid_morb_mort_values}.')

    base_error_msg = f'Relative risk data includes '

    no_morbidity = data.morbidity == 0
    no_mortality = data.mortality == 0

    morbidity = ~no_morbidity
    mortality = ~no_mortality

    if (no_morbidity & no_mortality).any():
        raise DataAbnormalError(base_error_msg + 'rows with both mortality and morbidity flags set to 0.')

    elif (morbidity & mortality).any():
        if no_morbidity.any() or no_mortality.any():
            raise DataAbnormalError(base_error_msg + 'row with both mortality and morbidity flags set to 1 as well as '
                                                     'rows with only one of the mortality or morbidity flags set to 1.')
    else:
        if morbidity.any() and no_mortality.all() and not yld_only:
            raise DataAbnormalError(base_error_msg + 'only rows with the morbidity flag set to 1 but the affected '
                                                     'entity is not restricted to yld_only.')
        elif mortality.any() and no_morbidity.all() and not yll_only:
            raise DataAbnormalError(base_error_msg + 'only rows with the mortality flag set to 1 but the affected '
                                                     'entity is not restricted to yll_only.')
        elif mortality.any() and morbidity.any() and (yld_only or yll_only):
            raise DataAbnormalError(base_error_msg + f'rows for both morbidity and mortality, but the affected entity '
                                    f'is restricted to {"yll_only" if yll_only else "yld_only"}.')
        else:
            pass


def _check_covariate_sex_restriction(data: pd.DataFrame, by_sex: bool):
    if by_sex and not {gbd.MALE[0], gbd.FEMALE[0]}.issubset(set(data.sex_id)):
        raise DataAbnormalError('Data is supposed to be by sex, but does not contain both male and female data.')
    elif not by_sex and set(data.sex_id) != {gbd.COMBINED[0]}:
        raise DataAbnormalError('Data is not supposed to be separated by sex, but contains sex ids beyond that '
                                'for combined male and female data.')


def _check_covariate_age_restriction(data: pd.DataFrame, by_age: bool):
    if by_age and not set(data.age_group_id).intersection(set(gbd.get_age_group_id())):
        # if we have any of the expected gbd age group ids, restriction is not violated
        raise DataAbnormalError('Data is supposed to be age-separated, but does not contain any GBD age group ids.')
    # if we have any age group ids besides all ages and age standardized, restriction is violated
    if not by_age and bool((set(data.age_group_id) - {ALL_AGES_AGE_GROUP_ID, AGE_STANDARDIZED_AGE_GROUP_ID})):
        raise DataAbnormalError('Data is not supposed to be separated by ages, but contains age groups '
                                'beyond all ages and age standardized.')


def _check_cause_age_restrictions(entity: Cause):
    if entity.restrictions.yld_only or entity.restrictions.yll_only:
        pass
    else:
        yll_ages = get_restriction_age_ids(entity.restrictions.yll_age_group_id_start,
                                           entity.restrictions.yll_age_group_id_end)
        yld_ages = get_restriction_age_ids(entity.restrictions.yld_age_group_id_start,
                                           entity.restrictions.yld_age_group_id_end)
        if set(yll_ages) > set(yld_ages):
            raise NotImplementedError(f'{entity.name} has a broader yll age range than yld age range.'
                                      f' We currently do not support these causes.')
