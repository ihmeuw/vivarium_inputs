from typing import Union, List, Tuple, Set
import operator

import pandas as pd
import numpy as np
from loguru import logger

from gbd_mapping import (ModelableEntity, Cause, Sequela, RiskFactor, Etiology, Covariate, causes)

from vivarium_inputs import utility_data
from vivarium_inputs.globals import (DRAW_COLUMNS, DEMOGRAPHIC_COLUMNS, SEXES, SPECIAL_AGES, METRICS, MEASURES,
                                     PROTECTIVE_CAUSE_RISK_PAIRS, DataAbnormalError, InvalidQueryError,
                                     DataDoesNotExistError, Population, PROBLEMATIC_RISKS, PAF_OUTSIDE_AGE_RESTRICTIONS,
                                     EXCLUDE_ABNORMAL_DATA, RISKS_WITH_NEGATIVE_PAF)

from vivarium_inputs.mapping_extension import AlternativeRiskFactor, HealthcareEntity, HealthTechnology
from vivarium_inputs.utilities import get_restriction_age_ids, get_restriction_age_boundary
from vivarium_inputs.validation.shared import check_value_columns_boundary


MAX_INCIDENCE = 10
MAX_REMISSION = 365/3
MAX_CATEG_REL_RISK = 20
MAX_CONT_REL_RISK = 10
MAX_PAF = 1
MIN_PAF = 0
MIN_PROTECTIVE_PAF = -1
MAX_UTILIZATION = 50
MAX_LIFE_EXP = 90
MAX_POP = 145_000_000  # pop data includes both sexes combined at this point


class RawValidationContext:
    def __init__(self, location_id, **additional_data):
        self.context_data = {'location_id': location_id}
        self.context_data.update(additional_data)

        if 'estimation_years' not in self.context_data:
            self.context_data['estimation_years'] = utility_data.get_estimation_years()
        if 'age_group_ids' not in self.context_data:
            self.context_data['age_group_ids'] = utility_data.get_age_group_ids()
        if 'sexes' not in self.context_data:
            self.context_data['sexes'] = SEXES
        if 'parent_locations' not in self.context_data:
            self.context_data['parent_locations'] = utility_data.get_location_id_parents(location_id)

    def __getitem__(self, key):
        return self.context_data[key]

    def __setitem__(self, key, value):
        self.context_data[key] = value


def do_nothing(ignore_1, ignore_2):
    pass


def check_metadata(entity: ModelableEntity, measure: str) -> None:
    """ Check metadata associated with the given entity and measure for any
    relevant warnings or errors.

    Check that the 'exists' flag in metadata corresponding to `measure` is
    True and that the corresponding 'in_range' flag is also True. Warn if
    either is False.

    If the `entity` has any violated restrictions pertaining to `measure`
    listed in metadata, warn about them.

    Almost all checks result in warnings rather than errors because most flags
    are based on a survey done on data from a single location.

    Parameters
    ----------
    entity
        Entity for which to check metadata.
    measure
        Measure for which to check metadata.

    Raises
    -------
    InvalidQueryError
        If a measure is requested for an entity for which that measure is not
        expected to exist.

    """
    metadata_checkers = {
        'sequela': do_nothing,
        'cause': check_cause_metadata,
        'risk_factor': do_nothing,
        'etiology': do_nothing,
        'covariate': do_nothing,
        'health_technology': check_health_technology_metadata,
        'healthcare_entity': check_healthcare_entity_metadata,
        'population': do_nothing,
        'alternative_risk_factor': do_nothing,
    }
    if entity.kind not in metadata_checkers:
        raise InvalidQueryError(f'No metadata checker found for {entity.kind}.')

    metadata_checkers[entity.kind](entity, measure)


def validate_raw_data(data: pd.DataFrame, entity: ModelableEntity,
                      measure: str, location_id: int, **additional_data) -> None:
    """Validate data conforms to the format expected from raw GBD data, that all
    values are within expected ranges,

    The following checks are performed for each entity-measure pair (some may
    be excluded for certain pairs if not applicable):

    1. Verify data exist.
    2. Verify all expected columns and only expected columns are present.
    3. Verify measure_id, metric_id, year, and location columns contain only
        expected values.
    4. Verify expected age and sex ids are present in data, based on data
        source and entity type.
    5. Verify age and sex restrictions for entity match values in data.
    6. Verify values in value columns are within expected ranges.
    7. Any entity-measure specific checks.

    Verifications that do not pass result in errors or warnings, depending on
    the entity, measure, and verification.

    Parameters
    ----------
    data
        Data to be validated.
    entity
        Entity to which the data belong.
    measure
        Measure to which the data pertain.
    location_id
        Location for which the data were pulled.
    additional_data
        Any additional data needed to validate the measure-entity data. This
        most often applies to RiskFactor data where data from an additional
        measure are often required to validate the necessary extents of the
        data.

    Raises
    -------
    DataAbnormalError
        If critical verifications (e.g., data exist, expected columns are all
        present) fail.

    InvalidQueryError
        If an unknown measure is requested for which no validator exists.

    """
    validators = {
        # Cause-like measures
        'incidence_rate': validate_incidence_rate,
        'prevalence': validate_prevalence,
        'birth_prevalence': validate_birth_prevalence,
        'disability_weight': validate_disability_weight,
        'remission_rate': validate_remission_rate,
        'deaths': validate_deaths,
        # Risk-like measures
        'exposure': validate_exposure,
        'exposure_standard_deviation': validate_exposure_standard_deviation,
        'exposure_distribution_weights': validate_exposure_distribution_weights,
        'relative_risk': validate_relative_risk,
        'population_attributable_fraction': validate_population_attributable_fraction,
        'etiology_population_attributable_fraction': validate_etiology_population_attributable_fraction,
        'mediation_factors': validate_mediation_factors,
        # Covariate measures
        'estimate': validate_estimate,
        # Health system measures
        'cost': validate_cost,
        'utilization_rate': validate_utilization_rate,
        # Population measures
        'structure': validate_structure,
        'theoretical_minimum_risk_life_expectancy': validate_theoretical_minimum_risk_life_expectancy,
    }

    if measure not in validators:
        raise InvalidQueryError(f'No raw validator found for {measure}.')

    context = RawValidationContext(location_id, **additional_data)

    validators[measure](data, entity, context)


##############################################
#   CHECK METADATA ENTITY SPECIFIC METHODS   #
# ------------------------------------------ #
# Signatures to match wrapper check_metadata #
##############################################


def check_cause_metadata(entity: Cause, measure: str) -> None:
    """Check all relevant metadata flags for cause pertaining to measure.

    If the entity is restricted to YLL only or the age group set corresponding
    to the YLL restrictions is greater than that corresponding to the YLD
    restrictions, error as we don't currently know how to model such causes.

    For all measures except remission, check the `consistent` and `aggregates`
    flags for measure, which indicate whether the data was found to
    exist/not exist consistently with any subcauses or sequela and whether
    the estimates for the subcauses/sequela aggregate were found to correctly
    aggregate to the `entity` estimates. Warn if either are False.

    Parameters
    ----------
    entity
        Cause for which to check metadata.
    measure
        Measure for which to check metadata.

    Raises
    ------
    NotImplementedError
        If the `entity` is YLL only or the YLL age range is broader than the
        YLD age range.

    InvalidQueryError
        If the 'exists' metadata flag on `entity` for `measure` is None.

    """
    if entity.restrictions.yll_only:
        raise NotImplementedError(f"{entity.name.capitalize()} is YLL only cause, and we currently do not have a"
                                  f" model to support such a cause.")

    check_cause_age_restrictions_sets(entity)


def check_health_technology_metadata(entity: HealthTechnology, measure: str) -> None:
    """ Because HealthTechnology does not contain any metadata flags, this
    check simply warns the user that cost data is constant over years.

    Parameters
    ----------
    entity
        HealthTechnology for which to check metadata.
    measure
        Measure for which to check metadata.

    """
    if measure == 'cost':
        logger.warning(f'Cost data for {entity.kind} {entity.name} does not vary by year.')


def check_healthcare_entity_metadata(entity: HealthcareEntity, measure: str) -> None:
    """ Because HealthCareEntity does not contain any metadata flags, this
    check simply warns the user that cost data outside of years between
    [1995, 2016] has been duplicated from the nearest year for which there is
    data.

    Parameters
    ----------
    entity
        HealthEntity for which to check metadata.
    measure
        Measure for which to check metadata.

    """
    if measure == 'cost':
        logger.warning(f'2017 cost data for {entity.kind} {entity.name} is duplicated from 2016 data, and all data '
                       f'before 1995 is backfilled from 1995 data.')

#################################################
#   VALIDATE RAW DATA ENTITY SPECIFIC METHODS   #
# --------------------------------------------- #
# Signatures to match wrapper validate_raw_data #
#################################################


def validate_incidence_rate(data: pd.DataFrame, entity: Union[Cause, Sequela], context: RawValidationContext) -> None:
    """ Check the standard set of validations on raw incidence data for entity.

    Parameters
    ----------
    data
        Incidence data pulled for entity in location_id.
    entity
        Cause or sequela to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data, or
        any values in columns do not match the expected set of values.

    """
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', 'metric_id', f'{entity.kind}_id'] + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['Incidence rate'])
    check_metric_id(data, 'rate')

    check_years(data, context, 'annual')
    check_location(data, context)

    if entity.kind == 'cause':
        restrictions = entity.restrictions
    else:  # sequela
        cause = [c for c in causes if c.sequelae and entity in c.sequelae][0]
        restrictions = cause.restrictions

    check_age_group_ids(data, context, restrictions.yld_age_group_id_start, restrictions.yld_age_group_id_end)
    # como should return all sexes regardless of restrictions
    check_sex_ids(data, context, male_expected=True, female_expected=True)

    check_age_restrictions(data, context, restrictions.yld_age_group_id_start, restrictions.yld_age_group_id_end)
    check_sex_restrictions(data, context, restrictions.male_only, restrictions.female_only)

    check_value_columns_boundary(data, 0, 'lower', inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, MAX_INCIDENCE, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=None)


def validate_prevalence(data: pd.DataFrame, entity: Union[Cause, Sequela], context: RawValidationContext) -> None:
    """ Check the standard set of validations on raw prevalence data for entity.

    Parameters
    ----------
    data
        Prevalence data pulled for entity in location_id.
    entity
        Cause or sequela to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data, or
        any values in columns do not match the expected set of values.

    """
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', 'metric_id', f'{entity.kind}_id'] + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['Prevalence'])
    check_metric_id(data, 'rate')

    check_years(data, context, 'annual')
    check_location(data, context)

    if entity.kind == 'cause':
        restrictions = entity.restrictions
    else:  # sequela
        cause = [c for c in causes if c.sequelae and entity in c.sequelae][0]
        restrictions = cause.restrictions

    check_age_group_ids(data, context, restrictions.yld_age_group_id_start, restrictions.yld_age_group_id_end)
    # como should return all sexes regardless of restrictions
    check_sex_ids(data, context, male_expected=True, female_expected=True)

    check_age_restrictions(data, context, restrictions.yld_age_group_id_start, restrictions.yld_age_group_id_end)
    if not EXCLUDE_ABNORMAL_DATA(entity, context):
        check_sex_restrictions(data, context, restrictions.male_only, restrictions.female_only)
    else:
        logger.warning(
            f'In validate_prevalence() -- skipping check_sex_restrictions() due to data issues for {entity.name}'
        )

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, 1, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)


def validate_birth_prevalence(data: pd.DataFrame, entity: Union[Cause, Sequela], context: RawValidationContext) -> None:
    """ Check the standard set of validations on raw birth prevalence data for
    entity, replacing the standard age id checks with a custom check of the
    birth age group.

    Parameters
    ----------
    data
        Birth prevalence data pulled for entity in location_id.
    entity
        Cause or sequela to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data, or
        any values in columns do not match the expected set of values.

    """
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', 'metric_id', f'{entity.kind}_id'] + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['Incidence rate'])
    check_metric_id(data, 'rate')

    check_years(data, context, 'annual')
    check_location(data, context)

    birth_age_group_id = 164
    if data.age_group_id.unique() != birth_age_group_id:
        raise DataAbnormalError(f'Birth prevalence data for {entity.kind} {entity.name} includes age groups beyond '
                                f'the expected birth age group (id {birth_age_group_id}).')

    # como should return all sexes regardless of restrictions
    check_sex_ids(data, context, male_expected=True, female_expected=True)

    if entity.kind == 'cause':
        check_sex_restrictions(data, context, entity.restrictions.male_only, entity.restrictions.female_only)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, 1, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)


def validate_disability_weight(data: pd.DataFrame, entity: Sequela, context: RawValidationContext) -> None:
    """ Check the standard set of validations on raw disability weight data
    for entity, replacing the age ids check with a custom check for the
    all ages age group since disability weights are not age specific.

    Parameters
    ----------
    data
        Disability weight data pulled for entity in location_id.
    entity
        Cause or sequela to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data, or
        any values in columns do not match the expected set of values.

    """
    check_data_exist(data, zeros_missing=False)

    expected_columns = ['location_id', 'age_group_id', 'sex_id', 'measure',
                        'healthstate', 'healthstate_id'] + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_location(data, context)

    if set(data.age_group_id) != {SPECIAL_AGES['all_ages']}:
        raise DataAbnormalError(f'Disability weight data for {entity.kind} {entity.name} includes age groups beyond '
                                f'the expected all ages age group (id {SPECIAL_AGES["all_ages"]}).')

    check_sex_ids(data, context, male_expected=False, female_expected=False, combined_expected=True)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, 1, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)


def validate_remission_rate(data: pd.DataFrame, entity: Cause, context: RawValidationContext) -> None:
    """ Check the standard set of validations on raw remission data for entity.

    Parameters
    ----------
    data
        Remission data pulled for entity in location_id.
    entity
        Cause to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data, or
        any values in columns do not match the expected set of values.

    """
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', 'metric_id', 'model_version_id',
                        'modelable_entity_id'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['Remission rate'])
    check_metric_id(data, 'rate')

    check_years(data, context, 'binned')
    check_location(data, context)

    restrictions = entity.restrictions

    check_age_group_ids(data, context, restrictions.yld_age_group_id_start, restrictions.yld_age_group_id_end)

    male_expected = not restrictions.female_only
    female_expected = not restrictions.male_only
    check_sex_ids(data, context, male_expected, female_expected)

    check_age_restrictions(data, context, restrictions.yld_age_group_id_start, restrictions.yld_age_group_id_end)
    check_sex_restrictions(data, context, restrictions.male_only, restrictions.female_only)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, MAX_REMISSION, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=None)


def validate_deaths(data: pd.DataFrame, entity: Cause, context: RawValidationContext) -> None:
    """ Check the standard set of validations on raw deaths data for entity,
    pulling population data for location_id to use as the upper boundary
    for values in deaths.

    Parameters
    ----------
    data
        Deaths data pulled for entity in location_id.
    entity
        Cause to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data, or
        any values in columns do not match the expected set of values.

    """
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', f'{entity.kind}_id', 'metric_id'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['Deaths'])
    check_metric_id(data, 'number')

    check_years(data, context, 'annual')
    check_location(data, context)

    restrictions = entity.restrictions

    check_age_group_ids(data, context, restrictions.yll_age_group_id_start, restrictions.yll_age_group_id_end)

    male_expected = not restrictions.female_only
    female_expected = not restrictions.male_only
    check_sex_ids(data, context, male_expected, female_expected)

    check_age_restrictions(data, context, restrictions.yll_age_group_id_start, restrictions.yll_age_group_id_end)
    check_sex_restrictions(data, context, restrictions.male_only, restrictions.female_only)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)
    idx_cols = ['age_group_id', 'year_id', 'sex_id']
    population = context['population']
    population = population[(population.age_group_id.isin(data.age_group_id.unique()))
                            & (population.year_id.isin(data.year_id.unique()))
                            & (population.sex_id != context['sexes']['Combined'])
                            & (population.sex_id.isin(data.sex_id.unique()))].set_index(idx_cols).population
    check_value_columns_boundary(data.set_index(idx_cols), population, 'upper',
                                 value_columns=DRAW_COLUMNS, inclusive=True, error=None)


def validate_exposure(data: pd.DataFrame, entity: Union[RiskFactor, AlternativeRiskFactor],
                      context: RawValidationContext) -> None:
    """Check the standard set of validations on raw exposure data for entity.
    Check age group and sex ids and restrictions for each category individually
    for risk factors and alternative risk factors and all together for coverage
    gaps. For risk factors and alternative risk factors, check age restrictions
    but only warn if the data is missing or has extra age groups since
    the restrictions are really about a risk-cause pair. Check draw column
    value boundaries based on distribution type and verify that exposure sums
    to 1 over demographic groups for categorical entities.

    Parameters
    ----------
    data
        Exposure data for `entity` in location `location_id`.
    entity
        Risk factor, coverage gap, or alternative risk factor to which the
        data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data,
        any values in columns do not match the expected set of values,
        or values do not sum to 1 across demographic groups for a categorical
        entity.

    """
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['rei_id', 'modelable_entity_id', 'parameter',
                        'measure_id', 'metric_id'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data,  ['Prevalence', 'Proportion', 'Continuous'])
    check_metric_id(data, 'rate')

    check_years(data, context, 'either')
    check_location(data, context)

    if entity.kind in ['risk_factor', 'alternative_risk_factor']:
        restrictions = entity.restrictions
        age_start = get_restriction_age_boundary(entity, 'start')
        age_end = get_restriction_age_boundary(entity, 'end')
        male_expected = not restrictions.female_only
        female_expected = not restrictions.male_only

        check_age_group_ids(data, context, None, None)
        check_sex_ids(data, context, male_expected, female_expected)

        check_age_restrictions(data, context, age_start, age_end, error=False)
        check_sex_restrictions(data, context, entity.restrictions.male_only, entity.restrictions.female_only)

        # we only have metadata about tmred for risk factors
        if (entity.kind == 'risk_factor' and entity.distribution in ('ensemble', 'lognormal', 'normal')
                and entity.tmred.distribution == 'uniform'):  # continuous risk w/ tmred min and max
            tmrel = (entity.tmred.max + entity.tmred.min)/2
            if entity.tmred.inverted:
                check_value_columns_boundary(data, tmrel, 'upper',
                                             value_columns=DRAW_COLUMNS, inclusive=True, error=None)
            else:
                check_value_columns_boundary(data, tmrel, 'lower',
                                             value_columns=DRAW_COLUMNS, inclusive=True, error=None)
    else:  # CoverageGap
        check_age_group_ids(data, context, None, None)
        check_sex_ids(data, context, True, True)

    if entity.distribution in ('dichotomous', 'ordered_polytomous', 'unordered_polytomous'):  # categorical
        check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS,
                                     inclusive=True, error=DataAbnormalError)
        check_value_columns_boundary(data, 1, 'upper', value_columns=DRAW_COLUMNS,
                                     inclusive=True, error=DataAbnormalError)

        g = data.groupby(DEMOGRAPHIC_COLUMNS)[DRAW_COLUMNS].sum()
        if not np.allclose(g, 1.0):
            msg = (f'Exposure data for {entity.kind} {entity.name} '
                    f'does not sum to 1 across all categories.')
            logger.warning(msg)



def validate_exposure_standard_deviation(data: pd.DataFrame, entity: Union[RiskFactor, AlternativeRiskFactor],
                                         context: RawValidationContext) -> None:
    """Check the standard set of validations on raw exposure standard
    deviation data for entity. Check that the data exist for age groups where
    we have exposure data. Use the age groups from the corresponding
    exposure data as the boundaries for age group checks. Skip age restriction
    checks as risk factor age restrictions don't correspond to this data.

    Parameters
    ----------
    data
        Exposure standard deviation data for `entity` in location `location_id`.
    entity
        Risk factor or alternative risk factor to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data, or
        any values in columns do not match the expected set of values.

    """
    exposure_age_groups = set(context['exposure'].age_group_id)
    valid_age_group_data = data[data.age_group_id.isin(exposure_age_groups)]

    check_data_exist(valid_age_group_data, zeros_missing=True)

    expected_columns = ['rei_id', 'modelable_entity_id', 'measure_id',
                        'metric_id'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data,  ['Continuous'])
    check_metric_id(data, 'rate')

    check_years(data, context, 'either')
    check_location(data, context)

    age_start = min(exposure_age_groups)
    age_end = max(exposure_age_groups)

    check_age_group_ids(data, context, age_start, age_end)
    check_sex_ids(data, context, True, True)

    check_sex_restrictions(data, context, entity.restrictions.male_only, entity.restrictions.female_only)

    check_value_columns_boundary(valid_age_group_data, 0, 'lower',
                                 value_columns=DRAW_COLUMNS, inclusive=False, error=DataAbnormalError)


def validate_exposure_distribution_weights(data: pd.DataFrame, entity: Union[RiskFactor, AlternativeRiskFactor],
                                           context: RawValidationContext) -> None:
    """Check the standard set of validations on raw exposure distribution
    weights data for entity, replacing the age ids check with a custom check
    for the all ages age group since distribution weights are not age specific.
    Because exposure distribution weights are neither age nor sex specific
    (and risk factor age restrictions don't correspond to data), skip all
    restriction checks for risk factors.

    Additionally, verify that distribution weights sum to 1.

    Parameters
    ----------
    data
        Exposure distribution weight data for `entity` in location
        `location_id`.
    entity
        Risk factor or alternative risk factor to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data,
        any values in columns do not match the expected set of values,
        or distribution weights do not sum to 1.

    """
    key_cols = ['rei_id', 'location_id', 'sex_id', 'age_group_id', 'measure']
    distribution_cols = ['exp', 'gamma', 'invgamma', 'llogis', 'gumbel', 'invweibull', 'weibull',
                         'lnorm', 'norm', 'glnorm', 'betasr', 'mgamma', 'mgumbel']

    check_data_exist(data, zeros_missing=True, value_columns=distribution_cols)

    check_columns(key_cols + distribution_cols, data.columns)

    if set(data.measure) != {'ensemble_distribution_weight'}:
        raise DataAbnormalError(f'Exposure distribution weight data for {entity.kind} {entity.name} '
                                f'contains abnormal measure values.')

    check_location(data, context)

    if set(data.age_group_id) != {SPECIAL_AGES["all_ages"]}:
        raise DataAbnormalError(f'Exposure distribution weight data for {entity.kind} {entity.name} includes '
                                f'age groups beyond the expected all ages age group (id {SPECIAL_AGES["all_ages"]}.')

    check_sex_ids(data, context, male_expected=False, female_expected=False, combined_expected=True)

    check_value_columns_boundary(data, 0, 'lower', value_columns=distribution_cols,
                                 inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, 1, 'upper', value_columns=distribution_cols,
                                 inclusive=True, error=DataAbnormalError)

    if not np.allclose(data[distribution_cols].sum(axis=1), 1.0):
        raise DataAbnormalError(f'Distribution weights for {entity.kind} {entity.name} do not sum to 1.')


def validate_relative_risk(data: pd.DataFrame, entity: RiskFactor,
                           context: RawValidationContext) -> None:
    """Check the standard set of validations on raw relative risk data for
    entity, replacing the age ids check with a custom check based on the age
    groups present in the exposure data for this entity. Also replacing the
    sex id and sex restrictions checks based on both sex restrictions of
    risk factor and affected cause. Since risk factor restrictions are not
    applied to this measure, we apply the affected cause restrictions to check
    age restrictions. Sex restrictions are only checked if one of male only or
    female only flag is turned on. Check age and sex ids, age and sex
    restrictions on data grouped by cause, mortality, morbidity, and parameter.

    The boundary value checks is also done on the same grouped data to apply
    the different boundary for the pair of risk factor and cause if risk
    factor has a protective effect on a particular cause.

    Additionally, mortality and morbidity flags in data are checked to ensure
    they contain only valid values and only valid combinations of those values
    across mortality and morbidity.

    Parameters
    ----------
    data
        Relative risk data for `entity` in location `location_id`.
    entity
        Risk factor or alternative risk factor to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data, or
        any values in columns do not match the expected set of values (or the
        expected combinations of values in the case of the mortality and
        morbidity columns).

    """
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['rei_id', 'modelable_entity_id', 'cause_id', 'mortality',
                        'morbidity', 'metric_id', 'parameter', 'exposure'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_metric_id(data, 'rate')

    check_years(data, context, 'binned')
    check_location(data, context)

    for c_id in data.cause_id.unique():
        cause = [c for c in causes if c.gbd_id == c_id][0]
        check_mort_morb_flags(data[data.cause_id == c_id], cause.restrictions.yld_only, cause.restrictions.yll_only)

    grouped = data.groupby(['cause_id', 'morbidity', 'mortality', 'parameter'])
    if entity.kind == 'risk_factor':
        restrictions = entity.restrictions
        exposure_age_groups = set(context['exposure'].age_group_id)
        age_start = min(exposure_age_groups)
        age_end = max(exposure_age_groups)
        male_expected = not restrictions.female_only
        female_expected = not restrictions.male_only
        grouped.apply(check_age_group_ids, context, age_start, age_end)

        #  We cannot check age_restrictions with exposure_age_groups since RR may have a subset of age_group_ids.
        #  In this case we do not want to raise an error because RR data may include only specific age_group_ids for
        #  age-specific-causes even if risk-exposure may exist for the other age_group_ids. Instead we check age
        #  restrictions with affected causes.
        for (c_id, morb, mort, _), g in grouped:
            cause = [c for c in causes if c.gbd_id == c_id][0]
            if morb == 1:
                start, end = cause.restrictions.yld_age_group_id_start, cause.restrictions.yld_age_group_id_end
            else:  # morb = 0 , mort = 1
                start, end = cause.restrictions.yll_age_group_id_start, cause.restrictions.yll_age_group_id_end

            male_expected = male_expected and not cause.restrictions.female_only
            female_expected = female_expected and not cause.restrictions.male_only
            check_sex_ids(g, context, male_expected, female_expected)
            check_age_restrictions(g, context, start, end, error=False)

            #  check only if there is a sex restriction (male only or female only).
            if not male_expected or not female_expected:
                check_sex_restrictions(g, context, male_expected, female_expected)

            if entity.name in PROTECTIVE_CAUSE_RISK_PAIRS and cause in PROTECTIVE_CAUSE_RISK_PAIRS[entity.name]:
                check_value_columns_boundary(g, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True,
                                             error=DataAbnormalError)
                check_value_columns_boundary(g, 1, 'upper', value_columns=DRAW_COLUMNS, inclusive=True)
            else:
                #  FIXME: we need to revisit this. There are risk-cause pair when paf > 0 but RR < 1
                check_value_columns_boundary(g, 1, 'lower', value_columns=DRAW_COLUMNS, inclusive=True)

            max_val = MAX_CONT_REL_RISK if entity.distribution in ('ensemble', 'lognormal', 'normal') else MAX_CATEG_REL_RISK
            check_value_columns_boundary(g, max_val, 'upper', value_columns=DRAW_COLUMNS, inclusive=True)

    else:  # coverage gap
        grouped.apply(check_age_group_ids, context, None, None)
        grouped.apply(check_sex_ids, True, True)
        grouped.apply(check_value_columns_boundary, 1, 'lower')
        grouped.apply(check_value_columns_boundary, MAX_CATEG_REL_RISK, 'upper')


def validate_population_attributable_fraction(data: pd.DataFrame, entity: Union[RiskFactor, Etiology],
                                              context: RawValidationContext) -> None:
    """Check the standard set of validations on raw population attributable
    fraction data for entity, replacing the age restrictions check with
    a custom method. Also replacing the sex id and sex restrictions checks
    based on both sex restrictions of risk factor and affected cause.
    Sex restrictions are only checked if one of male only or
    female only flag is turned on. Check age and sex ids, age and sex
    restrictions on data grouped by cause and measure_id.

    The boundary value checks is also done on the same grouped data to apply
    the different boundary for the pair of risk factor and cause if risk
    factor has a protective effect on a particular cause.

    Additionally, check yll/yld only restrictions to ensure that data
    do not include the data with the excluded measure_id by restrictions.
    Instead of the standard age restrictions check, custom method is
    applied to the data grouped by cause and measure id. This method
    is to verify that data follows the cause level age restrictions as well
    as data align with associated exposure and relative risk data.

    Parameters
    ----------
    data
        Population attributable fraction data for `entity` in location `location_id`.
    entity
        Risk factor to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data, or
        any values in columns do not match the expected set of values, or
        yll/yld data exist for yld only/yll only causes.(or data exist outside
        of cause age restrictions or data do not exist for the age groups for
        which both exposure and relative risk exist.

    """

    check_data_exist(data, zeros_missing=True)

    expected_columns = ['metric_id', 'measure_id', 'rei_id', 'cause_id'] + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['YLLs', 'YLDs'], single_only=False)
    check_metric_id(data, 'percent')

    check_years(data, context, 'annual')
    check_location(data, context)

    restrictions = entity.restrictions
    male_expected = not restrictions.female_only
    female_expected = not restrictions.male_only

    grouped = data.groupby(['cause_id', 'measure_id'])

    for (c_id, _), g in grouped:
        cause = [c for c in causes if c.gbd_id == c_id][0]
        male_expected = male_expected and not cause.restrictions.female_only
        female_expected = female_expected and not cause.restrictions.male_only

        check_age_group_ids(g, context, None, None)
        check_sex_ids(g, context, male_expected, female_expected)
        #  check only if there is a sex restriction (male only or female only).
        if not male_expected or not female_expected:
            check_sex_restrictions(g, context, male_expected, female_expected)
        check_paf_rr_exposure_age_groups(g, context, entity)

    protective_causes = PROTECTIVE_CAUSE_RISK_PAIRS[entity.name] if entity.name in PROTECTIVE_CAUSE_RISK_PAIRS else []

    protective = data[data.cause_id.isin([c.gbd_id for c in protective_causes])]
    non_protective = data.loc[data.index.difference(protective.index)]

    if not protective.empty:
        check_value_columns_boundary(protective, MIN_PROTECTIVE_PAF, 'lower', value_columns=DRAW_COLUMNS, inclusive=True,
                                     error=DataAbnormalError)
        check_value_columns_boundary(protective, MIN_PAF, 'upper', value_columns=DRAW_COLUMNS, inclusive=True)
        check_value_columns_boundary(protective, MAX_PAF, 'upper', value_columns=DRAW_COLUMNS, inclusive=True,
                                     error=DataAbnormalError)
    if not non_protective.empty:
        error = None if entity.name in RISKS_WITH_NEGATIVE_PAF else DataAbnormalError
        check_value_columns_boundary(non_protective, MIN_PAF, 'lower', value_columns=DRAW_COLUMNS, inclusive=True,
                                     error=error)
        check_value_columns_boundary(non_protective, MAX_PAF, 'upper', value_columns=DRAW_COLUMNS, inclusive=True,
                                     error=DataAbnormalError)

    check_cause_yll_yld_only_restrictions(data, entity)


def validate_etiology_population_attributable_fraction(data: pd.DataFrame, entity: Etiology,
                                                       context: RawValidationContext) -> None:

    """Check the standard set of validations on raw etiology population
    attributable fraction data for entity. Check age group and sex ids
    and restrictions.

    Additionally, check yll/yld only restrictions to ensure that data
    do not include the data with the excluded measure_id by restrictions.

    Parameters
    ----------
    data
        Population attributable fraction data for `entity` in location `location_id`.
    entity
        Etiology to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data,
        any values in columns do not match the expected set of values,
        or yll/yld data exist for yld only/yll only causes.

    """
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['metric_id', 'measure_id', 'rei_id', 'cause_id'] + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['YLLs', 'YLDs'], single_only=False)
    check_metric_id(data, 'percent')

    check_years(data, context, 'annual')
    check_location(data, context)

    restrictions_entity = [c for c in causes if entity in c.etiologies][0]

    restrictions = restrictions_entity.restrictions
    age_start = get_restriction_age_boundary(restrictions_entity, 'start')
    age_end = get_restriction_age_boundary(restrictions_entity, 'end')
    male_expected = not restrictions.female_only
    female_expected = not restrictions.male_only

    check_age_group_ids(data, context, age_start, age_end)
    check_sex_ids(data, context, male_expected, female_expected)

    check_age_restrictions(data, context, age_start, age_end)
    check_sex_restrictions(data, context, restrictions.male_only, restrictions.female_only)

    #  Loosen the lower boundary since we know that there exist negative paf for a certain etiology.
    #  However, keep the upper boundary until we hit the actual case.
    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True)
    check_value_columns_boundary(data, 1, 'upper', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)

    check_cause_yll_yld_only_restrictions(data, entity)


def validate_mediation_factors(data: pd.DataFrame, entity: RiskFactor, context: RawValidationContext) -> None:
    raise NotImplementedError()


def validate_estimate(data: pd.DataFrame, entity: Covariate, context: RawValidationContext) -> None:
    """ Check the standard set of validations on raw estimate data
    for entity, allowing for the possibility of all 0s in the data as valid.
    Additionally, the standard age and sex checks are replaced with
    custom covariate versions since covariate restrictions only signal whether
    an entity is age and/or sex specific, nothing about the actual age or sex
    values expected in the data.

    Parameters
    ----------
    data
        Estimate data pulled for entity in location_id.
    entity
        Covariate to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data, or
        any values in columns do not match the expected set of values.

    """
    value_columns = ['mean_value', 'upper_value', 'lower_value']

    check_data_exist(data, zeros_missing=False, value_columns=value_columns)

    expected_columns = ['model_version_id', 'covariate_id', 'covariate_name_short', 'location_id',
                        'location_name', 'year_id', 'age_group_id', 'age_group_name', 'sex_id',
                        'sex'] + value_columns
    check_columns(expected_columns, data.columns)

    check_years(data, context, 'annual')
    check_location(data, context)

    if entity.by_age:
        check_age_group_ids(data, context, None, None)
        if not set(data.age_group_id).intersection(set(context['age_group_ids'])):
            # if we have any of the expected gbd age group ids, restriction is not violated
            raise DataAbnormalError('Data is supposed to be age-separated, but does not contain any GBD age group ids.')

    # if we have any age group ids besides all ages and age standardized, restriction is violated
    if not entity.by_age and bool((set(data.age_group_id) - {SPECIAL_AGES['all_ages'], SPECIAL_AGES['age_standardized']})):
        raise DataAbnormalError('Data is not supposed to be separated by ages, but contains age groups '
                                'beyond all ages and age standardized.')

    sexes = context['sexes']
    if entity.by_sex and not {sexes['Male'], sexes['Female']}.issubset(set(data.sex_id)):
        raise DataAbnormalError('Data is supposed to be by sex, but does not contain both male and female data.')
    elif not entity.by_sex and set(data.sex_id) != {sexes['Combined']}:
        raise DataAbnormalError('Data is not supposed to be separated by sex, but contains sex ids beyond that '
                                'for combined male and female data.')


def validate_cost(data: pd.DataFrame, entity: Union[HealthcareEntity, HealthTechnology],
                  context: RawValidationContext) -> None:
    """ Check the standard set of validations on raw cost data for entity,
    replacing the age ids check with a custom check for the
    all ages age group since cost data are not age specific and skipping all
    restrictions checks since neither HealthCareEntities nor HealthTechnologies
    have restrictions.

    Parameters
    ----------
    data
        Cost data pulled for entity in location_id.
    entity
        HealthcareEntity or HealthTechnology to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data, or
        any values in columns do not match the expected set of values.

    """
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure', entity.kind] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    if set(data.measure) != {'cost'}:
        raise DataAbnormalError(f'Cost data for {entity.kind} {entity.name} contains '
                                f'measures beyond the expected cost.')

    check_years(data, context, 'annual')
    check_location(data, context)

    if set(data.age_group_id) != {SPECIAL_AGES['all_ages']}:
        raise DataAbnormalError(f'Cost data for {entity.kind} {entity.name} includes age groups beyond '
                                f'the expected all ages age group (id {SPECIAL_AGES["all_ages"]}).')

    check_sex_ids(data, context, male_expected=False, female_expected=False, combined_expected=True)
    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)


def validate_utilization_rate(data: pd.DataFrame, entity: HealthcareEntity, context: RawValidationContext) -> None:
    """ Check the standard set of validations on raw utilization data for
    entity, skipping all restrictions checks since HealthCareEntities do not
    have restrictions.

    Parameters
    ----------
    data
        Utilization data pulled for entity in location_id.
    entity
        HealthcareEntity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data, or
        any values in columns do not match the expected set of values.

    """
    check_data_exist(data, zeros_missing=True)

    expected_columns = ['measure_id', 'metric_id', 'model_version_id',
                        'modelable_entity_id'] + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data, ['Continuous'])
    check_metric_id(data, 'rate')

    check_years(data, context, 'binned')
    check_location(data, context)

    check_age_group_ids(data, context, None, None)
    check_sex_ids(data, context, male_expected=True, female_expected=True, combined_expected=False)

    check_value_columns_boundary(data, 0, 'lower', value_columns=DRAW_COLUMNS, inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, MAX_UTILIZATION, 'upper', value_columns=DRAW_COLUMNS,
                                 inclusive=True, error=None)


def validate_structure(data: pd.DataFrame, entity: Population, context: RawValidationContext) -> None:
    """Check the standard set of validations on raw population data,
    skipping all restrictions checks since Population entities do not
    have restrictions.

    Parameters
    ----------
    data
        Population data pulled for location_id.
    entity
        Generic population entity.
    context
         Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data does not exist, expected columns are not found in data, or
        any values in columns do not match the expected set of values.

    """
    check_data_exist(data, zeros_missing=True, value_columns=['population'])

    expected_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id', 'population', 'run_id']
    check_columns(expected_columns, data.columns)

    check_years(data, context, 'annual')
    check_location(data, context)

    check_age_group_ids(data, context, None, None)
    check_sex_ids(data, context, male_expected=True, female_expected=True, combined_expected=True)

    check_value_columns_boundary(data, 0, 'lower', value_columns=['population'],
                                 inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, MAX_POP, 'upper', value_columns=['population'],
                                 inclusive=True, error=DataAbnormalError)


def validate_theoretical_minimum_risk_life_expectancy(data: pd.DataFrame, entity: Population,
                                                      context: RawValidationContext) -> None:
    """ Check the standard set of validations on raw life expectancy data,
    skipping the standard age and sex checks since life expectancy is not sex
    specific and is reported in custom age bins rather than the standard GBD
    age bins. Instead, the ages in data are verified to span the range [0, 110].
    All restrictions checks are also skipped since Population entities do not
    have restrictions.

    Parameters
    ----------
    data
       Life expectancy data pulled.
    entity
       Generic population entity.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
       If data does not exist, expected columns are not found in data, or
       any values in columns do not match the expected set of values, including
       if the ages in the data don't span [0, 105].

    """
    check_data_exist(data, zeros_missing=True, value_columns=['life_expectancy'])

    expected_columns = ['age', 'life_expectancy']
    check_columns(expected_columns, data.columns)

    min_age, max_age = 0, 105
    if data.age.min() != min_age or data.age.max() != max_age:
        raise DataAbnormalError(f'Data does not contain life expectancy values for ages [{min_age}, {max_age}].')

    check_value_columns_boundary(data, 0, 'lower', value_columns=['life_expectancy'],
                                 inclusive=True, error=DataAbnormalError)
    check_value_columns_boundary(data, MAX_LIFE_EXP, 'upper', value_columns=['life_expectancy'],
                                 inclusive=True, error=DataAbnormalError)


############################
# CHECK METADATA UTILITIES #
############################


def check_cause_age_restrictions_sets(entity: Cause) -> None:
    """Check that a cause does not have an age range based on its YLL
    restrictions that is broader than that based on its YLD restrictions.

    Parameters
    ----------
    entity
        Cause for which to check restriction age ranges.

    Warns
    ------
        If a wider age range is found so users can further investigate.
    """
    if entity.restrictions.yld_only or entity.restrictions.yll_only:
        pass
    else:

        yll_start, yll_end = entity.restrictions.yll_age_group_id_start, entity.restrictions.yll_age_group_id_end
        yld_start, yld_end = entity.restrictions.yld_age_group_id_start, entity.restrictions.yld_age_group_id_end

        if yll_start < yld_start or yld_end < yll_end:
            logger.warning(
                f'{entity.name} has a broader yll age range than yld age range. This likely means there are age '
                f'groups for which there is no incidence or prevalence but there are deaths. Data will be filtered '
                f'by these age ranges. If you are putting this data into a simulation, please ensure that you have '
                f'thoroughly investigated this and verified your model.'
            )

############################
# RAW VALIDATION UTILITIES #
############################


def check_mort_morb_flags(data: pd.DataFrame, yld_only: bool, yll_only: bool) -> None:
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

    morbidity, no_morbidity = data.morbidity == 1, data.morbidity == 0
    mortality, no_mortality = data.mortality == 1, data.mortality == 0

    if (no_morbidity & no_mortality).any():
        raise DataAbnormalError(base_error_msg + 'rows with both mortality and morbidity flags set to 0.')

    elif (morbidity & mortality).any():
        if no_morbidity.any() or no_mortality.any():
            raise DataAbnormalError(base_error_msg + 'row with both mortality and morbidity flags set to 1 as well as '
                                                     'rows with only one of the mortality or morbidity flags set to 1.')
    else:
        if yld_only and mortality.any():
            raise DataAbnormalError(base_error_msg + 'rows with the mortality flag set to 1 but the affected entity '
                                                     'is restricted to yld_only.')
        elif yll_only and morbidity.any():
            raise DataAbnormalError(base_error_msg + 'rows with the morbidity flag set to 1 but the affected entity '
                                                     'is restricted to yll_only.')
        else:
            pass


def check_cause_yll_yld_only_restrictions(data: pd.DataFrame, entity: Union[RiskFactor, Etiology]) -> None:
    """Verify that there is no data violating yll/yld only restrictions.

    Parameters
    ----------
    data
        Dataframe containing measure_id and cause_id. Only expected to be
        population attributable fraction data.
    entity
        RiskFactor or Etiology to which the data pertain.
    Raises
    -------
    DataAbnormalError
        If yll measure id is found for yld only cause or yld measure id is
        found for yll only cause.

    """
    for c_id in set(data.cause_id):
        cause = [c for c in causes if c.gbd_id == c_id][0]
        if cause.restrictions.yld_only and np.any(data[(data.cause_id == c_id) & (data.measure_id == MEASURES['YLLs'])]):
            raise DataAbnormalError(f'Paf data for {entity.kind} {entity.name} affecting {cause.name} contains yll '
                                    f'values despite the affected entity being restricted to yld only.')
        if cause.restrictions.yll_only and np.any(data[(data.cause_id == c_id) & (data.measure_id == MEASURES['YLDs'])]):
            raise DataAbnormalError(f'Paf data for {entity.kind} {entity.name} affecting {cause.name} contains yld '
                                    f'values despite the affected entity being restricted to yll only.')


def _get_valid_rr_and_age_groups(context: RawValidationContext, entity: RiskFactor, cause: Cause,
                                 measure_id: int) -> Tuple[Set, pd.DataFrame]:
    """According to the distribution type of RiskFactor, it finds the non-
    trivial relative risk and returns its age groups ids and relative risk
    only containing the given `cause` and `measure id`.

    Parameters
    ----------
    context
        Wrapper for additional data used in the validation process.
    entity
        RiskFactor to which the data pertain.
    cause
        Cause of which the restrictions to be used.
    measure_id
        Measure_id to be used to filter relative risk.
    Returns
    -------
    rr_age_groups is set of age group ids that have non trivial relative risk.
    valid_rr is a dataframe filtered by given cause and measure.

    """
    rr = context['relative_risk']
    exposure = context['exposure']
    rr_measures = {'YLLs': (rr.morbidity == 0) & (rr.mortality == 1), 'YLDs': (rr.morbidity == 1)}

    measure = 'YLLs' if measure_id == MEASURES['YLLs'] else 'YLDs'
    valid_rr = rr[(rr.cause_id == cause.gbd_id) & rr_measures[measure]]

    if entity.distribution in ['ensemble', 'lognormal', 'normal']:
        tmrel = (entity.tmred.max + entity.tmred.min) / 2

        #  Non-trivial rr for continuous risk factors is where exposure is bigger(smaller) than tmrel.
        e_othercols = [c for c in exposure.columns if c not in DRAW_COLUMNS]
        df = exposure.set_index(e_othercols)
        op = operator.lt if entity.tmred.inverted else operator.gt
        exposed_age_groups = set(df[op(df, tmrel)].reset_index().age_group_id)

        valid_rr = valid_rr[valid_rr.age_group_id.isin(exposed_age_groups)]
        rr_age_groups = set(valid_rr.age_group_id)

    else:  # categorical distribution
        #  Non-trivial rr for categorical risk factors is where relative risk is not equal to 1.
        #  Since non-trivial rr is determined by rr itself and rr age_group_id set is guaranteed to be
        #  a subset of exposure age_group_id set, we do not check exposure here.
        rr_othercols = [c for c in rr.columns if c not in DRAW_COLUMNS]
        df = rr.set_index(rr_othercols)
        rr_age_groups = set(df[df != 1].reset_index().age_group_id)

    return rr_age_groups, valid_rr


def check_paf_rr_exposure_age_groups(paf: pd.DataFrame, context: RawValidationContext, entity: RiskFactor)-> None:
    """Check whether population attributable fraction data have consistent
    age group ids to the exposure, relative risk and cause restrictions.
    Since this function applies after the data is grouped by cause and measure,
    we expect input paf to have exactly one cause and one measure.
    We check the followings:
    1.  paf should not have extra age groups outside of cause restrictions
    2.  paf should exist for the age groups where exposure and relative risk
        exist and cause restrictions are valid.

    The only exception is when paf has yll measure and there is only relative
    risk with both mortality and morbidity flags turned on. (pass without check)

    Parameters
    ----------
    paf
        Population attributable data for a `entity` for a single cause and
        a single measure.
    context
        Wrapper for additional data used in the validation process.
    entity
        RiskFactor for which to check age groups.
    Raises
    -------
    DataAbnormalError
        If any of two checks described above fails.

    """
    age_group_ids = context['age_group_ids']

    cause_id = paf.cause_id.unique()[0]
    measure_id = paf.measure_id.unique()[0]
    cause = [c for c in causes if c.gbd_id == cause_id][0]

    age_restrictions = {MEASURES['YLLs']: (cause.restrictions.yll_age_group_id_start, cause.restrictions.yll_age_group_id_end),
                        MEASURES['YLDs']: (cause.restrictions.yld_age_group_id_start, cause.restrictions.yld_age_group_id_end)}

    rr_age_groups, valid_rr = _get_valid_rr_and_age_groups(context, entity, cause, measure_id)

    # It means we have YLL Paf but mortality = morbidity = 1 and we do not support this case.
    if measure_id == MEASURES['YLLs'] and valid_rr.empty:
        pass

    else:
        #  We apply the narrowest restrictions among exposed_age_groups, rr_age_gruops and cause age restrictions.
        #  We may have paf outside of exposure/rr but inside of cause age restrictions, then warn it.
        #  If paf does not exist for the narrowest range of exposure/rr/cause, raise an error.
        cause_age_start, cause_age_end = age_restrictions[measure_id]
        cause_restriction_ages = set(get_restriction_age_ids(cause_age_start, cause_age_end, age_group_ids))

        age_groups_paf_should_exist = rr_age_groups.intersection(cause_restriction_ages)

        valid_but_no_rr = set(cause_restriction_ages) - rr_age_groups

        #  since paf may not exist for the full age group ids in cause_restriction_ages, we only raise an error
        #  if there are extra data than cause_restriction_ages.
        not_valid_paf = set(paf.age_group_id) > cause_restriction_ages
        missing_pafs = age_groups_paf_should_exist - set(paf.age_group_id)
        extra_paf = set(paf.age_group_id).intersection(valid_but_no_rr)

        measure = 'YLLs' if measure_id == MEASURES['YLLs'] else 'YLDs'
        if not_valid_paf and not (entity.name in PAF_OUTSIDE_AGE_RESTRICTIONS
                                  and cause in PAF_OUTSIDE_AGE_RESTRICTIONS[entity.name]):
            raise DataAbnormalError(f'{measure} paf for {cause.name} and {entity.name} have data outside '
                                    f'of cause restrictions: {set(paf.age_group_id) - cause_restriction_ages}')

        if missing_pafs:
            raise DataAbnormalError(f"Paf for {cause.name} and {entity.name} have missing data for "
                                    f"the age groups: {missing_pafs}.")
        if extra_paf:
            logger.warning(
                f"{measure} paf for {cause.name} and {entity.name} have data for the age groups: {extra_paf}, "
                f"which do not have either relative risk or exposure data."
            )


def check_years(data: pd.DataFrame, context: RawValidationContext, year_type: str) -> None:
    """Check that years in passed data match expected range based on type.

    Parameters
    ----------
    data
        Dataframe containing 'year_id' column.
    context
        Wrapper for additional data used in the validation process.
    year_type
        'annual', 'binned', or 'either' indicating expected year range.

    Raises
    ------
    DataAbnormalError
        If `error` is turned on and any expected years are not found in data or
        any extra years found and `year_type` is 'binned'.

    """
    data_years = set(data.year_id.unique())
    estimation_years = set(context['estimation_years'])
    annual_estimation_years = set(range(min(estimation_years), max(estimation_years) + 1))
    if year_type == 'annual':
        if data_years < annual_estimation_years:
            raise DataAbnormalError(f'Data has missing years: {annual_estimation_years.difference(data_years)}.')
    elif year_type == 'binned':
        if data_years < estimation_years:
            raise DataAbnormalError(f'Data has missing years: {estimation_years.difference(data_years)}.')
        if data_years > estimation_years:
            raise DataAbnormalError(f'Data has extra years: {data_years.difference(estimation_years)}.')
    else:  # year_type == either
        valid = (data_years == estimation_years) or (data_years >= annual_estimation_years)
        if not valid:
            raise DataAbnormalError(f'Data year range is neither annual or appropriately binned '
                                    f'with the expected year range.')


def check_location(data: pd.DataFrame, context: RawValidationContext) -> None:
    """Check that data contains only a single unique location id and that that
    location id matches the requested `location_id` or one of its parents up to
    the global id.

    Parameters
    ----------
    data
        Dataframe containing a 'location_id' column.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataAbnormalError
        If data contains multiple location ids or a location id other than the
        global or requested location id.

    """
    if len(data['location_id'].unique()) > 1:
        raise DataAbnormalError(f'Data contains multiple location ids.')

    data_location_id = data['location_id'].unique()[0]

    if data_location_id not in context['parent_locations'] + [context['location_id']]:
        raise DataAbnormalError(f'Data pulled for {context["location_id"]} actually has location '
                                f'id {data_location_id}, which is not in its hierarchy.')


def check_columns(expected_cols: List, existing_cols: List) -> None:
    """Verify that the passed lists of columns match.

    Parameters
    ----------
    expected_cols
        List of column names expected.
    existing_cols
        List of column names actually found in data.

    Raises
    ------
    DataAbnormalError
        If `expected_cols` does not match `existing_cols`.

    """
    if set(existing_cols) < set(expected_cols):
        raise DataAbnormalError(f'Data is missing columns: {set(expected_cols).difference(set(existing_cols))}.')
    elif set(existing_cols) > set(expected_cols):
        raise DataAbnormalError(f'Data returned extra columns: {set(existing_cols).difference(set(expected_cols))}.')


def check_data_exist(data: pd.DataFrame, zeros_missing: bool,
                     value_columns: list = DRAW_COLUMNS, error: bool = True) -> bool:
    """Check that values in data exist and none are missing and, if
    `zeros_missing` is turned on, not all zero.

    Parameters
    ----------
    data
        Dataframe containing `value_columns`.
    zeros_missing
        Boolean indicating whether to treat all zeros in `value_columns` as
        missing or not.
    value_columns
        List of columns in `data` to check for missing values.
    error
        Boolean indicating whether or not to error if data is missing.

    Returns
    -------
    bool
        True if non-missing, non-infinite, non-zero (if zeros_missing) values
        exist in data, False otherwise.

    Raises
    -------
    DataDoesNotExistError
        If error flag is set to true and data is empty or contains any NaN
        values in `value_columns`, or contains all zeros in `value_columns` and
        zeros_missing is True.

    """
    if (data.empty or np.any(pd.isnull(data[value_columns]))
            or (zeros_missing and np.all(data[value_columns] == 0)) or np.any(np.isinf(data[value_columns]))):
        if error:
            raise DataDoesNotExistError(f'Data contains no non-missing{", non-zero" if zeros_missing else ""} values.')
        return False
    return True


def _check_continuity(data_ages: set, all_ages: set) -> None:
    """Make sure data_ages is contiguous block in all_ages."""
    data_ages = list(data_ages)
    all_ages = list(all_ages)
    all_ages.sort()
    data_ages.sort()
    if all_ages[all_ages.index(data_ages[0]):all_ages.index(data_ages[-1])+1] != data_ages:
        raise DataAbnormalError(f'Data contains a non-contiguous age groups: {data_ages}.')


def check_age_group_ids(data: pd.DataFrame, context: RawValidationContext,
                        restriction_start: Union[int, None], restriction_end: Union[int, None]) -> None:
    """Check the set of age_group_ids included in data pulled from GBD for
    the following conditions:

        - if data ages contain invalid age group ids, error.
        - if data ages are equal to the set of all GBD age groups or the set of
        age groups within restriction bounds (if restrictions apply), pass.
        - if data ages are not a contiguous block of GBD age groups, error.
        - if data ages are a proper subset of the set of restriction age groups
        or the restriction age groups are a proper subset of the data ages,
        warn.

    Parameters
    ----------
    data
        Dataframe pulled containing age_group_id column.
    context
        Wrapper for additional data used in the validation process.
    restriction_start
        Age group id representing the start of the restriction range
        if applicable.
    restriction_end
        Age group id representing the end of the restriction range
        if applicable.

    Raises
    ------
    DataAbnormalError
        If age group ids contained in data aren't all valid GBD age group ids
        or they don't make up a contiguous block.

    """
    all_ages = set(context['age_group_ids'])
    restriction_ages = set(get_restriction_age_ids(restriction_start, restriction_end, context['age_group_ids']))
    data_ages = set(data.age_group_id)

    invalid_ages = data_ages.difference(all_ages)
    if invalid_ages:
        raise DataAbnormalError(f'Data contains invalid age group ids: {invalid_ages}.')

    _check_continuity(data_ages, all_ages)

    if data_ages < restriction_ages:
        logger.warning('Data does not contain all age groups in restriction range.')
    elif restriction_ages and restriction_ages < data_ages:
        logger.warning('Data contains additional age groups beyond those specified by restriction range.')
    else:  # data_ages == restriction_ages
        pass


def check_sex_ids(data: pd.DataFrame, context: RawValidationContext,
                  male_expected: bool = True, female_expected: bool = True, combined_expected: bool = False) -> None:
    """Check whether the data contains valid GBD sex ids and whether the set of
    sex ids in the data matches the expected set.

    Parameters
    ----------
    data
        Dataframe containing a sex_id column.
    context
        Wrapper for additional data used in the validation process.
    male_expected
        Boolean indicating whether the male sex id is expected in this data.
        For some data pulling tools, this may correspond to whether the entity
        the data describes has a male_only sex restriction.
    female_expected
        Boolean indicating whether the female sex id is expected in this data.
        For some data pulling tools, this may correspond to whether the entity
        the data describes has a female_only sex restriction.
    combined_expected
        Boolean indicating whether data is expected to include the
        combined sex id.

    Raises
    ------
    DataAbnormalError
        If data contains any sex ids that aren't valid GBD sex ids.

    """
    sexes = context['sexes']
    valid_sex_ids = [sexes['Male'], sexes['Female'], sexes['Combined']]
    gbd_sex_ids = set(np.array(valid_sex_ids)[[male_expected, female_expected, combined_expected]])
    data_sex_ids = set(data.sex_id)

    invalid_sex_ids = data_sex_ids.difference(set(valid_sex_ids))
    if invalid_sex_ids:
        raise DataAbnormalError(f'Data contains invalid sex ids: {invalid_sex_ids}.')

    extra_sex_ids = data_sex_ids.difference(gbd_sex_ids)
    if extra_sex_ids:
        logger.warning(f'Data contains the following extra sex ids {extra_sex_ids}.')

    missing_sex_ids = set(gbd_sex_ids).difference(data_sex_ids)
    if missing_sex_ids:
        logger.warning(f'Data is missing the following expected sex ids: {missing_sex_ids}.')


def check_age_restrictions(data: pd.DataFrame, context: RawValidationContext,
                           age_group_id_start: int, age_group_id_end: int,
                           value_columns: list = DRAW_COLUMNS, error=True) -> None:
    """Check that all expected age groups between age_group_id_start and
    age_group_id_end, inclusive, and only those age groups, appear in data with
    non-missing values in `value_columns`.

    Parameters
    ----------
    data
        Dataframe containing an age_group_id column.
    context
        Wrapper for additional data used in the validation process.
    age_group_id_start
        Lower boundary of age group ids expected in data, inclusive.
    age_group_id_end
        Upper boundary of age group ids expected in data, exclusive.
    value_columns
        List of columns to verify values are non-missing for expected age
        groups and missing for not expected age groups.
    error
        Boolean indicating whether or not to error if any age_restriction
        is violated. If this flag is set to false, raise a warning.

    Raises
    ------
    DataAbnormalError
        If error flag is set to true and if any age group ids in the range
        [`age_group_id_start`, `age_group_id_end`] don't appear in the data or
        if any additional age group ids (with the exception of 235) appear in
        the data.

    """
    expected_gbd_age_ids = get_restriction_age_ids(age_group_id_start, age_group_id_end, context['age_group_ids'])

    # age groups we expected in data but that are not
    missing_age_groups = set(expected_gbd_age_ids).difference(set(data.age_group_id))
    extra_age_groups = set(data.age_group_id).difference(set(expected_gbd_age_ids))

    if missing_age_groups:
        message = (f'Data was expected to contain all age groups between ids {age_group_id_start} '
                   f'and {age_group_id_end} but was missing the following: {missing_age_groups}.')
        if error:
            raise DataAbnormalError(message)
        logger.warning(message)

    if extra_age_groups:
        # we treat all 0s as missing in accordance with gbd so if extra age groups have all 0 data, that's fine
        should_be_zero = data[data.age_group_id.isin(extra_age_groups)]
        if check_data_exist(should_be_zero, zeros_missing=True, value_columns=value_columns, error=False):
            logger.warning(
                f'Data was only expected to contain values for age groups between ids {age_group_id_start} and '
                f'{age_group_id_end} but also included values for age groups {extra_age_groups}.'
            )

    # make sure we're not missing data for all ages in restrictions
    if not check_data_exist(data[data.age_group_id.isin(expected_gbd_age_ids)], zeros_missing=True,
                            value_columns=value_columns, error=False):
        message = 'Data is missing for all age groups within restriction range.'
        if error:
            raise DataAbnormalError(message)
        logger.warning(message)


def check_sex_restrictions(data: pd.DataFrame, context: RawValidationContext, male_only: bool, female_only: bool,
                           value_columns: list = DRAW_COLUMNS) -> None:
    """Check that all expected sex ids based on restrictions, and only those
    sex ids, appear in data with non-missing values in `value_columns`.

    Parameters
    ----------
    data
        Dataframe contained sex_id column.
    context
        Wrapper for additional data used in the validation process.
    male_only
        Boolean indicating whether data is restricted to male only estimates.
    female_only
        Boolean indicating whether data is restricted to female only estimates.
    value_columns
        List of columns to verify values are non-missing for expected sex
        ids and missing for not expected sex ids.

    Raises
    -------
    DataAbnormalError
        If data violates passed sex restrictions.

    """
    sexes = context['sexes']
    female, male, combined = sexes['Female'], sexes['Male'], sexes['Combined']

    if male_only:
        if not check_data_exist(data[data.sex_id == male], zeros_missing=True,
                                value_columns=value_columns, error=False):
            raise DataAbnormalError('Data is restricted to male only, but is missing data values for males.')

        if (set(data.sex_id) != {male} and
                check_data_exist(data[data.sex_id != male], zeros_missing=True,
                                 value_columns=value_columns, error=False)):
            logger.warning(
                'Data is restricted to male only, but contains non-male sex ids for which data values are not all 0.'
            )

    if female_only:
        if not check_data_exist(data[data.sex_id == female], zeros_missing=True,
                                value_columns=value_columns, error=False):
            raise DataAbnormalError('Data is restricted to female only, but is missing data values for females.')

        if (set(data.sex_id) != {female} and
                check_data_exist(data[data.sex_id != female], zeros_missing=True,
                                 value_columns=value_columns, error=False)):
            logger.warning('Data is restricted to female only, but contains '
                           'non-female sex ids for which data values are not all 0.')

    if not male_only and not female_only:
        if {male, female}.issubset(set(data.sex_id)):
            if (not check_data_exist(data[data.sex_id == male], zeros_missing=True,
                                     value_columns=value_columns, error=False) or
               not check_data_exist(data[data.sex_id == female], zeros_missing=True,
                                    value_columns=value_columns, error=False)):
                raise DataAbnormalError('Data has no sex restrictions, but does not contain non-zero '
                                        'values for both males and females.')
        else:  # check combined sex id
            if not check_data_exist(data[data.sex_id == combined], zeros_missing=True,
                                    value_columns=value_columns, error=False):
                raise DataAbnormalError('Data has no sex restrictions, but does not contain non-zero '
                                        'values for both males and females.')


def check_measure_id(data: pd.DataFrame, allowable_measures: List[str], single_only: bool = True) -> None:
    """Check that data contains a measure id that is one of the allowed
    measure ids.

    Parameters
    ----------
    data
        Dataframe containing 'measure_id' column.
    allowable_measures
        List of strings dictating the possible values for measure id when
        mapped via MEASURES.
    single_only
        Boolean indicating whether a single measure id is expected in the data
        or whether multiple are allowable.

    Raises
    ------
    DataAbnormalError
        If data contains either multiple measure ids and `single_only` is True
        or a non-permissible measure id.

    """
    if single_only and len(set(data.measure_id)) > 1:
        raise DataAbnormalError(f'Data has multiple measure ids: {set(data.measure_id)}.')
    if not set(data.measure_id).issubset(set([MEASURES[m] for m in allowable_measures])):
        raise DataAbnormalError(f'Data includes a measure id not in the expected measure ids for this measure.')


def check_metric_id(data: pd.DataFrame, expected_metric: str) -> None:
    """Check that data contains only a single metric id and that it matches the
    expected metric.

    Parameters
    ----------
    data
        Dataframe containing 'metric_id' column.
    expected_metric
        String dictating the expected metric, the id of which can be found via
        METRICS.

    Raises
    ------
    DataAbnormalError
        If data contains any metric id other than the expected.

    """
    if set(data.metric_id) != {METRICS[expected_metric.capitalize()]}:
        raise DataAbnormalError(f'Data includes metrics beyond the expected {expected_metric.lower()} '
                                f'(metric_id {METRICS[expected_metric.capitalize()]}')
