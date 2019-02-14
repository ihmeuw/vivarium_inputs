from typing import Union, Dict

import numpy as np
import pandas as pd

from gbd_mapping import ModelableEntity, Cause, Sequela, RiskFactor, CoverageGap, Etiology, Covariate, causes
from vivarium_inputs import utilities, utility_data
from vivarium_inputs.globals import (DataTransformationError, Population,
                                     PROTECTIVE_CAUSE_RISK_PAIRS, BOUNDARY_SPECIAL_CASES)
from vivarium_inputs.mapping_extension import HealthcareEntity, HealthTechnology, AlternativeRiskFactor
from vivarium_inputs.validation.shared import check_value_columns_boundary


VALID_INCIDENCE_RANGE = (0.0, 50.0)
VALID_PREVALENCE_RANGE = (0.0, 1.0)
VALID_BIRTH_PREVALENCE_RANGE = (0.0, 1.0)
VALID_DISABILITY_WEIGHT_RANGE = (0.0, 1.0)
VALID_REMISSION_RANGE = (0.0, 120.0)  # James' head
# FIXME: bumping csmr max to 3 because of age group scaling 2/8/19 - K.W.
VALID_CAUSE_SPECIFIC_MORTALITY_RANGE = (0.0, 6)  # used mortality viz, picked worst country 15q45, mul by ~1.25
VALID_EXCESS_MORT_RANGE = (0.0, 120.0)  # James' head
VALID_EXPOSURE_RANGE = (0.0, {'continuous': 10_000.0, 'categorical': 1.0})
VALID_EXPOSURE_SD_RANGE = (0.0, 1000.0)  # James' brain
VALID_EXPOSURE_DIST_WEIGHTS_RANGE = (0.0, 1.0)
VALID_RELATIVE_RISK_RANGE = (1.0, {'continuous': 10.0, 'categorical': 350.0})
VALID_PAF_RANGE = (0.0, 1.0)
VALID_PROTECTIVE_PAF_MIN = -1.0
VALID_COST_RANGE = (0, {'healthcare_entity': 30_000, 'health_technology': 50})
VALID_UTILIZATION_RANGE = (0, 50)
VALID_POPULATION_RANGE = (0, 100_000_000)
VALID_LIFE_EXP_RANGE = (0, 90)


class SimulationValidationContext:

    def __init__(self, location, **additional_data):
        self.context_data = {'location': location}
        self.context_data.update(additional_data)

        if 'years' not in self.context_data:
            self.context_data['years'] = utility_data.get_year_block()
        if 'age_bins' not in self.context_data:
            self.context_data['age_bins'] = utility_data.get_age_bins()

    def __getitem__(self, key):
        return self.context_data[key]

    def __setitem__(self, key, value):
        self.context_data[key] = value


def validate_for_simulation(data: pd.DataFrame, entity: ModelableEntity,
                            measure: str, location: str, **context_args) -> None:
    """Validate data conforms to the format that is expected by the simulation
    and conforms to normal expectations for a measure.

    Data coming in to the simulation is expected to have a full demographic set
    in most instances, as well non-missing, non-infinite, reasonable data. This
    function enforces column names, the demographic extents, and expected
    ranges and relationships of measure data.

    The following standard checks are applied:
    1. Validate standard columns:
        For all demographic columns, ensure that the column names are correct
        and the values in the columns matched the expected set contained in
        the given context.
    2. Validate value columns:
        Ensure that the column name is correct and check that all values within
        the column fall in the expected range.
    3. Validate age/sex restrictions if applicable:
        Ensure that the data matches any age or sex restrictions of the entity.

    Parameters
    ----------
    data
        Data to be validated.
    entity
        The GBD Entity to which the data pertain.
    measure
        The measure to which the data pertain.
    location
        The location to which the data pertain.
    context_args
        Any data or information needed to construct the SimulationContext used
        by the individual entity-measure validator functions.

    Raises
    -------
    DataTransformationError
        If any columns are mis-formatted or assumptions about the data are
        violated.
    """
    validators = {
        # Cause-like measures
        'incidence': validate_incidence,
        'prevalence': validate_prevalence,
        'birth_prevalence': validate_birth_prevalence,
        'disability_weight': validate_disability_weight,
        'remission': validate_remission,
        'cause_specific_mortality': validate_cause_specific_mortality,
        'excess_mortality': validate_excess_mortality,
        # Risk-like measures
        'exposure': validate_exposure,
        'exposure_standard_deviation': validate_exposure_standard_deviation,
        'exposure_distribution_weights': validate_exposure_distribution_weights,
        'relative_risk': validate_relative_risk,
        'population_attributable_fraction': validate_population_attributable_fraction,
        'mediation_factors': validate_mediation_factors,
        # Covariate measures
        'estimate': validate_estimate,
        # Health system measures
        'cost': validate_cost,
        'utilization': validate_utilization,
        # Population measures
        'structure': validate_structure,
        'theoretical_minimum_risk_life_expectancy': validate_theoretical_minimum_risk_life_expectancy,
        'age_bins': validate_age_bins,
        'demographic_dimensions': validate_demographic_dimensions,
    }

    if measure not in validators:
        raise NotImplementedError()

    context = SimulationValidationContext(location, **context_args)
    validators[measure](data, entity, context)


#########################################################
#   VALIDATE SIM DATA ENTITY-MEASURE SPECIFIC METHODS   #
# ----------------------------------------------------- #
#   Signatures to match as used in validate_sim_data    #
#########################################################


def validate_incidence(data: pd.DataFrame, entity: Union[Cause, Sequela], context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped incidence
    data.

    Parameters
    ----------
    data
        Simulation-prepped incidence data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if yld age or sex restrictions are violated, or data falls outside the
        expected boundary values.

    """
    validate_standard_columns(data, context)

    check_value_columns_boundary(data, boundary_value=VALID_INCIDENCE_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_INCIDENCE_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    restrictions_entity = [c for c in causes if entity in c.sequelae][0] if entity.kind == 'sequela' else entity
    check_age_restrictions(data, restrictions_entity, rest_type='yld', fill_value=0.0, context=context)
    check_sex_restrictions(data, restrictions_entity.restrictions.male_only, 
                           restrictions_entity.restrictions.female_only, fill_value=0.0)


def validate_prevalence(data: pd.DataFrame, entity: Union[Cause, Sequela],
                        context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped prevalence
    data.

    Parameters
    ----------
    data
        Simulation-prepped prevalence data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if yld age or sex restrictions are violated, or data falls outside the
        expected boundary values.

    """
    validate_standard_columns(data, context)

    check_value_columns_boundary(data, boundary_value=VALID_PREVALENCE_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_PREVALENCE_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    restrictions_entity = [c for c in causes if entity in c.sequelae][0] if entity.kind == 'sequela' else entity
    check_age_restrictions(data, restrictions_entity, rest_type='yld', fill_value=0.0, context=context)
    check_sex_restrictions(data, restrictions_entity.restrictions.male_only, 
                           restrictions_entity.restrictions.female_only, fill_value=0.0)


def validate_birth_prevalence(data: pd.DataFrame, entity: Union[Cause, Sequela],
                              context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped birth
    prevalence data, skipping the check on age columns since birth prevalence
    is not age-specific.

    Parameters
    ----------
    data
        Simulation-prepped birth prevalence data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if sex restrictions are violated, or data falls outside the expected
        boundary values.

    """
    validate_location_column(data, context)
    validate_sex_column(data)
    validate_year_columns(data, context)
    validate_draw_column(data)
    validate_value_column(data)

    check_value_columns_boundary(data, boundary_value=VALID_BIRTH_PREVALENCE_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_BIRTH_PREVALENCE_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    restrictions_entity = [c for c in causes if entity in c.sequelae][0] if entity.kind == 'sequela' else entity
    check_sex_restrictions(data, restrictions_entity.restrictions.male_only, 
                           restrictions_entity.restrictions.female_only, fill_value=0.0)


def validate_disability_weight(data: pd.DataFrame, entity: Union[Cause, Sequela],
                               context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped disability
    weight data.

    Parameters
    ----------
    data
        Simulation-prepped disability weight data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if yld age or sex restrictions are violated, or data falls outside the
        expected boundary values.

    """
    validate_standard_columns(data, context)
    check_value_columns_boundary(data, boundary_value=VALID_DISABILITY_WEIGHT_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_DISABILITY_WEIGHT_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    restrictions_entity = [c for c in causes if entity in c.sequelae][0] if entity.kind == 'sequela' else entity
    check_age_restrictions(data, restrictions_entity, rest_type='yld', fill_value=0.0, context=context)
    check_sex_restrictions(data, restrictions_entity.restrictions.male_only, 
                           restrictions_entity.restrictions.female_only, fill_value=0.0)


def validate_remission(data: pd.DataFrame, entity: Cause, context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped remission
    data.

    Parameters
    ----------
    data
        Simulation-prepped remission data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if yld age or sex restrictions are violated, or data falls outside the
        expected boundary values.

    """
    validate_standard_columns(data, context)

    check_value_columns_boundary(data, boundary_value=VALID_REMISSION_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_REMISSION_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    check_age_restrictions(data, entity, rest_type='yld', fill_value=0.0, context=context)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_cause_specific_mortality(data: pd.DataFrame, entity: Cause, context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped cause
    specific mortality data.

    Parameters
    ----------
    data
        Simulation-prepped cause specific mortality data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if yll age or sex restrictions are violated, or data falls outside the
        expected boundary values.

    """
    validate_standard_columns(data, context)

    check_value_columns_boundary(data, boundary_value=VALID_CAUSE_SPECIFIC_MORTALITY_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_CAUSE_SPECIFIC_MORTALITY_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    check_age_restrictions(data, entity, rest_type='yll', fill_value=0.0, context=context)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_excess_mortality(data: pd.DataFrame, entity: Cause, context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped excess
    mortality data.

    Parameters
    ----------
    data
        Simulation-prepped excess mortality data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if yll age or sex restrictions are violated, or data falls outside the
        expected boundary values.

    """
    validate_standard_columns(data, context)

    check_value_columns_boundary(data, boundary_value=VALID_EXCESS_MORT_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)

    if entity.name in BOUNDARY_SPECIAL_CASES['excess_mortality'].get(context['location'], {}):
        max_val = BOUNDARY_SPECIAL_CASES['excess_mortality'][context['location']][entity.name]
    else:
        max_val = VALID_EXCESS_MORT_RANGE[1]
    check_value_columns_boundary(data, boundary_value=max_val, boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    check_age_restrictions(data, entity, rest_type='yll', fill_value=0.0, context=context)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_exposure(data: pd.DataFrame, entity: Union[RiskFactor, CoverageGap, AlternativeRiskFactor],
                      context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped exposure
    data, with the upper boundary of values determined by the distribution type.
    The broadest age range determined by yll and yld age restrictions is used
    in the restriction check. For categorical risks, because exposure is filled
    with two values outside the restriction boundaries (one for the TMREL
    category and one for all other categories), check that only these two
    values are found outside restrictions and that they are matched to the
    right categories.

    Additionally, check that the parameter column contains either only
    'continuous' for an entity with a continuous distribution or the full set
    of categories for an entity with a categorical distribution. For entities
    with categorical distributions, further check that exposure values sum to
    one across categories for all demographic groups.

    Parameters
    ----------
    data
        Simulation-prepped exposure data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if the broadest age or sex restrictions are violated, data falls
        outside the expected boundary values, the parameter column does not
        contain the expected values based on the entity's distribution type, or
        exposure values do not sum to 1 for entities with a categorical
        distribution.

    """
    is_continuous = entity.distribution in ['normal', 'lognormal', 'ensemble']
    is_categorical = entity.distribution in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']

    if is_continuous:
        if set(data.parameter) != {"continuous"}:
            raise DataTransformationError("Continuous exposure data should contain "
                                          "'continuous' in the parameter column.")
        valid_kwd = 'continuous'
    elif is_categorical:
        if set(data.parameter) != set(entity.categories.to_dict()):  # to_dict removes None
            raise DataTransformationError("Categorical exposure data does not contain all "
                                          "categories in the parameter column.")
        valid_kwd = 'categorical'
    else:
        raise NotImplementedError()

    check_value_columns_boundary(data, boundary_value=VALID_EXPOSURE_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_EXPOSURE_RANGE[1][valid_kwd],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    cats = data.groupby('parameter')
    cats.apply(validate_standard_columns, context)

    if is_categorical:
        non_categorical_columns = list(set(data.columns).difference({'parameter', 'value'}))
        if not np.allclose(data.groupby(non_categorical_columns)['value'].sum(), 1.0):
            raise DataTransformationError("Categorical exposures do not sum to one across categories.")

    if entity.kind in ['risk_factor', 'alternative_risk_factor']:
        fill_value = {'exposed': 0.0, 'unexposed': 1.0} if is_categorical else 0.0
        check_age_restrictions(data, entity, rest_type='outer', fill_value=fill_value,
                               context=context)
        check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only,
                               fill_value=fill_value, entity=entity)


def validate_exposure_standard_deviation(data: pd.DataFrame, entity: Union[RiskFactor, AlternativeRiskFactor],
                                         context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped exposure
    standard deviation data, using the broadest age range determined by yll and
    yld age restrictions in the restriction check.

    Parameters
    ----------
    data
        Simulation-prepped exposure standard deviation data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if the broadest age or sex restrictions are violated, or data falls
        outside the expected boundary values.

    """
    validate_standard_columns(data, context)

    check_value_columns_boundary(data, boundary_value=VALID_EXPOSURE_SD_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_EXPOSURE_SD_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    check_age_restrictions(data, entity, rest_type='outer', fill_value=0.0, context=context)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_exposure_distribution_weights(data: pd.DataFrame, entity: Union[RiskFactor, AlternativeRiskFactor],
                                           context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped exposure
    distribution weights data, skipping the check on draw columns since weights
    data is not by draw and using the broadest age range determined by yll
    and yld age restrictions in the restriction check.

    Additionally, ensure that weights sum to 1 in all demographic groups
    expected to have valid weights (i.e., all age groups included in
    corresponding exposure data) and to 0 in all other demographic groups.

    Parameters
    ----------
    data
        Simulation-prepped exposure distribution weights data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if the broadest age or sex restrictions are violated, data falls
        outside the expected boundary values, or weights don't sum to 1 or 0.

    """
    validate_demographic_columns(data, context)
    validate_value_column(data)

    check_value_columns_boundary(data, boundary_value=VALID_EXPOSURE_DIST_WEIGHTS_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_EXPOSURE_DIST_WEIGHTS_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    non_weight_columns = list(set(data.columns).difference({'parameter', 'value'}))
    weights_sum = data.groupby(non_weight_columns)['value'].sum()
    if not weights_sum.apply(lambda s: np.isclose(s, 1.0) or np.isclose(s, 0.0)).all():
        raise DataTransformationError("Exposure weights do not sum to one across demographics.")

    check_age_restrictions(data, entity, rest_type='outer', fill_value=0.0, context=context)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_relative_risk(data: pd.DataFrame, entity: Union[RiskFactor, CoverageGap],
                           context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped relative risk
    data, with the upper boundary of values determined by the distribution type.
    Because some RRs may be protective, allow the lower boundary to be 0 for
    those special cases. Otherwise, check a lower boundary of 1. Use the
    affected_measure to determine which set of age restrictions to check:
    using yll for excess mortality and the narrowest range for incidence rate.

    Additionally, for entities with categorical distributions, check that the
    theoretical minimum risk exposure level (TMREL) category contains exposure
    values of only 1.

    Parameters
    ----------
    data
        Simulation-prepped relative risk data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if the age or sex restrictions are violated, data falls
        outside the expected boundary values, or TMREL values are not all 1 for
        entities with categorical distributions.

    """
    risk_relationship = data.groupby(['affected_entity', 'affected_measure', 'parameter'])
    risk_relationship.apply(validate_standard_columns, context)

    is_continuous = entity.distribution in ['normal', 'lognormal', 'ensemble']
    is_categorical = entity.distribution in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']

    if is_categorical:
        range_kwd = 'categorical'
    elif is_continuous:
        range_kwd = 'continuous'
    else:
        raise NotImplementedError()

    protective_causes = PROTECTIVE_CAUSE_RISK_PAIRS[entity.name] if entity.name in PROTECTIVE_CAUSE_RISK_PAIRS else []
    protective = data[data.affected_entity.isin([c.name for c in protective_causes])]
    non_protective = data.loc[data.index.difference(protective.index)]

    if not protective.empty:
        check_value_columns_boundary(protective, boundary_value=0, boundary_type='lower',
                                     value_columns=['value'], inclusive=False,
                                     error=DataTransformationError)
        check_value_columns_boundary(protective, boundary_value=VALID_RELATIVE_RISK_RANGE[0],
                                     boundary_type='upper', value_columns=['value'])
    if not non_protective.empty:
        check_value_columns_boundary(non_protective, boundary_value=VALID_RELATIVE_RISK_RANGE[0],
                                     boundary_type='lower', value_columns=['value'])

    check_value_columns_boundary(non_protective, boundary_value=VALID_RELATIVE_RISK_RANGE[1][range_kwd], boundary_type='upper',
                                 value_columns=['value'], error=DataTransformationError)

    if is_categorical:
        tmrel_cat = sorted(list(entity.categories.to_dict()), key=lambda x: int(x[3:]))[-1]  # chop 'cat' and sort
        if not (data.loc[data.parameter == tmrel_cat, 'value'] == 1.0).all():
            raise DataTransformationError(f"The TMREL category {tmrel_cat} contains values other than 1.0.")

    if entity.kind in ['risk_factor', 'alternative_risk_factor']:
        if (data.affected_measure == 'incidence_rate').all():
            check_age_restrictions(data, entity, rest_type='inner', fill_value=1.0, context=context)
        else:
            check_age_restrictions(data, entity, rest_type='yll', fill_value=1.0, context=context)
        check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=1.0)


def validate_population_attributable_fraction(data: pd.DataFrame, entity: Union[RiskFactor, Etiology],
                                              context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped population
    attributable fraction data. For protective cause-risk pairs: check a hard
    lower boundary of -1, a soft upper bound of 0 and a hard upper bound of 1.
    For non-protective cause-risk pairs: check a hard lower bound of 0 and a
    hard upper bound of 1. Check age and sex restrictions based on the affected
    entity and measure.

    Parameters
    ----------
    data
        Simulation-prepped population attributable fraction data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if the age or sex restrictions are violated, or data falls
        outside the expected boundary values.

    """
    risk_relationship = data.groupby(['affected_entity', 'affected_measure'])
    risk_relationship.apply(validate_standard_columns, context)

    protective_causes = PROTECTIVE_CAUSE_RISK_PAIRS[entity.name] if entity.name in PROTECTIVE_CAUSE_RISK_PAIRS else []
    protective = data[data.affected_entity.isin([c.name for c in protective_causes])]
    non_protective = data.loc[data.index.difference(protective.index)]

    if not protective.empty:
        check_value_columns_boundary(protective, boundary_value=VALID_PROTECTIVE_PAF_MIN, boundary_type='lower',
                                     value_columns=['value'], error=DataTransformationError)
        check_value_columns_boundary(protective, boundary_value=VALID_PAF_RANGE[0], boundary_type='upper',
                                     value_columns=['value'])
        check_value_columns_boundary(protective, boundary_value=VALID_PAF_RANGE[1], boundary_type='upper',
                                     value_columns=['value'], error=DataTransformationError)
    if not non_protective.empty:
        check_value_columns_boundary(non_protective, boundary_value=VALID_PAF_RANGE[0],
                                     boundary_type='lower', value_columns=['value'],
                                     error=DataTransformationError)
        check_value_columns_boundary(non_protective, boundary_value=VALID_PAF_RANGE[1],
                                     boundary_type='upper', value_columns=['value'],
                                     error=DataTransformationError)

    for (c_name, measure), g in risk_relationship:
        cause = [c for c in causes if c.name == c_name][0]
        if measure == 'incidence_rate':
            check_age_restrictions(g, cause, rest_type='yld', fill_value=0.0, context=context)
        else:  # excess mortality
            check_age_restrictions(g, cause, rest_type='yll', fill_value=0.0, context=context)
        check_sex_restrictions(g, cause.restrictions.male_only, cause.restrictions.female_only, fill_value=0.0)


def validate_mediation_factors(data: pd.DataFrame, entity: RiskFactor, context: SimulationValidationContext) -> None:
    raise NotImplementedError()


def validate_estimate(data: pd.DataFrame, entity: Covariate, context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped covariate
    estimate data, adjusting as needed for covariates that are not by age or
    by sex to skip those column checks. Check that the lower, mean, and upper
    values for each demographic group are either all 0 or lower < mean < upper.

    Parameters
    ----------
    data
        Simulation-prepped covariate estimate data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if yld age or sex restrictions are violated, or data does not satisfy
        lower = mean = upper = 0 or lower < mean < upper for each demographic
        group.

    """
    cols = ['location', 'year_start', 'year_end']
    validate_location_column(data, context)
    if entity.by_sex:
        validate_sex_column(data)
        cols += ['sex']
    if entity.by_age:
        validate_age_columns(data, context=context)
        cols += ['age_group_start', 'age_group_end']
    validate_year_columns(data, context)
    validate_value_column(data)

    data.groupby(cols).apply(check_covariate_values)


def validate_cost(data: pd.DataFrame, entity: Union[HealthTechnology, HealthcareEntity],
                  context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped cost data.

    Parameters
    ----------
    data
        Simulation-prepped cost data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if yld age or sex restrictions are violated, or data falls outside the
        expected boundary values.

    """
    validate_standard_columns(data, context)
    check_value_columns_boundary(data, VALID_COST_RANGE[0], 'lower',
                                 value_columns=['value'], inclusive=True,
                                 error=DataTransformationError)
    check_value_columns_boundary(data, VALID_COST_RANGE[1][entity.kind], 'upper',
                                 value_columns=['value'], inclusive=True,
                                 error=DataTransformationError)


def validate_utilization(data: pd.DataFrame, entity: HealthcareEntity, context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped utilization
    data.

    Parameters
    ----------
    data
        Simulation-prepped cost utilization to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if yld age or sex restrictions are violated, or data falls outside the
        expected boundary values.

    """
    validate_standard_columns(data, context)
    check_value_columns_boundary(data, VALID_UTILIZATION_RANGE[0], 'lower',
                                 value_columns=['value'], inclusive=True,
                                 error=DataTransformationError)
    check_value_columns_boundary(data, VALID_UTILIZATION_RANGE[1], 'upper',
                                 value_columns=['value'], inclusive=True,
                                 error=DataTransformationError)


def validate_structure(data: pd.DataFrame, entity: Population, context: SimulationValidationContext) -> None:
    """Check the standard set of validations on simulation-prepped population
    structure data, skipping the check on the draw column since structure data
    isn't by draw.

    Parameters
    ----------
    data
        Simulation-prepped population structure data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any standard columns are incorrectly named or contain invalid values,
        if yld age or sex restrictions are violated, or data falls outside the
        expected boundary values.

    """
    validate_demographic_columns(data, context)
    validate_value_column(data)

    check_value_columns_boundary(data, VALID_POPULATION_RANGE[0], 'lower',
                                 value_columns=['value'], inclusive=True,
                                 error=DataTransformationError)
    check_value_columns_boundary(data, VALID_POPULATION_RANGE[1], 'upper',
                                 value_columns=['value'], inclusive=True,
                                 error=DataTransformationError)


def validate_theoretical_minimum_risk_life_expectancy(data: pd.DataFrame, entity: Population,
                                                      context: SimulationValidationContext) -> None:
    """ Because the structure of life expectancy data is somewhat different,
    containing a custom age column that doesn't match the standard GBD age
    groups, this validator doesn't use the standard column checks. Instead, it
    verifies that the data has the correct age columns and that ages range from
    0 to 110 years. It checks that all life expectancy values are within the
    expected range and ensures that life expectancy is monotonically decreasing
    by age.

    Parameters
    ----------
    data
        Simulation-prepped theoretical minimum risk life expectancy data.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If age columns are incorrectly named or contain invalid values or if
        any life expectancy values are outside the expected range or not
        monotonically decreasing over age.

    """
    if 'age_group_start' not in data.columns or 'age_group_end' not in data.columns:
        raise DataTransformationError("Age data must be contained in columns named "
                                      "'age_group_start' and 'age_group_end'.")

    age_min, age_max = 0, 110
    if data.age_group_start.min() > age_min or data.age_group_start.max() < age_max:
        raise DataTransformationError(f'Life expectancy data does not span the '
                                      f'entire age range [{age_min}, {age_max}].')

    check_value_columns_boundary(data, VALID_LIFE_EXP_RANGE[0], 'lower',
                                 value_columns=['value'], inclusive=False,
                                 error=DataTransformationError)
    check_value_columns_boundary(data, VALID_LIFE_EXP_RANGE[1], 'upper',
                                 value_columns=['value'], inclusive=False,
                                 error=DataTransformationError)

    if not data.sort_values(by='age_group_start', ascending=False).value.is_monotonic:
        raise DataTransformationError('Life expectancy data is not monotonically decreasing over age.')


def validate_age_bins(data: pd.DataFrame, entity: Population, context: SimulationValidationContext) -> None:
    """With only age columns in this data, the validator is an abbreviated
    version employing only the standard column check on ages.

    Parameters
    ----------
    data
        Simulation-prepped age bin data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any age columns are incorrectly named or contain invalid values.

    """
    validate_age_columns(data, context=context)


def validate_demographic_dimensions(data: pd.DataFrame, entity: Population,
                                    context: SimulationValidationContext) -> None:
    """With only demographic columns in this data, the validator is an
    abbreviated version employing only the standard demographic column checks.

    Parameters
    ----------
    data
        Simulation-prepped demographic dimensions data to validate.
    entity
        Entity to which the data pertain.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any demographic columns are incorrectly named or contain invalid
        values.

    """
    validate_demographic_columns(data, context)


#############
# UTILITIES #
#############


def validate_standard_columns(data: pd.DataFrame, context: SimulationValidationContext) -> None:
    """Validate that location, sex, age, year, draw, and value columns in the
    passed dataframe all have the expected names and values.

    Parameters
    ----------
    data
        Simulation-prepped data to validate.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any location, sex, age, year, draw, or value columns are incorrectly
        named or contain invalid values.

    """
    validate_demographic_columns(data, context)
    validate_draw_column(data)
    validate_value_column(data)


def validate_demographic_columns(data: pd.DataFrame, context: SimulationValidationContext) -> None:
    """Validate that demographic (location, sex, age, and year) columns in the
    passed dataframe all have the expected names and values. The given context
    provides the full range of expected values for location, age, and year.

    Parameters
    ----------
    data
        Simulation-prepped data to validate.
    context
        Wrapper for additional data used in the validation process.

    Raises
    ------
    DataTransformationError
        If any demographic columns are incorrectly named or contain invalid
        values.

    """
    validate_location_column(data, context)
    validate_sex_column(data)
    validate_age_columns(data, context=context)
    validate_year_columns(data, context)


def validate_draw_column(data: pd.DataFrame) -> None:
    """Validate that draw column in the data has the expected name and values.

    Parameters
    ----------
    data
        Simulation-prepped data to validate.

    Raises
    ------
    DataTransformationError
        If data does not contain a column named 'draw' or that column does not
        contain all values in the range [0, 999].

    """
    if 'draw' not in data.columns:
        raise DataTransformationError("Draw data must be contained in a column named 'draw'.")

    if set(data['draw']) != set(range(1000)):
        raise DataTransformationError('Draw must contain [0, 999].')


def validate_location_column(data: pd.DataFrame, context: SimulationValidationContext) -> None:
    """Validate that location column in the data has the expected name
    and value.

    Parameters
    ----------
    data
        Simulation-prepped data to validate.
    context
        Wrapper for additional data used in the validation process

    Raises
    ------
    DataTransformationError
        If data does not contain a column named 'location' or that column does
        not only the single location given in `context`.

    """
    if 'location' not in data.columns:
        raise DataTransformationError("Location data must be contained in a column named 'location'.")

    if len(data['location'].unique()) != 1 or data['location'].unique()[0] != context['location']:
        raise DataTransformationError('Location must contain a single value that matches specified location.')


def validate_sex_column(data: pd.DataFrame) -> None:
    """Validate that sex column in the data has the expected name and values.

    Parameters
    ----------
    data
        Simulation-prepped data to validate.

    Raises
    ------
    DataTransformationError
        If data does not contain a column named 'sex' or that column does not
        contain only the values 'Male' and 'Female'.

    """
    if 'sex' not in data.columns:
        raise DataTransformationError("Sex data must be contained in a column named 'sex'.")

    if set(data['sex']) != {'Male', 'Female'}:
        raise DataTransformationError("Sex must contain 'Male' and 'Female' and nothing else.")


def validate_age_columns(data: pd.DataFrame, context: SimulationValidationContext) -> None:
    """Validate that age columns in the data have the expected names and values.

    Parameters
    ----------
    data
        Simulation-prepped data to validate.
    context
        Wrapper for additional data used in validation.

    Raises
    ------
    DataTransformationError
        If data does not contain columns named 'age_grouo_start' and
        'age_group_end' or if those columns do not contain the full range of
        expected age bins supplied in `context`.

    """
    if 'age_group_start' not in data.columns or 'age_group_end' not in data.columns:
        raise DataTransformationError("Age data must be contained in columns named"
                                      " 'age_group_start' and 'age_group_end'.")

    expected_ages = (context['age_bins']
                     .filter(['age_group_start', 'age_group_end'])
                     .sort_values(['age_group_start', 'age_group_end']))
    age_block = (data[['age_group_start', 'age_group_end']]
                 .drop_duplicates()
                 .sort_values(['age_group_start', 'age_group_end'])
                 .reset_index(drop=True))

    if not age_block.equals(expected_ages):
        raise DataTransformationError('Age_group_start and age_group_end must contain all gbd age groups.')


def validate_year_columns(data: pd.DataFrame, context: SimulationValidationContext) -> None:
    """Validate that year columns in the data have the expected names and
    values.

    Parameters
    ----------
    data
        Simulation-prepped data to validate.
    context
        Wrapper for additional data used in validation.

    Raises
    ------
    DataTransformationError
        If data does not contain columns named 'year_start' and
        'year_end' or if those columns do not contain the full range of
        expected year bins supplied in `context`.

    """
    if 'year_start' not in data.columns or 'year_end' not in data.columns:
        raise DataTransformationError("Year data must be contained in columns named 'year_start', and 'year_end'.")

    expected_years = context['years'].sort_values(['year_start', 'year_end'])
    year_block = (data[['year_start', 'year_end']]
                  .drop_duplicates()
                  .sort_values(['year_start', 'year_end'])
                  .reset_index(drop=True))

    if not year_block.equals(expected_years):
        raise DataTransformationError('Year_start and year_end must cover [1990, 2017] in intervals of one year.')


def validate_value_column(data: pd.DataFrame) -> None:
    """Validate that value column in the data has the expected name and no
    missing values.

    Parameters
    ----------
    data
        Simulation-prepped data to validate.
    context
        Wrapper for additional data used in validation.

    Raises
    ------
    DataTransformationError
        If data does not contain a column named `value` or that column contains
        any NaN or Inf values.

    """
    if 'value' not in data.columns:
        raise DataTransformationError("Value data must be contained in a column named 'value'.")

    if np.any(data.value.isna()):
        raise DataTransformationError('Value data found to contain NaN.')
    if np.any(np.isinf(data.value.values)):
        raise DataTransformationError('Value data found to contain infinity.')


def check_age_restrictions(data: pd.DataFrame, entity: ModelableEntity, rest_type: str,
                           fill_value: Union[float, Dict[str, float]], context: SimulationValidationContext):
    """Given an entity and which restrictions to use, ensure that all data for
    age groups outside of the range of restrictions for that entity has values
    only of fill_value.

    Parameters
    ----------
    data
        Simulation-prepped data to validate.
    entity
        Entity for which to validate restrictions.
    rest_type
        Which restrictions from entity to use. One of: yll, yld, inner, outer.
    fill_value
        The only allowable value in data outside of age restrictions. For
        categorical risks, this should be dictionary containing values for
        'exposed' and 'unexposed' categories.
    context
        Wrapper containing additional data used in the simulation.

    Raises
    ------
    DataTransformationError
        If any values other than fill_value are found in data outside the
        restrictions of entity.

    """
    start_id, end_id = utilities.get_age_group_ids_by_restriction(entity, rest_type)
    age_bins = context['age_bins']
    age_start = float(age_bins.loc[age_bins.age_group_id == start_id, 'age_group_start'])
    age_end = float(age_bins.loc[age_bins.age_group_id == end_id, 'age_group_end'])

    outside = data.loc[(data.age_group_start < age_start) | (data.age_group_end > age_end)]

    if (entity.kind in ['risk_factor', 'alternative_risk_factor'] and
            entity.distribution in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous'] and
            isinstance(fill_value, dict)):
        _check_cat_risk_fill_values(outside, entity, fill_value, 'age')

    elif not outside.empty and (outside.value != fill_value).any():
        raise DataTransformationError(f"Age restrictions are violated by a value other than fill={fill_value}.")


def check_sex_restrictions(data: pd.DataFrame, male_only: bool, female_only: bool,
                           fill_value: Union[float, Dict[str, float]], entity=None):
    """Given an entity and which restrictions to use, ensure that all data for
    sexes outside of the restrictions for that entity has values only of
    fill_value.

    Parameters
    ----------
    data
        Simulation-prepped data to validate.
    female_only
        Boolean indicating whether the data should be restricted to females
        only. If true, all male data should have values only of fill_value.
    male_only
        Boolean indicating whether the data should be restricted to females
        only. If true, all female data should have values only of fill_value.
    fill_value
        The only allowable value in data outside of age restrictions. For
        categorical risks, this should be dictionary containing values for
        'exposed' and 'unexposed' categories.
    context
        Wrapper containing additional data used in the simulation.
    entity
        Optional. Used to check if the entity is a categorical risk, in which
        case only used for exposure data validation where the fill_values vary
        by category.

    Raises
    ------
    DataTransformationError
        If any values other than fill_value are found in data outside the
        restrictions of entity.

    """
    outside = None
    if male_only:
        outside = data[data.sex == 'Female']
        sex = 'male'
    elif female_only:
        outside = data[data.sex == 'Male']
        sex = 'female'
    if outside is not None:
        if entity is not None and (entity.kind in ['risk_factor', 'alternative_risk_factor'] and
                                   entity.distribution in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']
                                   and isinstance(fill_value, dict)):
            _check_cat_risk_fill_values(outside, entity, fill_value, 'sex')

        elif (outside.value != fill_value).any():
            raise DataTransformationError(f"Restriction to {sex} sex only is violated "
                                          f"by a value other than fill={fill_value}.")


def _check_cat_risk_fill_values(outside_data: pd.DataFrame, entity: Union[RiskFactor, AlternativeRiskFactor],
                                fill_value: Dict[str, float], restriction: str):
    """Helper method for checking restrictions for categorical risks where two
    fill values are allowed: one for exposed categories and one for the
    unexposed category.

    Parameters
    ----------
    outside_data
        Simulation-prepped data outside the restrictions.
    entity
        Entity to which the data pertain.
    fill_value
        Dictionary containing the fill values for the exposed and unexposed
        categories.
    restriction
        Whether the restriction being checked is 'sex' or 'age'.

    Raises
    ------
    DataTransformationError
        If the outside_data contains values other than fill_value for the
        correct categories.

    """
    tmrel_cat = sorted(list(entity.categories.to_dict()), key=lambda x: int(x[3:]))[-1]
    outside_unexposed = outside_data[outside_data.parameter == tmrel_cat]
    outside_exposed = outside_data[outside_data.parameter != tmrel_cat]
    if not outside_unexposed.empty and (outside_unexposed.value != fill_value['unexposed']).any():
        raise DataTransformationError(f'{restriction.capitalize()} restrictions for TMREL cat are violated by a '
                                      f'value other than fill={fill_value["unexposed"]}.')
    if not outside_exposed.empty and (outside_exposed.value != fill_value['exposed']).any():
        raise DataTransformationError(f'{restriction.capitalize()} restrictions for non-TMREL categories are violated '
                                      f'by a value other than fill={fill_value["exposed"]}.')


def check_covariate_values(data: pd.DataFrame) -> None:
    """Validator for covariate estimate data to check that for each demographic
    group either lower, mean, and upper values are all 0 or lower < mean < upper.

    Parameters
    ----------
    data
        Simulation-prepped covariate estimate data for a single demographic
        group.

    Raises
    ------
    DataTransformationError
        If lower, mean, and upper values are not all 0 and it is not the case
         that lower < mean < upper.

    """
    lower = data[data.parameter == 'lower_value'].value.values
    mean = data[data.parameter == 'mean_value'].value.values
    upper = data[data.parameter == 'upper_value'].value.values

    # allow the case where lower = mean = upper = 0 b/c of things like age
    # specific fertility rate where all estimates are 0 for young age groups
    if np.all(data.value != 0) and not np.all(lower < mean < upper):
        raise DataTransformationError('Covariate data contains demographic groups for which the '
                                      'estimates for lower, mean, and upper values are not all 0 '
                                      'and it is not the case that lower < mean < upper. ')

