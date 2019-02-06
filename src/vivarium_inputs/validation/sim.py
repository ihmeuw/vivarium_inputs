from typing import Union

import numpy as np
import pandas as pd

from gbd_mapping import ModelableEntity, Cause, Sequela, RiskFactor, CoverageGap, Etiology, Covariate, causes

from vivarium_inputs import utilities, utility_data
from vivarium_inputs.globals import DataTransformationError, Population, PROTECTIVE_CAUSE_RISK_PAIRS
from vivarium_inputs.mapping_extension import HealthcareEntity, HealthTechnology, AlternativeRiskFactor
from vivarium_inputs.validation.shared import check_value_columns_boundary


VALID_INCIDENCE_RANGE = (0.0, 50.0)
VALID_PREVALENCE_RANGE = (0.0, 1.0)
VALID_BIRTH_PREVALENCE_RANGE = (0.0, 1.0)
VALID_DISABILITY_WEIGHT_RANGE = (0.0, 1.0)
VALID_REMISSION_RANGE = (0.0, 120.0)  # James' head
VALID_CAUSE_SPECIFIC_MORTALITY_RANGE = (0.0, 0.4)  # used mortality viz, picked worst country 15q45, mul by ~1.25
VALID_EXCESS_MORT_RANGE = (0.0, 120.0)  # James' head
VALID_EXPOSURE_RANGE = (0.0, {'continuous': 10_000.0, 'categorical': 1.0})
VALID_EXPOSURE_SD_RANGE = (0.0, 1000.0)  # James' brain
VALID_EXPOSURE_DIST_WEIGHTS_RANGE = (0.0, 1.0)
VALID_RELATIVE_RISK_RANGE = (1.0, {'continuous': 10.0, 'categorical': 20.0})
VALID_PAF_RANGE = (0.0, 1.0)
VALID_PROTECTIVE_PAF_MIN = -1.0
VALID_COST_RANGE = (0, {'healthcare_entity': 30_000, 'health_technology': 50})
VALID_UTILIZATION_RANGE = (0, 50)
VALID_POPULATION_RANGE = (0, 100_000_000)
VALID_LIFE_EXP_RANGE = (0, 90)


class SimulationValidationContext:

    def __init__(self, location, estimation_years: pd.DataFrame = None, **kwargs):
        self.context_data = {'location': location}
        if estimation_years is None:
            self.context_data['years'] = utility_data.get_year_block()
        else:
            self.context_data['years'] = estimation_years

        self.context_data.update(kwargs)

    def __getitem__(self, key):
        return self.context_data[key]

    def __setitem__(self, key, value):
        self.context_data[key] = value


def validate_for_simulation(data: pd.DataFrame, entity: ModelableEntity, measure: str, location: str, **context_args):
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


def validate_incidence(data: pd.DataFrame, entity: Union[Cause, Sequela], context: SimulationValidationContext):
    validate_standard_columns(data, context)

    check_value_columns_boundary(data, boundary_value=VALID_INCIDENCE_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_INCIDENCE_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    check_age_restrictions(data, entity, rest_type='yld', fill_value=0.0)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_prevalence(data: pd.DataFrame, entity: Union[Cause, Sequela], context: SimulationValidationContext):
    validate_standard_columns(data, context)

    check_value_columns_boundary(data, boundary_value=VALID_PREVALENCE_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_PREVALENCE_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    check_age_restrictions(data, entity, rest_type='yld', fill_value=0.0)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_birth_prevalence(data: pd.DataFrame, entity: Union[Cause, Sequela], context: SimulationValidationContext):
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

    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_disability_weight(data: pd.DataFrame, entity: Union[Cause, Sequela], context: SimulationValidationContext):
    validate_standard_columns(data, context)

    check_value_columns_boundary(data, boundary_value=VALID_DISABILITY_WEIGHT_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_DISABILITY_WEIGHT_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    check_age_restrictions(data, entity, rest_type='yld', fill_value=0.0)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_remission(data: pd.DataFrame, entity: Cause, context: SimulationValidationContext):
    validate_standard_columns(data, context)

    check_value_columns_boundary(data, boundary_value=VALID_REMISSION_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_REMISSION_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    check_age_restrictions(data, entity, rest_type='yld', fill_value=0.0)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_cause_specific_mortality(data: pd.DataFrame, entity: Cause, context: SimulationValidationContext):
    validate_standard_columns(data, context)

    check_value_columns_boundary(data, boundary_value=VALID_CAUSE_SPECIFIC_MORTALITY_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_CAUSE_SPECIFIC_MORTALITY_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    check_age_restrictions(data, entity, rest_type='yll', fill_value=0.0)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_excess_mortality(data: pd.DataFrame, entity: Cause, context: SimulationValidationContext):
    validate_standard_columns(data, context)

    check_value_columns_boundary(data, boundary_value=VALID_EXCESS_MORT_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_EXCESS_MORT_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    check_age_restrictions(data, entity, rest_type='yll', fill_value=0.0)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_exposure(data: pd.DataFrame, entity: Union[RiskFactor, CoverageGap, AlternativeRiskFactor],
                      context: SimulationValidationContext):
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

    check_age_restrictions(data, entity, rest_type='outer', fill_value=0.0)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_exposure_standard_deviation(data: pd.DataFrame, entity: Union[RiskFactor, AlternativeRiskFactor],
                                         context: SimulationValidationContext):
    validate_standard_columns(data, context)

    check_value_columns_boundary(data, boundary_value=VALID_EXPOSURE_SD_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_EXPOSURE_SD_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    check_age_restrictions(data, entity, rest_type='outer', fill_value=0.0)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_exposure_distribution_weights(data: pd.DataFrame, entity: Union[RiskFactor, AlternativeRiskFactor],
                                           context: SimulationValidationContext):
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

    check_age_restrictions(data, entity, rest_type='outer', fill_value=0.0)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_relative_risk(data: pd.DataFrame, entity: Union[RiskFactor, CoverageGap], context: SimulationValidationContext):
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

    #  We want to have hard lower limit 0 for RR and soft low limit 1 for RR
    #  because some risks are protective against some causes.
    check_value_columns_boundary(data, boundary_value=VALID_RELATIVE_RISK_RANGE[0],
                                 boundary_type='lower', value_columns=['value'])
    check_value_columns_boundary(data, boundary_value=0, boundary_type='lower',
                                 value_columns=['value'], inclusive=False,
                                 error=DataTransformationError)
    check_value_columns_boundary(data, boundary_value=VALID_RELATIVE_RISK_RANGE[1][range_kwd],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    if is_categorical:
        tmrel_cat = sorted(list(entity.categories.to_dict()), key=lambda x: int(x[3:]))[-1]  # chop 'cat' and sort
        if not (data.loc[data.parameter == tmrel_cat, 'value'] == 1.0).all():
            raise DataTransformationError(f"The TMREL category {tmrel_cat} contains values other than 1.0.")

    if (data.affected_measure == 'incidence_rate').all():
        check_age_restrictions(data, entity, rest_type='inner', fill_value=1.0)
    else:
        check_age_restrictions(data, entity, rest_type='yll', fill_value=1.0)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=1.0)


def validate_population_attributable_fraction(data: pd.DataFrame, entity: Union[RiskFactor, Etiology],
                                              context: SimulationValidationContext):

    protective_causes = PROTECTIVE_CAUSE_RISK_PAIRS[entity.name] if entity.name in PROTECTIVE_CAUSE_RISK_PAIRS else []
    protective = data[data.affected_entity.isin(protective_causes)]
    non_protective = data.loc[data.index.difference(protective.index)]

    risk_relationship = data.groupby(['affected_entity', 'affected_measure'])
    risk_relationship.apply(validate_standard_columns, context)

    if protective and not protective.empty:
        check_value_columns_boundary(protective, boundary_value=VALID_PROTECTIVE_PAF_MIN, boundary_type='lower',
                                     value_columns=['value'], error=DataTransformationError)
        check_value_columns_boundary(protective, boundary_value=VALID_PAF_RANGE[0], boundary_type='upper',
                                     value_columns=['value'])
        check_value_columns_boundary(protective, boundary_value=VALID_PAF_RANGE[1], boundary_type='upper',
                                     value_columns=['value'], error=DataTransformationError)

    check_value_columns_boundary(non_protective, boundary_value=VALID_PAF_RANGE[0],
                                 boundary_type='lower', value_columns=['value'],
                                 error=DataTransformationError)
    check_value_columns_boundary(non_protective, boundary_value=VALID_PAF_RANGE[1],
                                 boundary_type='upper', value_columns=['value'],
                                 error=DataTransformationError)

    for (entity, measure), g in risk_relationship:
        cause = [c for c in causes if c.name == entity][0]
        if measure == 'incidence_rate':
            check_age_restrictions(g, cause, rest_type='yll', fill_value=0.0)
        else:  # excess mortality
            check_age_restrictions(g, cause, rest_type='yld', fill_value=0.0)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def validate_mediation_factors(data: pd.DataFrame, entity: RiskFactor, context: SimulationValidationContext):
    raise NotImplementedError()


def validate_estimate(data: pd.DataFrame, entity: Covariate, context: SimulationValidationContext):
    cols = ['location', 'year_start', 'year_end']
    validate_location_column(data, context)
    if entity.by_sex:
        validate_sex_column(data)
        cols += ['sex']
    if entity.by_age:
        validate_age_columns(data)
        cols += ['age_group_start', 'age_group_end']
    validate_year_columns(data, context)
    validate_value_column(data)

    data.groupby(cols).apply(check_covariate_values)


def validate_cost(data: pd.DataFrame, entity: Union[HealthTechnology, HealthcareEntity],
                  context: SimulationValidationContext):
    validate_standard_columns(data, context)
    check_value_columns_boundary(data, VALID_COST_RANGE[0], 'lower',
                                 value_columns=['value'], inclusive=True,
                                 error=DataTransformationError)
    check_value_columns_boundary(data, VALID_COST_RANGE[1][entity.kind], 'upper',
                                 value_columns=['value'], inclusive=True,
                                 error=DataTransformationError)


def validate_utilization(data: pd.DataFrame, entity: HealthcareEntity, context: SimulationValidationContext):
    validate_standard_columns(data, context)
    check_value_columns_boundary(data, VALID_UTILIZATION_RANGE[0], 'lower',
                                 value_columns=['value'], inclusive=True,
                                 error=DataTransformationError)
    check_value_columns_boundary(data, VALID_UTILIZATION_RANGE[1], 'upper',
                                 value_columns=['value'], inclusive=True,
                                 error=DataTransformationError)


def validate_structure(data: pd.DataFrame, entity: Population, context: SimulationValidationContext):
    validate_demographic_columns(data, context)
    validate_value_column(data)

    check_value_columns_boundary(data, VALID_POPULATION_RANGE[0], 'lower',
                                 value_columns=['value'], inclusive=True,
                                 error=DataTransformationError)
    check_value_columns_boundary(data, VALID_POPULATION_RANGE[1], 'upper',
                                 value_columns=['value'], inclusive=True,
                                 error=DataTransformationError)


def validate_theoretical_minimum_risk_life_expectancy(data: pd.DataFrame, entity: Population,
                                                      context: SimulationValidationContext):
    if 'age_group_start' not in data.columns or 'age_group_end' not in data.columns:
        raise DataTransformationError("Age data must be contained in columns named "
                                      "'age_group_start' and 'age_group_end'.")

    age_min, age_max = 0, 110
    if data.age_group_start.min() > age_min or data.age_group_start.max() < age_max:
        raise DataTransformationError(f'Life expectancy data does not span the entire age range [{age_min}, {age_max}].')

    check_value_columns_boundary(data, VALID_LIFE_EXP_RANGE[0], 'lower',
                                 value_columns=['value'], inclusive=False,
                                 error=DataTransformationError)
    check_value_columns_boundary(data, VALID_LIFE_EXP_RANGE[1], 'upper',
                                 value_columns=['value'], inclusive=False,
                                 error=DataTransformationError)

    if not data.sort_values(by='age_group_start', ascending=False).value.is_monotonic:
        raise DataTransformationError('Life expectancy data is not monotonically decreasing over age.')


def validate_age_bins(data: pd.DataFrame, entity: Population, context: SimulationValidationContext):
    validate_age_columns(data)


def validate_demographic_dimensions(data: pd.DataFrame, entity: Population, context: SimulationValidationContext):
    validate_demographic_columns(data, context)


#############
# UTILITIES #
#############


def validate_standard_columns(data: pd.DataFrame, context: SimulationValidationContext):
    validate_demographic_columns(data, context)
    validate_draw_column(data)
    validate_value_column(data)


def validate_demographic_columns(data: pd.DataFrame, context: SimulationValidationContext):
    validate_location_column(data, context)
    validate_sex_column(data)
    validate_age_columns(data)
    validate_year_columns(data, context)


def validate_draw_column(data: pd.DataFrame):
    if 'draw' not in data.columns:
        raise DataTransformationError("Draw data must be contained in a column named 'draw'.")

    if set(data['draw']) != set(range(1000)):
        raise DataTransformationError('Draw must contain [0, 999].')


def validate_location_column(data: pd.DataFrame, context: SimulationValidationContext):
    if 'location' not in data.columns:
        raise DataTransformationError("Location data must be contained in a column named 'location'.")

    if len(data['location'].unique()) != 1 or data['location'].unique()[0] != context['location']:
        raise DataTransformationError('Location must contain a single value that matches specified location.')


def validate_sex_column(data: pd.DataFrame):
    if 'sex' not in data.columns:
        raise DataTransformationError("Sex data must be contained in a column named 'sex'.")

    if set(data['sex']) != {'Male', 'Female'}:
        raise DataTransformationError("Sex must contain 'Male' and 'Female' and nothing else.")


def validate_age_columns(data: pd.DataFrame):
    if 'age_group_start' not in data.columns or 'age_group_end' not in data.columns:
        raise DataTransformationError("Age data must be contained in columns named"
                                      " 'age_group_start' and 'age_group_end'.")

    expected_ages = utilities.get_age_bins()[['age_group_start', 'age_group_end']].sort_values(['age_group_start',
                                                                                                'age_group_end'])
    age_block = (data[['age_group_start', 'age_group_end']]
                 .drop_duplicates()
                 .sort_values(['age_group_start', 'age_group_end'])
                 .reset_index(drop=True))

    if not age_block.equals(expected_ages):
        raise DataTransformationError('Age_group_start and age_group_end must contain all gbd age groups.')


def validate_year_columns(data: pd.DataFrame, context: SimulationValidationContext):
    if 'year_start' not in data.columns or 'year_end' not in data.columns:
        raise DataTransformationError("Year data must be contained in columns named 'year_start', and 'year_end'.")

    expected_years = context['years'].sort_values(['year_start', 'year_end'])
    year_block = (data[['year_start', 'year_end']]
                  .drop_duplicates()
                  .sort_values(['year_start', 'year_end'])
                  .reset_index(drop=True))

    if not year_block.equals(expected_years):
        raise DataTransformationError('Year_start and year_end must cover [1990, 2017] in intervals of one year.')


def validate_value_column(data: pd.DataFrame):
    if 'value' not in data.columns:
        raise DataTransformationError("Value data must be contained in a column named 'value'.")

    if np.any(data.value.isna()):
        raise DataTransformationError('Value data found to contain NaN.')
    if np.any(np.isinf(data.value.values)):
        raise DataTransformationError('Value data found to contain infinity.')


def check_age_restrictions(data: pd.DataFrame, entity: ModelableEntity, rest_type: str, fill_value: float):

    start_id, end_id = utilities.get_age_group_ids_by_restriction(entity, rest_type)
    age_bins = utilities.get_age_bins()
    age_start = float(age_bins.loc[age_bins.age_group_id == start_id, 'age_group_start'])
    age_end = float(age_bins.loc[age_bins.age_group_id == end_id, 'age_group_end'])

    outside = data.loc[(data.age_group_start < age_start) | (data.age_group_end > age_end)]
    if not outside.empty and (outside.value != fill_value).any():
        raise DataTransformationError(f"Age restrictions are violated by a value other than fill={fill_value}.")


def check_sex_restrictions(data: pd.DataFrame, male_only: bool, female_only: bool, fill_value: float):
    if male_only and (data.loc[data.sex == 'Female', 'value'] != fill_value).any():
        raise DataTransformationError(f"Restriction to male sex only is violated by a value other than fill={fill_value}.")
    elif female_only and (data.loc[data.sex == 'Male', 'value'] != fill_value).any():
        raise DataTransformationError(f"Restriction to female sex only is violated by a value other than fill={fill_value}.")


def check_covariate_values(data: pd.DataFrame):
    lower = data[data.parameter == 'lower_value'].value.values
    mean = data[data.parameter == 'mean_value'].value.values
    upper = data[data.parameter == 'upper_value'].value.values

    # allow the case where lower = mean = upper = 0 b/c of things like age
    # specific fertility rate where all estimates are 0 for young age groups
    if np.all(data.value != 0) and not np.all(lower < mean < upper):
        raise DataTransformationError('Covariate data contains demographic groups for which the '
                                      'estimates for lower, mean, and upper values are not all 0 '
                                      'and it is not the case that lower < mean < upper. ')
