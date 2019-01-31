from typing import Sequence, Union, NamedTuple

import numpy as np
import pandas as pd

from gbd_mapping import ModelableEntity, Cause, Sequela, RiskFactor, CoverageGap, Etiology
from vivarium_inputs import utilities
from vivarium_inputs.validation import utilities as validation_utilities
from vivarium_inputs.globals import DataFormattingError
from vivarium_inputs.mapping_extension import HealthcareEntity, HealthTechnology, AlternativeRiskFactor


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
VALID_RELATIVE_RISK_RANGE = (1.0, {'continuous': 5.0, 'categorical': 20.0})
VALID_PAF_RANGE = (0.0, 1.0)
VALID_COST_RANGE = (0, {'healthcare_entity': 30_000, 'health_technology': 50})
VALID_UTILIZATION_RANGE = (0, 50)
VALID_POPULATION_RANGE = (0, 100_000_000)
VALID_LIFE_EXP_RANGE = (0, 90)


def validate_for_simulation(data: pd.DataFrame, entity: Union[ModelableEntity, NamedTuple], measure: str,
                            location: str):
    """Validate data conforms to the format that is expected by the simulation and conforms to normal expectations for a
    measure.

    Data coming in to the simulation is expected to have a full demographic set in most instances as well non-missing,
    non-infinite, reasonable data. This function enforces column names, the demographic extents, and expected ranges and
    relationships of measure data.

    Parameters
    ----------
    data
        Data to be validated.
    entity
        The GBD Entity the data pertains to.
    measure
        The measure the data pertains to.
    location
        The location the data pertains to.

    Raises
    -------
    DataFormattingError
        If any columns are mis-formatted or assumptions about the data are violated.
    """
    validators = {
        # Cause-like measures
        'incidence': _validate_incidence,
        'prevalence': _validate_prevalence,
        'birth_prevalence': _validate_birth_prevalence,
        'disability_weight': _validate_disability_weight,
        'remission': _validate_remission,
        'cause_specific_mortality': _validate_cause_specific_mortality,
        'excess_mortality': _validate_excess_mortality,
        'case_fatality': _validate_case_fatality,
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
        'age_bins': _validate_age_bins,
        'demographic_dimensions': _validate_demographic_dimensions,
    }

    if measure not in validators:
        raise NotImplementedError()

    validators[measure](data, entity, location)


def _validate_incidence(data: pd.DataFrame, entity: Union[Cause, Sequela], location: str):
    _validate_standard_columns(data, location)

    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_INCIDENCE_RANGE[0],
                                                      boundary_type='lower', value_columns=['value'], error=True)
    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_INCIDENCE_RANGE[1],
                                                      boundary_type='upper', value_columns=['value'], error=True)

    _check_age_restrictions(data, entity, type='yld', fill_value=0.0)
    _check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def _validate_prevalence(data: pd.DataFrame, entity: Union[Cause, Sequela], location: str):
    _validate_standard_columns(data, location)

    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_PREVALENCE_RANGE[0],
                                                      boundary_type='lower', value_columns=['value'], error=True)
    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_PREVALENCE_RANGE[1],
                                                      boundary_type='upper', value_columns=['value'], error=True)

    _check_age_restrictions(data, entity, type='yld', fill_value=0.0)
    _check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def _validate_birth_prevalence(data: pd.DataFrame, entity: Union[Cause, Sequela], location: str):
    _validate_location_column(data, location)
    _validate_sex_column(data)
    _validate_year_columns(data)
    _validate_draw_column(data)
    _validate_value_column(data)

    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_BIRTH_PREVALENCE_RANGE[0],
                                                      boundary_type='lower', value_columns=['value'], error=True)
    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_BIRTH_PREVALENCE_RANGE[1],
                                                      boundary_type='upper', value_columns=['value'], error=True)

    _check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def _validate_disability_weight(data: pd.DataFrame, entity: Union[Cause, Sequela], location: str):
    _validate_standard_columns(data, location)

    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_DISABILITY_WEIGHT_RANGE[0],
                                                      boundary_type='lower', value_columns=['value'], error=True)
    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_DISABILITY_WEIGHT_RANGE[1],
                                                      boundary_type='upper', value_columns=['value'], error=True)

    _check_age_restrictions(data, entity, type='yld', fill_value=0.0)
    _check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def _validate_remission(data: pd.DataFrame, entity: Cause, location: str):
    _validate_standard_columns(data, location)

    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_REMISSION_RANGE[0],
                                                      boundary_type='lower', value_columns=['value'], error=True)
    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_REMISSION_RANGE[1],
                                                      boundary_type='upper', value_columns=['value'], error=True)

    _check_age_restrictions(data, entity, type='yld', fill_value=0.0)
    _check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def _validate_cause_specific_mortality(data: pd.DataFrame, entity: Cause, location: str):
    _validate_standard_columns(data, location)

    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_CAUSE_SPECIFIC_MORTALITY_RANGE[0],
                                                      boundary_type='lower', value_columns=['value'], error=True)
    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_CAUSE_SPECIFIC_MORTALITY_RANGE[1],
                                                      boundary_type='upper', value_columns=['value'], error=True)

    _check_age_restrictions(data, entity, type='yll', fill_value=0.0)
    _check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def _validate_excess_mortality(data: pd.DataFrame, entity: Cause, location: str):
    _validate_standard_columns(data, location)

    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_EXCESS_MORT_RANGE[0],
                                                      boundary_type='lower', value_columns=['value'], error=True)
    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_EXCESS_MORT_RANGE[1],
                                                      boundary_type='upper', value_columns=['value'], error=True)

    _check_age_restrictions(data, entity, type='yll', fill_value=0.0)
    _check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def _validate_case_fatality(data, entity, location):
    raise NotImplementedError()


def _validate_exposure(data: pd.DataFrame, entity: Union[RiskFactor, CoverageGap, AlternativeRiskFactor],
                       location: str):
    is_continuous = entity.distribution in ['normal', 'lognormal', 'ensemble']
    is_categorical = (entity.distribution in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous'])

    if is_continuous:
        if data.parameter != "continuous":
            raise DataFormattingError("Continuous exposure data should contain 'continuous' in the parameter column.")
        valid_kwd = 'continuous'
    elif is_categorical:
        if set(data.parameter) != set(entity.categories.to_dict()):  # to_dict removes None
            raise DataFormattingError("Categorical exposure data does not contain all categories in the parameter "
                                      "column.")
        valid_kwd = 'categorical'
    else:
        raise NotImplementedError()

    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_EXPOSURE_RANGE[0],
                                                      boundary_type='lower', value_columns=['value'], error=True)
    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_EXPOSURE_RANGE[1][valid_kwd],
                                                      boundary_type='upper', value_columns=['value'], error=True)

    cats = data.groupby('parameter')
    cats.apply(_validate_standard_columns, location)

    if is_categorical:
        non_categorical_columns = list(set(data.columns).difference({'parameter', 'value'}))
        if not np.allclose(data.groupby(non_categorical_columns)['value'].sum(), 1.0):
            raise DataFormattingError("Categorical exposures do not sum to one across categories")

    _check_age_restrictions(data, entity, type='outer', fill_value=0.0)
    _check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def _validate_exposure_standard_deviation(data: pd.DataFrame, entity: Union[RiskFactor, AlternativeRiskFactor],
                                          location: str):
    _validate_standard_columns(data, location)

    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_EXPOSURE_SD_RANGE[0],
                                                      boundary_type='lower', value_columns=['value'], error=True)
    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_EXPOSURE_SD_RANGE[1],
                                                      boundary_type='upper', value_columns=['value'], error=True)

    _check_age_restrictions(data, entity, type='outer', fill_value=0.0)
    _check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def _validate_exposure_distribution_weights(data: pd.DataFrame, entity: Union[RiskFactor, AlternativeRiskFactor],
                                            location: str):
    _validate_standard_columns(data, location)

    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_EXPOSURE_DIST_WEIGHTS_RANGE[0],
                                                      boundary_type='lower', value_columns=['value'], error=True)
    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_EXPOSURE_DIST_WEIGHTS_RANGE[1],
                                                      boundary_type='upper', value_columns=['value'], error=True)

    non_weight_columns = list(set(data.columns).difference({'parameter', 'value'}))
    if not np.allclose(data.groupby(non_weight_columns)['value'].sum(), 1.0):
        raise DataFormattingError("Exposure weights do not sum to one across demographics.")

    _check_age_restrictions(data, entity, type='outer', fill_value=0.0)
    _check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def _validate_relative_risk(data: pd.DataFrame, entity: Union[RiskFactor, CoverageGap], location: str):
    risk_relationship = data.groupby(['affected_entity', 'affected_measure', 'parameter'])
    risk_relationship.apply(_validate_standard_columns, location)

    is_continuous = entity.distribution in ['normal', 'lognormal', 'ensemble']
    is_categorical = (entity.distribution in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous'])

    if is_categorical:
        range_kwd = 'categorical'
    elif is_continuous:
        range_kwd = 'continuous'
    else:
        raise NotImplementedError()

    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_RELATIVE_RISK_RANGE[0],
                                                      boundary_type='lower', value_columns=['value'], error=True)
    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_RELATIVE_RISK_RANGE[1][range_kwd],
                                                      boundary_type='upper', value_columns=['value'], error=True)

    if is_categorical:
        tmrel_cat = sorted(list(entity.categories.to_dict()), key=lambda x: int(x[3:]))[-1]  # chop 'cat' and sort
        if not np.allclose(data.loc[data.parameter == tmrel_cat, 'value'], 1.0):
            raise DataFormattingError(f"The TMREL category {tmrel_cat} contains values other than 1.0.")

    if data.affected_measure == 'incidence_rate':
        _check_age_restrictions(data, entity, type='inner', fill_value=1.0)
    else:
        _check_age_restrictions(data, entity, type='yll', fill_value=1.0)
    _check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=1.0)


def _validate_population_attributable_fraction(data: pd.DataFrame, entity: Union[RiskFactor, Etiology], location: str):
    risk_relationship = data.groupby(['affected_entity', 'affected_measure'])
    risk_relationship.apply(_validate_standard_columns, location)

    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_PAF_RANGE[0],
                                                      boundary_type='lower', value_columns=['value'], error=True)
    validation_utilities.check_value_columns_boundary(data, boundary_value=VALID_PAF_RANGE[1],
                                                      boundary_type='upper', value_columns=['value'], error=True)

    _check_age_restrictions(data, entity, type='inner', fill_value=0.0)
    _check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only, fill_value=0.0)


def _validate_mediation_factors(data, entity, location):
    raise NotImplementedError()


def _validate_estimate(data, entity, location):
    raise NotImplementedError()


def _validate_cost(data: pd.DataFrame, entity: Union[HealthTechnology, HealthcareEntity], location: str):
    _validate_standard_columns(data, location)
    validation_utilities.check_value_columns_boundary(data, VALID_COST_RANGE[0], 'lower',
                                                      value_columns=['value'], inclusive=True, error=True)
    validation_utilities.check_value_columns_boundary(data, VALID_COST_RANGE[1][entity.kind], 'upper',
                                                      value_columns=['value'], inclusive=True, error=True)


def _validate_utilization(data: pd.DataFrame, entity: HealthcareEntity, location: str):
    _validate_standard_columns(data, location)
    validation_utilities.check_value_columns_boundary(data, VALID_UTILIZATION_RANGE[0], 'lower',
                                                      value_columns=['value'], inclusive=True, error=True)
    validation_utilities.check_value_columns_boundary(data, VALID_UTILIZATION_RANGE[1], 'upper',
                                                      value_columns=['value'], inclusive=True, error=True)


def _validate_structure(data: pd.DataFrame, entity: NamedTuple, location: str):
    _validate_demographic_columns(data, location)
    _validate_value_column(data)

    validation_utilities.check_value_columns_boundary(data, VALID_POPULATION_RANGE[0], 'lower',
                                                      value_columns=['value'], inclusive=True, error=True)
    validation_utilities.check_value_columns_boundary(data, VALID_POPULATION_RANGE[1], 'upper',
                                                      value_columns=['value'], inclusive=True, error=True)


def _validate_theoretical_minimum_risk_life_expectancy(data: pd.DataFrame, entity: NamedTuple, location: str):
    if 'age_group_start' not in data.columns or 'age_group_end' not in data.columns:
        raise DataFormattingError("Age data must be contained in columns named 'age_group_start' and 'age_group_end'.")

    age_min, age_max = 0, 110
    if data.age_group_start.min() > age_min or data.age_group_start.max() < age_max:
        raise DataFormattingError(f'Life expectancy data does not span the entire age range [{age_min}, {age_max}].')

    validation_utilities.check_value_columns_boundary(data, VALID_LIFE_EXP_RANGE[0], 'lower',
                                                      value_columns=['value'], inclusive=False, error=True)
    validation_utilities.check_value_columns_boundary(data, VALID_LIFE_EXP_RANGE[1], 'upper',
                                                      value_columns=['value'], inclusive=False, error=True)

    if not data.sort_values(by='age_group_start', ascending=False).value.is_monotonic:
        raise DataFormattingError('Life expectancy data is not monotonically decreasing over age.')


def _validate_age_bins(data: pd.DataFrame, entity: NamedTuple, location: str):
    _validate_age_columns(data)


def _validate_demographic_dimensions(data: pd.DataFrame, entity: NamedTuple, location: str):
    _validate_demographic_columns(data, location)


#############
# UTILITIES #
#############


def _validate_standard_columns(data: pd.DataFrame, location: str):
    _validate_demographic_columns(data, location)
    _validate_draw_column(data)
    _validate_value_column(data)


def _validate_demographic_columns(data: pd.DataFrame, location: str):
    _validate_location_column(data, location)
    _validate_sex_column(data)
    _validate_age_columns(data)
    _validate_year_columns(data)


def _validate_draw_column(data: pd.DataFrame):
    if 'draw' not in data.columns:
        raise DataFormattingError("Draw data must be contained in a column named 'draw'.")

    if set(data['draw']) != set(range(1000)):
        raise DataFormattingError('Draw must contain [0, 999].')


def _validate_location_column(data: pd.DataFrame, location: str):
    if 'location' not in data.columns:
        raise DataFormattingError("Location data must be contained in a column named 'location'.")

    if len(data['location'].unique()) != 1 or data['location'].unique()[0] != location:
        raise DataFormattingError('Location must contain a single value that matches specified location.')


def _validate_sex_column(data: pd.DataFrame):
    if 'sex' not in data.columns:
        raise DataFormattingError("Sex data must be contained in a column named 'sex'.")

    if set(data['sex']) != {'Male', 'Female'}:
        raise DataFormattingError("Sex must contain 'Male' and 'Female' and nothing else.")


def _validate_age_columns(data: pd.DataFrame):
    if 'age_group_start' not in data.columns or 'age_group_end' not in data.columns:
        raise DataFormattingError("Age data must be contained in columns named 'age_group_start' and 'age_group_end'.")

    expected_ages = utilities.get_age_bins()[['age_group_start', 'age_group_end']].sort_values(['age_group_start',
                                                                                                'age_group_end'])
    age_block = (data[['age_group_start', 'age_group_end']]
                 .drop_duplicates()
                 .sort_values(['age_group_start', 'age_group_end'])
                 .reset_index(drop=True))

    if not age_block.equals(expected_ages):
        raise DataFormattingError('Age_group_start and age_group_end must contain all gbd age groups.')


def _validate_year_columns(data: pd.DataFrame):
    if 'year_start' not in data.columns or 'year_end' not in data.columns:
        raise DataFormattingError("Year data must be contained in columns named 'year_start', and 'year_end'.")

    expected_years = utilities.get_annual_year_bins().sort_values(['year_start', 'year_end'])
    year_block = (data[['year_start', 'year_end']]
                  .drop_duplicates()
                  .sort_values(['year_start', 'year_end'])
                  .reset_index(drop=True))

    if not year_block.equals(expected_years):
        raise DataFormattingError('Year_start and year_end must cover [1990, 2017] in intervals of one year.')


def _validate_value_column(data: pd.DataFrame):
    if 'value' not in data.columns:
        raise DataFormattingError("Value data must be contained in a column named 'value'.")

    if np.any(data.value.isna()):
        raise DataFormattingError('Value data found to contain NaN.')
    if np.any(np.isinf(data.value.values)):
        raise DataFormattingError('Value data found to contain infinity.')


def _check_age_restrictions(data: pd.DataFrame, entity: ModelableEntity, rest_type: str, fill_value: float):

    start_id, end_id = utilities.get_age_group_ids_by_restriction(entity, rest_type)
    age_bins = utilities.get_age_bins()
    age_start = float(age_bins.loc[age_bins.age_group_id == start_id, 'age_group_start'])
    age_end = float(age_bins.loc[age_bins.age_group_id == end_id, 'age_group_end'])

    outside = data.loc[(data.age_group_start < age_start) | (data.age_group_end > age_end)]
    if not outside.empty and (outside.value != fill_value).any():
        raise DataFormattingError(f"Age restrictions are violated by a value other than fill={fill_value}.")


def _check_sex_restrictions(data: pd.DataFrame, male_only: bool, female_only: bool, fill_value: float):
    if male_only and (data.loc[data.sex == 'Female', 'value'] != fill_value).any():
        raise DataFormattingError(f"Restriction to male sex only is violated by a value other than fill={fill_value}.")
    elif female_only and (data.loc[data.sex == 'Male', 'value'] != fill_value).any():
        raise DataFormattingError(f"Restriction to female sex only is violated by a value other than fill={fill_value}.")
