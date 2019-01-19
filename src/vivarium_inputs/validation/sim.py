"""Validates data is in the correct shape for the simulation."""
import numpy as np
import pandas as pd

from vivarium_inputs.globals import DataFormattingError, gbd


def validate_for_simulation(data, entity, measure, location):
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
        'demographic_dimensions': _validate_demographic_dimensions,
    }

    if measure not in validators:
        raise NotImplementedError()

    validators[measure](data, entity, location)


def _validate_incidence(data, entity, location):
    _validate_standard_columns(data, location)


def _validate_prevalence(data, entity, location):
    _validate_standard_columns(data, location)


def _validate_birth_prevalence(data, entity, location):
    _validate_draw_column(data)
    _validate_location_column(data, location)
    _validate_sex_column(data)
    _validate_year_columns(data)


def _validate_disability_weight(data, entity, location):
    _validate_standard_columns(data, location)


def _validate_remission(data, entity, location):
    _validate_standard_columns(data, location)


def _validate_cause_specific_mortality(data, entity, location):
    _validate_standard_columns(data, location)


def _validate_excess_mortality(data, entity, location):
    _validate_standard_columns(data, location)


def _validate_case_fatality(data, entity, location):
    _validate_standard_columns(data, location)
    raise NotImplementedError()


def _validate_exposure(data, entity, location):
    _validate_standard_columns(data, location)


def _validate_exposure_standard_deviation(data, entity, location):
    _validate_standard_columns(data, location)


def _validate_exposure_distribution_weights(data, entity, location):
    _validate_location_column(data, location)
    _validate_sex_column(data)
    _validate_age_columns(data)
    _validate_year_columns(data)
    _validate_value_column(data)


def _validate_relative_risk(data, entity, location):
    _validate_standard_columns(data, location)


def _validate_population_attributable_fraction(data, entity, location):
    _validate_standard_columns(data, location)


def _validate_mediation_factors(data, entity, location):
    _validate_standard_columns(data, location)
    raise NotImplementedError()


def _validate_estimate(data, entity, location):
    _validate_location_column(data, location)
    _validate_year_columns(data)
    if entity.by_age:
        _validate_age_columns(data)
    if entity.by_sex:
        _validate_sex_column(data)


def _validate_cost(data, entity, location):
    _validate_standard_columns(data, location)
    raise NotImplementedError()


def _validate_utilization(data, entity, location):
    _validate_standard_columns(data, location)
    raise NotImplementedError()


def _validate_structure(data, entity, location):
    _validate_location_column(data, location)
    _validate_sex_column(data)
    _validate_age_columns(data)
    _validate_year_columns(data)


def _validate_theoretical_minimum_risk_life_expectancy(data, entity, location):
    pass


def _validate_demographic_dimensions(data, entity, location):
    _validate_location_column(data, location)
    _validate_sex_column(data)
    _validate_age_columns(data)
    _validate_year_columns(data)


def _validate_standard_columns(data, location):
    _validate_draw_column(data)
    _validate_location_column(data, location)
    _validate_sex_column(data)
    _validate_age_columns(data)
    _validate_year_columns(data)
    _validate_value_column(data)


def _validate_draw_column(data):
    if 'draw' not in data.columns:
        raise DataFormattingError('Draw column name improperly specified.')

    if list(data['draw'].unique()) != list(range(1000)):
        raise DataFormattingError('Draw values improperly specified.')


def _validate_location_column(data, location):
    if 'location' not in data.columns:
        raise DataFormattingError('Location column name improperly specified.')

    if len(data['location'].unique()) != 1 or data['location'].unique()[0] != location:
        raise DataFormattingError('Location value improperly specified.')


def _validate_sex_column(data):
    if 'sex' not in data.columns:
        raise DataFormattingError('Sex column name improperly specified.')

    if set(data['sex'].unique()) != {'Male', 'Female'}:
        raise DataFormattingError('Sex value improperly specified.')


def _validate_age_columns(data):
    if 'age_group_start' not in data.columns or 'age_group_end' not in data.columns:
        raise DataFormattingError('Age column names improperly specified.')

    expected_age_block = (
        gbd.get_age_bins()[['age_group_years_start', 'age_group_years_end']]
            .rename(columns={'age_group_years_start': 'age_group_start',
                             'age_group_years_end': 'age_group_end'})
    )

    age_block = (data[['age_group_start', 'age_group_end']]
                 .drop_duplicates()
                 .sort_values(['age_group_start', 'age_group_end'])
                 .reset_index(drop=True))

    if not age_block.equals(expected_age_block):
        raise DataFormattingError('Age values improperly specified.')


def _validate_year_columns(data):
    if 'year_start' not in data.columns or 'year_end' not in data.columns:
        raise DataFormattingError('Year column names improperly specified.')

    expected_year_block = pd.DataFrame({'year_start': range(1990, 2018),
                                        'year_end': range(1991, 2019)})
    year_block = (data[['year_start', 'year_end']]
                  .drop_duplicates()
                  .sort_values(['year_start', 'year_end'])
                  .reset_index(drop=True))

    if not year_block.equals(expected_year_block):
        pass
        #raise DataFormattingError('Year values improperly specified.')


def _validate_value_column(data):
    if np.any(data.value.isna()):
        raise DataFormattingError('Nans found in data.')
    if np.any(np.isinf(data.value.values)):
        raise DataFormattingError('Inf found in data')

