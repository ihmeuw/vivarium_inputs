import pytest

import pandas as pd
import numpy as np

from vivarium_inputs.validation import sim
from vivarium_inputs import utilities
from vivarium_inputs.globals import DataFormattingError


def test__validate_draw_column_pass():
    df = pd.DataFrame({'draw': range(1000)})
    sim._validate_draw_column(df)


@pytest.mark.parametrize('draws', (900, 1100), ids=('too_few', 'too_many'))
def test__validate_draw_column_incorrect_number(draws):
    df = pd.DataFrame({'draw': range(draws)})
    with pytest.raises(DataFormattingError):
        sim._validate_draw_column(df)


def test_validate_draw_column_missing_column():
    df = pd.DataFrame({'draw_columns': range(1000)})
    with pytest.raises(DataFormattingError, match='Draw column'):
        sim._validate_draw_column(df)


@pytest.mark.parametrize("location", ("Kenya", "Papuea New Guinea"))
def test__validate_location_column_pass(location):
    df = pd.DataFrame({'location': [location]})
    sim._validate_location_column(df, location)


@ pytest.mark.parametrize('locations,expected_location', (
        (['Kenya', 'Kenya'], 'Egypt'),
        (['Algeria', 'Nigeria'], 'Algeria')
), ids=('mismatch', 'multiple'))
def test__validate_location_column_fail(locations, expected_location):
    df = pd.DataFrame({'location': locations})
    with pytest.raises(DataFormattingError):
        sim._validate_location_column(df, expected_location)


def test__validate_location_column_missing_column():
    df = pd.DataFrame({'location_column': ['Kenya']})
    with pytest.raises(DataFormattingError, match='Location column'):
        sim._validate_location_column(df, 'Kenya')


def test__validate_sex_column_pass():
    df = pd.DataFrame({'sex': ['Male', 'Female']})
    sim._validate_sex_column(df)


@pytest.mark.parametrize("sexes", (
        ('Male', 'female'),
        ('Male', 'Male'),
        ('Female', 'Female')
), ids=('lowercase', 'missing_female', 'missing_male'))
def test__validate_sex_column_fail(sexes):
    df = pd.DataFrame({'sex': sexes})
    with pytest.raises(DataFormattingError):
        sim._validate_sex_column(df)


def test_validate_sex_column_missing_column():
    df = pd.DataFrame({'sex_column': ['Male', 'Female']})
    with pytest.raises(DataFormattingError, match='Sex column'):
        sim._validate_sex_column(df)


def test__validate_age_columns_pass():
    expected_ages = utilities.get_age_bins()[['age_group_start',
                                              'age_group_end']].sort_values(['age_group_start', 'age_group_end'])
    sim._validate_age_columns(expected_ages)


def test__validate_age_columns_invalid_age():
    df = utilities.get_age_bins()[['age_group_start',
                                              'age_group_end']].sort_values(['age_group_start', 'age_group_end'])
    df.loc[2, 'age_group_start'] = -1
    with pytest.raises(DataFormattingError):
        sim._validate_age_columns(df)


def test__validate_age_columns_missing_group():
    df = utilities.get_age_bins()[['age_group_start',
                                              'age_group_end']].sort_values(['age_group_start', 'age_group_end'])
    df.drop(2, inplace=True)
    with pytest.raises(DataFormattingError):
        sim._validate_age_columns(df)


@ pytest.mark.parametrize("columns", (
        ('age_group_start'),
        ('age_group_end'),
        ('age_group_id_start', 'age_group_end')
), ids=('missing_end', 'missing_start', 'typo'))
def test__validate_age_columns_missing_column(columns):
    df = pd.DataFrame()
    for col in columns:
        df[col] = [1, 2]
    with pytest.raises(DataFormattingError, match='Age column'):
        sim._validate_age_columns(df)


def test__validate_year_columns_pass():
    expected_years = utilities.get_annual_year_bins().sort_values(['year_start', 'year_end'])
    sim._validate_year_columns(expected_years)


def test__validate_year_columns_invalid_year():
    df = utilities.get_annual_year_bins().sort_values(['year_start', 'year_end'])
    df.loc[2, 'year_end'] = -1
    with pytest.raises(DataFormattingError):
        sim._validate_year_columns(df)


def test__validate_year_columns_missing_group():
    df = utilities.get_annual_year_bins().sort_values(['year_start', 'year_end'])
    df.drop(0, inplace=True)
    with pytest.raises(DataFormattingError):
        sim._validate_year_columns(df)


@pytest.mark.parametrize("columns", (
    ('year_start'),
    ('year_end'),
    ('year_id_start', 'year_end')
), ids=("missing_end", "missing_start", "typo"))
def test__validate_year_columns_missing(columns):
    df = pd.DataFrame()
    for col in columns:
        df[col] = [1, 2, 3]
    with pytest.raises(DataFormattingError, match='Year column'):
        sim._validate_year_columns(df)


@pytest.mark.parametrize("values", [(-1, 2, 3)], ids=['integers'])
def test__validate_value_column_pass(values):
    df = pd.DataFrame({'value': values})
    sim._validate_value_column(df)


@pytest.mark.parametrize("values", [(1, 2, np.inf),
                                    (1, np.nan, 2)],
                         ids=["infinity", "missing"])
def test__validate_value_column_fail(values):
    df = pd.DataFrame({'value': values})
    with pytest.raises(DataFormattingError):
        sim._validate_value_column(df)


def test__validate_value_column_missing():
    df = pd.DataFrame({'value_column': [1, 2, 3]})
    with pytest.raises(DataFormattingError, match='Value column'):
        sim._validate_value_column(df)


@pytest.mark.parametrize('values, restrictions, fill', [
        ((1, 1, 1, 1, 1), (0, 5), 0.0),
        ((0, 0, 1, 1, 1), (2, 5), 0.0),
        ((0, 1, 1, 1, 0), (1, 4), 0.0),
        ((1, 1, 1, 0, 0), (0, 3), 0.0),
        ((0, 0, 0, 1, 1), (0, 3), 1.0)
], ids=('no_restr', 'left_restr', 'outer_restr', 'right_restr', 'nonzero_fill'))
def test__check_age_restrictions(values, restrictions, fill):
    df = pd.DataFrame({'age_group_start': range(5), 'age_group_end': range(1, 6)})
    df['value'] = values
    sim._check_age_restrictions(df, *restrictions, fill)


@pytest.mark.parametrize('values, restrictions, fill', [
        ((1, 1, 1, 1, 1), (1, 4), 0.0),
        ((0, 1, 1, 1, 1), (2, 5), 0.0),
        ((1, 1, 1, 1, 0), (0, 3), 0.0),
], ids=('both_sides', 'left_side', 'right_side'))
def test__check_age_restrictions_fail(values, restrictions, fill):
    df = pd.DataFrame({'age_group_start': range(5), 'age_group_end': range(1, 6)})
    df['value'] = values
    with pytest.raises(DataFormattingError):
        sim._check_age_restrictions(df, *restrictions, fill)


def test__check_sex_restrictions():
    pass


def test__check_sex_restrictions_fail():
    pass

