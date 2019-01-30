import pytest

import pandas as pd
import numpy as np

from vivarium_inputs.validation import sim
from vivarium_inputs.globals import DataFormattingError


@pytest.fixture
def mocked_get_estimation_years(mocker):
    gbd_mock = mocker.patch("vivarium_inputs.validation.sim.utilities.gbd.get_estimation_years")
    gbd_mock.return_value = range(1990, 2018)
    return gbd_mock


@pytest.fixture
def mocked_get_age_bins(mocker):
    gbd_mock = mocker.patch("vivarium_inputs.validation.sim.utilities.gbd.get_age_bins")
    df = pd.DataFrame({'age_group_id': [1, 2, 3, 4, 5],
                       'age_group_name': ['youngest', 'young', 'middle', 'older', 'oldest'],
                       'age_group_years_start': [0, 1, 15, 45, 60],
                       'age_group_years_end': [1, 15, 45, 60, 100]})
    gbd_mock.return_value = df
    return gbd_mock


def test__validate_draw_column_pass():
    df = pd.DataFrame({'draw': range(1000)})
    sim._validate_draw_column(df)


@pytest.mark.parametrize('draws', (900, 1100), ids=('too_few', 'too_many'))
def test__validate_draw_column_incorrect_number(draws):
    df = pd.DataFrame({'draw': range(draws)})
    with pytest.raises(DataFormattingError):
        sim._validate_draw_column(df)


def test__validate_draw_column_missing_column():
    df = pd.DataFrame({'draw_columns': range(1000)})
    with pytest.raises(DataFormattingError, match='in a column named'):
        sim._validate_draw_column(df)


@pytest.mark.parametrize("location", ("Kenya", "Papua New Guinea"))
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
    with pytest.raises(DataFormattingError, match='in a column named'):
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


def test__validate_sex_column_missing_column():
    df = pd.DataFrame({'sex_column': ['Male', 'Female']})
    with pytest.raises(DataFormattingError, match='in a column named'):
        sim._validate_sex_column(df)


def test__validate_age_columns_pass(mocked_get_age_bins):
    expected_ages = sim.utilities.get_age_bins()[['age_group_start',
                                              'age_group_end']].sort_values(['age_group_start', 'age_group_end'])
    sim._validate_age_columns(expected_ages)


def test__validate_age_columns_invalid_age(mocked_get_age_bins):
    df = sim.utilities.get_age_bins()[['age_group_start',
                                              'age_group_end']].sort_values(['age_group_start', 'age_group_end'])
    df.loc[2, 'age_group_start'] = -1
    with pytest.raises(DataFormattingError):
        sim._validate_age_columns(df)


def test__validate_age_columns_missing_group(mocked_get_age_bins):
    df = sim.utilities.get_age_bins()[['age_group_start',
                                              'age_group_end']].sort_values(['age_group_start', 'age_group_end'])
    df.drop(2, inplace=True)
    with pytest.raises(DataFormattingError):
        sim._validate_age_columns(df)


@pytest.mark.parametrize("columns", (
        ('age_group_start'),
        ('age_group_end'),
        ('age_group_id_start', 'age_group_end')
), ids=('missing_end', 'missing_start', 'typo'))
def test__validate_age_columns_missing_column(columns):
    df = pd.DataFrame()
    for col in columns:
        df[col] = [1, 2]
    with pytest.raises(DataFormattingError, match='in columns named'):
        sim._validate_age_columns(df)


def test__validate_year_columns_pass(mocked_get_estimation_years):
    expected_years = sim.utilities.get_annual_year_bins().sort_values(['year_start', 'year_end'])
    sim._validate_year_columns(expected_years)


def test__validate_year_columns_invalid_year(mocked_get_estimation_years):
    df = sim.utilities.get_annual_year_bins().sort_values(['year_start', 'year_end'])
    df.loc[2, 'year_end'] = -1
    with pytest.raises(DataFormattingError):
        sim._validate_year_columns(df)


def test__validate_year_columns_missing_group(mocked_get_estimation_years):
    df = sim.utilities.get_annual_year_bins().sort_values(['year_start', 'year_end'])
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
    with pytest.raises(DataFormattingError, match='in columns named'):
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
    with pytest.raises(DataFormattingError, match='in a column named'):
        sim._validate_value_column(df)


@pytest.mark.parametrize('values, start_ids, end_ids, type, fill', [
        ((1, 1, 1, 1, 1), [0], [5], 'outer', 0.0),
        ((0, 0, 1, 1, 1), [2], [5], 'outer', 0.0),
        ((0, 1, 1, 1, 0), [1], [4], 'outer', 0.0),
        ((1, 1, 1, 0, 0), [0], [3], 'outer', 0.0),
        ((0, 0, 0, 1, 1), [0], [3], 'outer', 1.0),
], ids=('no_restr', 'left_restr', 'outer_restr', 'right_restr', 'nonzero_fill'))
def test__check_age_restrictions(values, start_ids, end_ids, type, fill, mocked_get_age_bins):
    age_bins = mocked_get_age_bins()
    df = pd.DataFrame({'age_group_start': age_bins.age_group_years_start, 'age_group_end': age_bins.age_group_years_end})
    df['value'] = values
    sim._check_age_restrictions(df, start_ids, end_ids, type, fill)


@pytest.mark.parametrize('values, start_ids, end_ids, type, fill', [
        ((1, 1, 1, 1, 1), [1], [4], 'outer', 0.0),
        ((1, 1, 1, 1, 1), [2], [5], 'outer', 0.0),
        ((1, 1, 1, 1, 0), [0], [3], 'outer', 0.0),
        ((0, 0, 0, 0, 1), [0], [3], 'outer', 1.0),
        ((0, 1, 1, 1, 0), [0, 2], [3, 5], 'inner', 0.0)
], ids=('both_sides', 'left_side', 'right_side', 'nonzero_fill', 'inner'))
def test__check_age_restrictions_fail(values, start_ids, end_ids, type, fill, mocked_get_age_bins):
    age_bins = mocked_get_age_bins()
    df = pd.DataFrame({'age_group_start': age_bins.age_group_years_start, 'age_group_end': age_bins.age_group_years_end})
    df['value'] = values
    with pytest.raises(DataFormattingError):
        sim._check_age_restrictions(df, start_ids, end_ids, type, fill)


@pytest.mark.parametrize('values, restrictions, fill', [
    ((1, 1, 1, 1), (False, False), 0.0),
    ((1, 1, 0, 0), (True, False), 0.0),
    ((0, 0, 1, 1), (False, True), 0.0),
    ((1, 1, 2, 2), (False, True), 1.0)
], ids=('None', 'male', 'female', 'nonzero_fill'))
def test__check_sex_restrictions(values, restrictions, fill):
    df = pd.DataFrame({'sex': ['Male', 'Male', 'Female', 'Female'], 'value': values})
    sim._check_sex_restrictions(df, *restrictions, fill)


@pytest.mark.parametrize('values, restrictions, fill', [
    ((1, 1, 1, 0), (True, False), 0.0),
    ((0, 1, 1, 1), (False, True), 0.0),
    ((1, 2, 2, 2), (False, True), 1.0)
], ids=('male', 'female', 'nonzero_fill'))
def test__check_sex_restrictions_fail(values, restrictions, fill):
    df = pd.DataFrame({'sex': ['Male', 'Male', 'Female', 'Female'], 'value': values})
    with pytest.raises(DataFormattingError):
        sim._check_sex_restrictions(df, *restrictions, fill)

