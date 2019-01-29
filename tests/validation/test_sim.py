import pytest

import pandas as pd

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
    with pytest.raises(DataFormattingError):
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
    with pytest.raises(DataFormattingError):
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
    with pytest.raises(DataFormattingError):
        sim._validate_sex_column(df)


def test__validate_age_columns_pass():
    expected_ages = utilities.get_age_bins()[['age_group_start',
                                              'age_group_end']].sort_values(['age_group_start', 'age_group_end'])
    sim._validate_age_columns(expected_ages)


def test__validate_age_columns_invalid_age():
    expected_ages = utilities.get_age_bins()[['age_group_start',
                                              'age_group_end']].sort_values(['age_group_start', 'age_group_end'])
    expected_ages.loc[2, 'age_group_start'] = -1
    with pytest.raises(DataFormattingError):
        sim._validate_age_columns(expected_ages)


def test__validate_age_columns_missing_group():
    expected_ages = utilities.get_age_bins()[['age_group_start',
                                              'age_group_end']].sort_values(['age_group_start', 'age_group_end'])
    expected_ages.drop(2, inplace=True)
    with pytest.raises(DataFormattingError):
        sim._validate_age_columns(expected_ages)


@ pytest.mark.parametrize("columns", (
        ('age_group_start'),
        ('age_group_end'),
        ('age_group_id_start', 'age_group_end')
), ids=('missing_end', 'missing_start', 'typo'))
def test__validate_age_columns_missing_column(columns):
    df = pd.DataFrame()
    for col in columns:
        df[col] = [1, 2]
    with pytest.raises(DataFormattingError):
        sim._validate_age_columns(df)


# test_data = [{'year_start': }]
#
# @pytest.mark.parametrize("year_data", test_data, ids=[])
# def test__validate_year_columns_improper(year_data):
#     df = pd.DataFrame(year_data)
#     with pytest.raises(DataFormattingError):
#         sim._validate_year_columns(df)
#
#
# def test__validate_year_columns_missing():
#     year_start = range(1990, 2017)
#     year_end = range(1991, 2018)
#     df = pd.DataFrame({"year_start": year_start, "value": [0] * len(year_start)})
#     with pytest.raises(DataFormattingError):
#         sim._validate_year_columns(df)
#
#     df = pd.DataFrame({"year_end": year_end, "value": [0] * len(year_end)})
#     with pytest.raises(DataFormattingError):
#         sim._validate_year_columns(df)
#
#
# @pytest.mark.parametrize("values", [(1, 2, 3)],
#                          ids=['integers'])
# def test__validate_value_column_pass(values):
#     df = pd.DataFrame({'value': values})
#     sim._validate_value_column(df)
#
#
# @pytest.mark.parametrize("values", [(1, 2, np.inf),
#                                     (1, np.nan, 2)],
#                          ids=["infinity", "missing"])
# def test__validate_value_column_fail(values):
#     df = pd.DataFrame({'value': values})
#     with pytest.raises(DataFormattingError):
#         sim._validate_value_column(df)
#
#
# def test__validate_value_column_missing(values):
#     pass