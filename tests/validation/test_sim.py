import pytest

import pandas as pd
import numpy as np

from vivarium_inputs.validation import sim
from vivarium_inputs.globals import DataTransformationError


@pytest.fixture
def mock_validation_context():
    years = pd.DataFrame({'year_start': range(1990, 2017),
                          'year_end': range(1991, 2018)})
    age_bins = pd.DataFrame({'age_group_id': [1, 2, 3, 4, 5],
                             'age_group_name': ['youngest', 'young', 'middle', 'older', 'oldest'],
                             'age_group_start': [0, 1, 15, 45, 60],
                             'age_group_end': [1, 15, 45, 60, 100]})
    context = sim.SimulationValidationContext(
        location='United States',
        years=years,
        age_bins=age_bins
    )

    return context


def test__validate_draw_column_pass():
    df = pd.DataFrame({'draw': range(1000)})
    sim.validate_draw_column(df)


@pytest.mark.parametrize('draws', (900, 1100), ids=('too_few', 'too_many'))
def test_validate_draw_column_incorrect_number(draws):
    df = pd.DataFrame({'draw': range(draws)})
    with pytest.raises(DataTransformationError):
        sim.validate_draw_column(df)


@pytest.mark.parametrize("location", ("Kenya", "Papua New Guinea"))
def test__validate_location_column_pass(mock_validation_context, location):
    mock_validation_context['location'] = location
    df = pd.DataFrame({'location': [location]})
    sim.validate_location_column(df, mock_validation_context)


@ pytest.mark.parametrize('locations,expected_location', (
        (['Kenya', 'Kenya'], 'Egypt'),
        (['Algeria', 'Nigeria'], 'Algeria')
), ids=('mismatch', 'multiple'))
def test_validate_location_column_fail(mock_validation_context, locations, expected_location):
    mock_validation_context['location'] = expected_location
    df = pd.DataFrame({'location': locations})
    with pytest.raises(DataTransformationError):
        sim.validate_location_column(df, mock_validation_context)


def test_validate_sex_column_pass():
    df = pd.DataFrame({'sex': ['Male', 'Female']})
    sim.validate_sex_column(df)


@pytest.mark.parametrize("sexes", (
        ('Male', 'female'),
        ('Male', 'Male'),
        ('Female', 'Female')
), ids=('lowercase', 'missing_female', 'missing_male'))
def test_validate_sex_column_fail(sexes):
    df = pd.DataFrame({'sex': sexes})
    with pytest.raises(DataTransformationError):
        sim.validate_sex_column(df)


def test_validate_age_columns_pass(mock_validation_context):
    ages = mock_validation_context['age_bins'].filter(['age_group_start', 'age_group_end'])
    sim.validate_age_columns(ages, mock_validation_context)


def test_validate_age_columns_invalid_age(mock_validation_context):
    ages = mock_validation_context['age_bins'].filter(['age_group_start', 'age_group_end'])
    ages.loc[2, 'age_group_start'] = -1
    with pytest.raises(DataTransformationError):
        sim.validate_age_columns(ages, mock_validation_context)


def test_validate_age_columns_missing_group(mock_validation_context):
    ages = mock_validation_context['age_bins'].filter(['age_group_start', 'age_group_end'])
    ages.drop(2, inplace=True)
    with pytest.raises(DataTransformationError):
        sim.validate_age_columns(ages, mock_validation_context)


def test_validate_year_columns_pass(mock_validation_context):
    expected_years = mock_validation_context['years'].sort_values(['year_start', 'year_end'])
    sim.validate_year_columns(expected_years, mock_validation_context)


def test_validate_year_columns_invalid_year(mock_validation_context):
    df = mock_validation_context['years'].sort_values(['year_start', 'year_end'])
    df.loc[2, 'year_end'] = -1
    with pytest.raises(DataTransformationError):
        sim.validate_year_columns(df, mock_validation_context)


def test__validate_year_columns_missing_group(mock_validation_context):
    df = mock_validation_context['years'].sort_values(['year_start', 'year_end'])
    df.drop(0, inplace=True)
    with pytest.raises(DataTransformationError):
        sim.validate_year_columns(df, mock_validation_context)


@pytest.mark.parametrize("values", [(-1, 2, 3)], ids=['integers'])
def test_validate_value_column_pass(values):
    df = pd.DataFrame({'value': values})
    sim.validate_value_column(df)


@pytest.mark.parametrize("values", [(1, 2, np.inf),
                                    (1, np.nan, 2)],
                         ids=["infinity", "missing"])
def test_validate_value_column_fail(values):
    df = pd.DataFrame({'value': values})
    with pytest.raises(DataTransformationError):
        sim.validate_value_column(df)


@pytest.mark.parametrize('values, ids, restriction_type, fill', [
        ((1, 1, 1, 1, 1), (1, 5), 'outer', 0.0),
        ((0, 0, 1, 1, 1), (3, 5), 'outer', 0.0),
        ((0, 1, 1, 1, 0), (2, 4), 'outer', 0.0),
        ((1, 1, 1, 0, 0), (1, 3), 'outer', 0.0),
        ((2, 2, 2, 1, 1), (1, 3), 'outer', 1.0),
], ids=('no_restr', 'left_restr', 'outer_restr', 'right_restr', 'nonzero_fill'))
def test_check_age_restrictions(mocker, mock_validation_context, values, ids, restriction_type, fill):
    entity = mocker.patch('vivarium_inputs.validation.sim.utilities.get_age_group_ids_by_restriction')
    entity.return_value = ids
    age_bins = mock_validation_context['age_bins']
    df = age_bins.filter(['age_group_start', 'age_group_end'])
    df['value'] = values
    sim.check_age_restrictions(df, entity, restriction_type, fill, mock_validation_context)


@pytest.mark.parametrize('values, ids, restriction_type, fill', [
        ((1, 1, 1, 1, 1), (1, 4), 'outer', 0.0),
        ((0, 1, 1, 1, 1), (1, 3), 'outer', 0.0),
        ((1, 1, 1, 1, 0), (1, 3), 'outer', 0.0),
        ((2, 2, 2, 2, 1), (2, 5), 'outer', 1.0),
], ids=('both_sides', 'left_side', 'right_side', 'nonzero_fill'))
def test_check_age_restrictions_fail(mocker, mock_validation_context, values, ids, restriction_type, fill):
    entity = mocker.patch('vivarium_inputs.validation.sim.utilities.get_age_group_ids_by_restriction')
    entity.return_value = ids
    age_bins = mock_validation_context['age_bins']
    df = age_bins.filter(['age_group_start', 'age_group_end'])
    df['value'] = values
    with pytest.raises(DataTransformationError):
        sim.check_age_restrictions(df, entity, restriction_type, fill, mock_validation_context)


@pytest.mark.parametrize('values, restrictions, fill', [
    ((1, 1, 1, 1), (False, False), 0.0),
    ((1, 1, 0, 0), (True, False), 0.0),
    ((0, 0, 1, 1), (False, True), 0.0),
    ((1, 1, 2, 2), (False, True), 1.0)
], ids=('None', 'male', 'female', 'nonzero_fill'))
def test_check_sex_restrictions(values, restrictions, fill):
    df = pd.DataFrame({'sex': ['Male', 'Male', 'Female', 'Female'], 'value': values})
    sim.check_sex_restrictions(df, restrictions[0], restrictions[1], fill)


@pytest.mark.parametrize('values, restrictions, fill', [
    ((1, 1, 1, 0), (True, False), 0.0),
    ((0, 1, 1, 1), (False, True), 0.0),
    ((1, 2, 2, 2), (False, True), 1.0)
], ids=('male', 'female', 'nonzero_fill'))
def test_check_sex_restrictions_fail(values, restrictions, fill):
    df = pd.DataFrame({'sex': ['Male', 'Male', 'Female', 'Female'], 'value': values})
    with pytest.raises(DataTransformationError):
        sim.check_sex_restrictions(df, restrictions[0], restrictions[1], fill)


@pytest.mark.parametrize('df', [pd.DataFrame(columns=['a', 'b', 'c', 'd']),
                                pd.DataFrame({'a': 0, 'b': 0, 'c': 0, 'd': 0}, index=[17]),
                                pd.DataFrame({'a': 1, 'b': 2, 'c': 3, 'd': np.NaN}, index=[5, 10, 15])])
def test_validate_expected_columns_pass(df):
    cols = ['a', 'b', 'c', 'd']
    sim.validate_expected_columns(cols, df.columns)


@pytest.mark.parametrize('df', [pd.DataFrame(columns=['a', 'b', 'c', 'd']),
                                pd.DataFrame({'a': 0, 'b': 0, 'c': 0, 'd': 0}, index=[17]),
                                pd.DataFrame({'a': 1, 'b': 2, 'c': 3, 'd': np.NaN}, index=[5, 10, 15])])
def test_validate_expected_columns_fail(df):
    cols = ['a', 'b', 'c', 'd']
    with pytest.raises(DataTransformationError, match='missing'):
        sim.validate_expected_columns(cols + ['e'], df.columns)
    with pytest.raises(DataTransformationError, match='extra'):
        sim.validate_expected_columns(cols[1:], df.columns)
