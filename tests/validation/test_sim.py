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
                             'age_start': [0, 1, 15, 45, 60],
                             'age_end': [1, 15, 45, 60, 100]})
    context = sim.SimulationValidationContext(
        location='United States',
        years=years,
        age_bins=age_bins
    )

    return context


@pytest.mark.parametrize("location", ("Kenya", "Papua New Guinea"))
def test__validate_location_column_pass(mock_validation_context, location):
    mock_validation_context['location'] = location
    df = pd.DataFrame({'location': [location], 'other': [0]}).set_index(['location', 'other'])
    sim.validate_location_column(df, mock_validation_context)


@ pytest.mark.parametrize('locations,expected_location', (
        (['Kenya', 'Kenya'], 'Egypt'),
        (['Algeria', 'Nigeria'], 'Algeria')
), ids=('mismatch', 'multiple'))
def test_validate_location_column_fail(mock_validation_context, locations, expected_location):
    mock_validation_context['location'] = expected_location
    df = pd.DataFrame({'location': locations, 'other': [0]*len(locations)}).set_index(['location', 'other'])
    with pytest.raises(DataTransformationError):
        sim.validate_location_column(df, mock_validation_context)


def test_validate_sex_column_pass():
    df = pd.DataFrame({'sex': ['Male', 'Female'], 'other': [0]*2}).set_index(['sex', 'other'])
    sim.validate_sex_column(df)


@pytest.mark.parametrize("sexes", (
        ('Male', 'female'),
        ('Male', 'Male'),
        ('Female', 'Female')
), ids=('lowercase', 'missing_female', 'missing_male'))
def test_validate_sex_column_fail(sexes):
    df = pd.DataFrame({'sex': sexes, 'other': [0]*len(sexes)}).set_index(['sex', 'other'])
    with pytest.raises(DataTransformationError):
        sim.validate_sex_column(df)


def test_validate_age_columns_pass(mock_validation_context):
    ages = (mock_validation_context['age_bins']
            .filter(['age_start', 'age_end']))
    # Shuffle the rows and set index
    ages = ages.sample(frac=1).reset_index(drop=True)
    ages = ages.set_index(pd.IntervalIndex.from_arrays(ages.age_start, ages.age_end,
                                                       closed='left', name='age'),
                          append=True)
    sim.validate_age_column(ages, mock_validation_context)


def test_validate_age_columns_invalid_age(mock_validation_context):
    ages = (mock_validation_context['age_bins']
            .filter(['age_start', 'age_end']).copy())
    ages.loc[2, 'age_start'] = -1
    ages = ages.set_index(pd.IntervalIndex.from_arrays(ages.age_start, ages.age_end,
                                                       closed='left', name='age'),
                          append=True)
    with pytest.raises(DataTransformationError):
        sim.validate_age_column(ages, mock_validation_context)


def test_validate_age_columns_missing_group(mock_validation_context):
    ages = (mock_validation_context['age_bins']
            .filter(['age_start', 'age_end']))
    ages = ages.drop(2)
    ages = ages.set_index(pd.IntervalIndex.from_arrays(ages.age_start, ages.age_end,
                                                       closed='left', name='age'),
                          append=True)
    with pytest.raises(DataTransformationError):
        sim.validate_age_column(ages, mock_validation_context)


def test_validate_year_columns_pass(mock_validation_context):
    years = mock_validation_context['years']
    # Shuffle the rows and set index
    years = years.sample(frac=1).reset_index(drop=True)
    years = years.set_index(pd.IntervalIndex.from_arrays(years.year_start, years.year_end,
                                                         closed='left', name='year'),
                            append=True)
    sim.validate_year_column(years, mock_validation_context)


def test_validate_year_columns_invalid_year(mock_validation_context):
    years = mock_validation_context['years'].copy()
    years.loc[2, 'year_end'] = 20100
    # Shuffle the rows and set index
    years = years.sample(frac=1).reset_index(drop=True)
    years = years.set_index(pd.IntervalIndex.from_arrays(years.year_start, years.year_end,
                                                         closed='left', name='year'),
                            append=True)
    with pytest.raises(DataTransformationError):
        sim.validate_year_column(years, mock_validation_context)


def test_validate_year_columns_missing_group(mock_validation_context):
    years = mock_validation_context['years'].sort_values(['year_start', 'year_end'])
    years = years.drop(0)
    years = years.set_index(pd.IntervalIndex.from_arrays(years.year_start, years.year_end,
                                                         closed='left', name='year'),
                            append=True)
    with pytest.raises(DataTransformationError):
        sim.validate_year_column(years, mock_validation_context)


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
    idx = pd.IntervalIndex.from_arrays(age_bins.age_start, age_bins.age_end, closed='left', name='age')
    df = pd.DataFrame({'value': values}, index=idx)
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
    idx = pd.IntervalIndex.from_arrays(age_bins.age_start, age_bins.age_end, closed='left', name='age')
    df = pd.DataFrame({'value': values}, index=idx)
    with pytest.raises(DataTransformationError):
        sim.check_age_restrictions(df, entity, restriction_type, fill, mock_validation_context)


@pytest.mark.parametrize('values, restrictions, fill', [
    ((1, 1, 1, 1), (False, False), 0.0),
    ((1, 1, 0, 0), (True, False), 0.0),
    ((0, 0, 1, 1), (False, True), 0.0),
    ((1, 1, 2, 2), (False, True), 1.0)
], ids=('None', 'male', 'female', 'nonzero_fill'))
def test_check_sex_restrictions(values, restrictions, fill):
    df = pd.DataFrame({'sex': ['Male', 'Male', 'Female', 'Female'], 'other': [0]*4, 'value': values})
    df = df.set_index(['sex', 'other'])
    sim.check_sex_restrictions(df, restrictions[0], restrictions[1], fill)


@pytest.mark.parametrize('values, restrictions, fill', [
    ((1, 1, 1, 0), (True, False), 0.0),
    ((0, 1, 1, 1), (False, True), 0.0),
    ((1, 2, 2, 2), (False, True), 1.0)
], ids=('male', 'female', 'nonzero_fill'))
def test_check_sex_restrictions_fail(values, restrictions, fill):
    df = pd.DataFrame({'sex': ['Male', 'Male', 'Female', 'Female'], 'other': [0]*4, 'value': values})
    df = df.set_index(['sex', 'other'])
    with pytest.raises(DataTransformationError):
        sim.check_sex_restrictions(df, restrictions[0], restrictions[1], fill)


@pytest.mark.parametrize('df', [pd.DataFrame(columns=['a', 'b', 'c', 'd']),
                                pd.DataFrame({'a': [0], 'b': [0], 'c': [0], 'd': [0]}),
                                pd.DataFrame({'a': [1]*3, 'b': [2]*3, 'c': [3]*3, 'd': [np.NaN]*3})])
def test_validate_expected_index_columns_pass(df):
    cols = ['a', 'b', 'c', 'd']
    sim.validate_expected_index_and_columns([None], df.index.names, cols, df.columns)
    df = df.set_index(cols[:2])
    sim.validate_expected_index_and_columns(cols[:2], df.index.names, cols[2:], df.columns)
    df = df.set_index(cols[2:], append=True)
    sim.validate_expected_index_and_columns(cols, df.index.names, [], df.columns)


@pytest.mark.parametrize('df', [pd.DataFrame(columns=['a', 'b', 'c', 'd']),
                                pd.DataFrame({'a': [0], 'b': [0], 'c': [0], 'd': [0]}),
                                pd.DataFrame({'a': [1]*3, 'b': [2]*3, 'c': [3]*3, 'd': [np.NaN]*3})])
def test_validate_expected_columns_fail(df):
    cols = ['a', 'b', 'c', 'd']
    with pytest.raises(DataTransformationError, match='missing columns'):
        sim.validate_expected_index_and_columns([None], df.index.names, cols + ['e'], df.columns)
    with pytest.raises(DataTransformationError, match='missing index names'):
        data = df.set_index(cols[:2])
        sim.validate_expected_index_and_columns(cols[:3], data.index.names, cols[:2], data.columns)

    with pytest.raises(DataTransformationError, match='extra columns'):
        sim.validate_expected_index_and_columns([None], df.index.names, cols[1:], df.columns)
    with pytest.raises(DataTransformationError, match='extra index names'):
        data = df.set_index(cols[:2])
        sim.validate_expected_index_and_columns(cols[:1], data.index.names, cols[:2], data.columns)