import numpy as np
import pandas as pd
import pytest

from unittest import mock

from vivarium_inputs.validation import raw
from vivarium_inputs.globals import DataAbnormalError, DataNotExistError


@mock.patch('vivarium_inputs.validation.raw.gbd.get_estimation_years')
def test_check_years(mock_get_estimation_years):
    mock_get_estimation_years.return_value = list(range(1990, 2015, 5)) + [2017]

    df = pd.DataFrame({'year_id': [1990, 1991]})
    with pytest.raises(DataAbnormalError, match='missing years') as e:
        raw.check_years(df, 'annual')

    # extra years should be allowed for annual
    df = pd.DataFrame({'year_id': range(1950, 2020)})
    raw.check_years(df, 'annual')

    df = pd.DataFrame({'year_id': [1990, 1995]})
    with pytest.raises(DataAbnormalError, match='missing years'):
        raw.check_years(df, 'binned')

    df = pd.DataFrame({'year_id': list(range(1990, 2015, 5)) + [2017, 2020]})
    with pytest.raises(DataAbnormalError, match='extra years'):
        raw.check_years(df, 'binned')

    df = pd.DataFrame({'year_id': list(range(1990, 2015, 5)) + [2017]})
    raw.check_years(df, 'binned')


def test_check_location():
    df = pd.DataFrame({'location_id': [1, 2]})
    with pytest.raises(DataAbnormalError, match='multiple'):
        raw.check_location(df, 2)

    df = pd.DataFrame({'location_id': [3, 3, 3]})
    with pytest.raises(DataAbnormalError, match='actually has location id'):
        raw.check_location(df, 5)

    df = pd.DataFrame({'location_id': [1, 1, 1]})
    raw.check_location(df, 5)

    df = pd.DataFrame({'location_id': [3, 3, 3]})
    raw.check_location(df, 3)


def test_check_columns():
    expected_cols = ['a', 'b', 'c']

    with pytest.raises(DataAbnormalError, match='missing columns'):
        raw.check_columns(expected_cols, ['a', 'b'])

    with pytest.raises(DataAbnormalError, match='extra columns'):
        raw.check_columns(expected_cols, expected_cols+['d'])

    raw.check_columns(expected_cols, expected_cols)


def test_check_data_exist():
    df = pd.DataFrame()
    with pytest.raises(DataNotExistError):
        raw.check_data_exist(df)

    assert raw.check_data_exist(df, error=False) is False

    df = pd.DataFrame({'a': [np.nan, 5], 'b': [1, 1]})
    with pytest.raises(DataNotExistError):
        raw.check_data_exist(df, value_columns=['a', 'b'])
    assert raw.check_data_exist(df, value_columns=['b'])

    df['b'] = 0
    assert raw.check_data_exist(df, value_columns=['b'], zeros_missing=False)


@mock.patch('vivarium_inputs.validation.raw.gbd.get_age_group_id')
def test_check_age_group_ids(mock_get_age_group_id):
    mock_get_age_group_id.return_value = list(range(1, 6))

    df = pd.DataFrame({'age_group_id': [1, 2, 3, 100]})
    with pytest.raises(DataAbnormalError, match='invalid'):
        raw.check_age_group_ids(df)

    df = pd.DataFrame({'age_group_id': [1, 3, 4, 5]})
    with pytest.raises(DataAbnormalError, match='non-contiguous'):
        raw.check_age_group_ids(df)

    df = pd.DataFrame({'age_group_id': [2, 3]})
    with pytest.warns(Warning, match='not contain all age groups in restriction range'):
        raw.check_age_group_ids(df, 2, 4)

    df = pd.DataFrame({'age_group_id': [2, 3, 4]})
    with pytest.warns(Warning, match='additional age groups'):
        raw.check_age_group_ids(df, 2, 3)

    raw.check_age_group_ids(df, 2, 4)

    df = pd.DataFrame({'age_group_id': range(1, 6)})
    raw.check_age_group_ids(df, 2, 4)


@mock.patch('vivarium_inputs.validation.raw.gbd.MALE', 1)
@mock.patch('vivarium_inputs.validation.raw.gbd.FEMALE', 2)
@mock.patch('vivarium_inputs.validation.raw.gbd.COMBINED', 3)
def test_check_sex_ids():
    df = pd.DataFrame({'sex_id': [1, 2, 12]})
    with pytest.raises(DataAbnormalError, match='invalid sex ids'):
        raw.check_sex_ids(df)

    df = pd.DataFrame({'sex_id': [1, 2, 3]})
    with pytest.warns(Warning, match='extra sex ids'):
        raw.check_sex_ids(df, male_expected=True, female_expected=False, combined_expected=False)

    df = pd.DataFrame({'sex_id': [1, 2]})
    with pytest.warns(Warning, match='missing the following expected sex ids'):
        raw.check_sex_ids(df, male_expected=True, female_expected=True, combined_expected=True)


@mock.patch('vivarium_inputs.validation.raw.gbd.get_age_group_id')
def test_check_age_restrictions(mock_get_age_group_id):
    mock_get_age_group_id.return_value = list(range(1, 6))

    df = pd.DataFrame({'age_group_id': [1, 2, 3], 'a': 1, 'b': 0})
    with pytest.raises(DataAbnormalError, match='missing the following'):
        raw.check_age_restrictions(df, 1, 4, value_columns=['a', 'b'])

    df = pd.DataFrame({'age_group_id': [1, 2, 3], 'a': [1, 1, 1], 'b': [2, 3, 4]})
    with pytest.raises(DataAbnormalError, match='also included values for age groups'):
        raw.check_age_restrictions(df, 1, 2, value_columns=['a', 'b'])

    df = pd.DataFrame({'age_group_id': [1, 2, 3], 'a': [1, 1, 0], 'b': [2, 3, 0]})
    raw.check_age_restrictions(df, 1, 2, value_columns=['a', 'b'])


