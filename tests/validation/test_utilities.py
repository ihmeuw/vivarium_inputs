import numpy as np
import pandas as pd
import pytest

from vivarium_inputs.validation import utilities
from vivarium_inputs.globals import DataAbnormalError, DataNotExistError, DataFormattingError


@pytest.fixture
def gbd_mock(mocker):
    gbd_mock = mocker.patch('vivarium_inputs.validation.utilities.gbd')
    gbd_mock.get_estimation_years.return_value = list(range(1990, 2015, 5)) + [2017]
    gbd_mock.get_age_group_id.return_value = list(range(1, 6))
    gbd_mock.MALE = [1]
    gbd_mock.FEMALE = [2]
    gbd_mock.COMBINED = [3]
    return gbd_mock


@pytest.fixture
def measures_mock(mocker):
    return mocker.patch('vivarium_inputs.validation.utilities.MEASURES', {'A': 1, 'B': 2})


@pytest.fixture
def metrics_mock(mocker):
    return mocker.patch('vivarium_inputs.validation.utilities.METRICS', {'A': 1, 'B': 2})


@pytest.mark.parametrize('years, bin_type', [(range(1950, 2020), 'annual'),
                                             (range(1990, 2018), 'annual'),
                                             (list(range(1990, 2015, 5)) + [2017], 'binned')],
                         ids=['annual with extra', 'annual', 'binned'])
def test_check_years_pass(years, bin_type, gbd_mock):
    df = pd.DataFrame({'year_id': years})
    utilities.check_years(df, bin_type)


@pytest.mark.parametrize('years, bin_type, match', [([1990, 1992], 'annual', 'missing'),
                                                    ([1990, 1995], 'binned', 'missing'),
                                                    (list(range(1990, 2015, 5)) + [2017, 2020], 'binned', 'extra')],
                         ids=['annual with gap', 'binned with gap', 'binned with extra'])
def test_check_years_fail(years, bin_type, match, gbd_mock):
    df = pd.DataFrame({'year_id': years})
    with pytest.raises(DataAbnormalError, match=match):
        utilities.check_years(df, bin_type)


@pytest.mark.parametrize('location_id', list(range(2, 10)))
def test_check_location_pass(location_id):
    df = pd.DataFrame({'location_id': [location_id] * 5})
    utilities.check_and_replace_location(df, location_id)


@pytest.mark.parametrize('location_id', list(range(2, 10)))
def test_check_location_global_pass(location_id):
    df = pd.DataFrame({'location_id': [1] * 5})
    utilities.check_and_replace_location(df, location_id)


@pytest.mark.parametrize('location_ids, match', [([1, 2], 'multiple'),
                                                 ([2, 2], 'actually has location id')],
                         ids=['multiple locations', 'wrong location'])
def test_check_location_fail(location_ids, match):
    df = pd.DataFrame({'location_id': location_ids})
    with pytest.raises(DataAbnormalError, match=match):
        utilities.check_and_replace_location(df, -1)


@pytest.mark.parametrize('columns', [['a', 'b'], ['c', 'd', 'e'], ['a'], []])
def test_check_columns_pass(columns):
    utilities.check_columns(columns, columns)


@pytest.mark.parametrize('columns', [['a', 'b'], ['c', 'd', 'e'], ['a'], []])
def test_check_columns_extra_fail(columns):
    with pytest.raises(DataAbnormalError, match='extra columns'):
        utilities.check_columns(columns, columns+['extra'])


@pytest.mark.parametrize('columns', [['a', 'b'], ['c', 'd', 'e'], ['a']])
def test_check_columns_missing_fail(columns):
    with pytest.raises(DataAbnormalError, match='missing columns'):
        utilities.check_columns(columns, columns[:-1])


@pytest.mark.parametrize('data', [pd.DataFrame({'a': [1], 'b': [np.nan], 'c': [0]}),
                                  pd.DataFrame({'a': [np.nan], 'b': [np.nan]}),
                                  pd.DataFrame({'a': [np.inf], 'b': [1], 'c': [0]}),
                                  pd.DataFrame({'a': [np.inf], 'b': [np.inf]}),
                                  pd.DataFrame(columns=['a', 'b'])],
                         ids=['Single NaN', 'All NaN', 'Single Inf', 'All Inf', 'Empty'])
def test_check_data_exist_always_fail(data):
    with pytest.raises(DataNotExistError):
        utilities.check_data_exist(data, value_columns=data.columns, zeros_missing=False)

    assert utilities.check_data_exist(data, value_columns=data.columns, zeros_missing=False, error=False) is False

    with pytest.raises(DataNotExistError):
        utilities.check_data_exist(data, value_columns=data.columns, zeros_missing=True)

    assert utilities.check_data_exist(data, value_columns=data.columns, zeros_missing=True, error=False) is False


def test_check_data_exist_fails_only_on_missing_zeros():
    df = pd.DataFrame({'a': [0, 0], 'b': [0, 0]})

    utilities.check_data_exist(df, value_columns=df.columns, zeros_missing=False)

    with pytest.raises(DataNotExistError):
        utilities.check_data_exist(df, value_columns=df.columns, zeros_missing=True)


def test_check_data_exist_value_columns():
    df = pd.DataFrame({'a': [0, 0], 'b': [0, 0], 'c': [1, 1]})

    assert utilities.check_data_exist(df, value_columns=['a', 'b', 'c'], zeros_missing=True, error=False)

    assert utilities.check_data_exist(df, value_columns=['a', 'b'], zeros_missing=True, error=False) is False


@pytest.mark.parametrize('data', [pd.DataFrame({'a': [1], 'b': [1], 'c': [0]}),
                                  pd.DataFrame({'a': [0], 'b': [1]}),
                                  pd.DataFrame({'a': [0.0001], 'b': [0], 'c': [0]}),
                                  pd.DataFrame({'a': [1000], 'b': [-1000]})])
def test_check_data_exist_pass(data):
    assert utilities.check_data_exist(data, zeros_missing=True, value_columns=data.columns, error=False)


@pytest.mark.parametrize('age_ids, start, end, match', [([1, 2, 3, 100], None, None, 'invalid'),
                                                        ([1, 3, 4, 5], None, None, 'non-contiguous'),
                                                        ([1, 2, 3, 100], 1, 3, 'invalid'),
                                                        ([1, 2, 3, 5], 1, 3, 'non-contiguous')])
def test_check_age_group_ids_fail(age_ids, start, end, match, gbd_mock):
    df = pd.DataFrame({'age_group_id': age_ids})
    with pytest.raises(DataAbnormalError, match=match):
        utilities.check_age_group_ids(df, start, end)


@pytest.mark.parametrize('age_ids, start, end, match', [([2, 3], 2, 4, 'contain all age groups in restriction range'),
                                                        ([2, 3, 4], 2, 3, 'additional age groups'),
                                                        ([1, 2, 3, 4, 5], 1, 3, 'additional age groups'),
                                                        ([1, 2, 3], 1, 5, 'contain all age groups in restriction range')])
def test_check_age_group_ids_warn(age_ids, start, end, match, gbd_mock):
    df = pd.DataFrame({'age_group_id': age_ids})
    with pytest.warns(Warning, match=match):
        utilities.check_age_group_ids(df, start, end)


@pytest.mark.parametrize('age_ids, start, end', [([2, 3, 4], 2, 4),
                                                 ([2, 3, 4], None, None),
                                                 ([1, 2, 3, 4, 5], 1, 5)])
def test_check_age_group_ids_pass(age_ids, start, end, gbd_mock, recwarn):
    df = pd.DataFrame({'age_group_id': age_ids})
    utilities.check_age_group_ids(df, start, end)

    assert len(recwarn) == 0, 'An unexpected warning was raised.'


@pytest.mark.parametrize('sex_ids, male, female, both', [([1, 2, 12], True, False, False),
                                                         ([1, 2, 3, 3, 0], True, True, True),
                                                         ([5], False, False, False)])
def test_check_sex_ids_fail(sex_ids, male, female, both, gbd_mock):
    df = pd.DataFrame({'sex_id': sex_ids})
    with pytest.raises(DataAbnormalError):
        utilities.check_sex_ids(df, male, female, both)


@pytest.mark.parametrize('sex_ids, male, female, both, match', [([1, 2, 1], True, True, True, ['missing']),
                                                                ([1, 2, 3, 2], True, True, False, ['extra']),
                                                                ([1], False, True, False, ['extra', 'missing'])])
def test_check_sex_ids_warn(sex_ids, male, female, both, match, gbd_mock):
    df = pd.DataFrame({'sex_id': sex_ids})
    with pytest.warns(None) as record:
        utilities.check_sex_ids(df, male, female, both)

    assert len(record) == len(match), 'The expected number of warnings were not raised.'
    for i, m in enumerate(match):
        assert m in str(record[i].message), f'The expected warning message was not raised for warning {i+1}.'


@pytest.mark.parametrize('sex_ids, male, female, both', [([1, 1, 1], True, False, False),
                                                         ([1, 2, 2, 2], True, True, False),
                                                         ([1, 2, 3, 3, 2, 1], True, True, True),
                                                         ([], False, False, False)])
def test_check_sex_ids_pass(sex_ids, male, female, both, gbd_mock, recwarn):
    df = pd.DataFrame({'sex_id': sex_ids})
    utilities.check_sex_ids(df, male, female, both)

    assert len(recwarn) == 0, 'An unexpected warning was raised.'


test_data = [(pd.DataFrame({'age_group_id': [1, 2, 3], 'a': 1, 'b': 0}), 1, 4, ['a', 'b'], 'missing'),
             (pd.DataFrame({'age_group_id': [1, 2], 'a': [0, 0], 'b': [0, 0]}), 1, 2, ['a', 'b'], 'missing for all age groups')]


@pytest.mark.parametrize('data, start, end, val_cols, match', test_data)
def test_check_age_restrictions_fail(data, start, end, val_cols, match, gbd_mock):
    with pytest.raises(DataAbnormalError, match=match):
        utilities.check_age_restrictions(data, start, end, value_columns=val_cols)


test_data = [(pd.DataFrame({'age_group_id': [1, 2, 3, 4, 5], 'a': 1, 'b': 0}), 1, 4, ['a', 'b']),
             (pd.DataFrame({'age_group_id': [1, 2, 3], 'a': [1, 1, 1], 'b': [2, 3, 0]}), 1, 2, ['a'])]


@pytest.mark.parametrize('data, start, end, val_cols', test_data)
def test_check_age_restrictions_warn(data, start, end, val_cols, gbd_mock):
    with pytest.warns(Warning, match='also included'):
        utilities.check_age_restrictions(data, start, end, value_columns=val_cols)


test_data = [(pd.DataFrame({'age_group_id': [1, 2, 3], 'a': 1, 'b': 0}), 1, 3, ['a', 'b']),
             (pd.DataFrame({'age_group_id': [1, 2], 'a': [1, 0], 'b': [1, 0.1]}), 1, 2, ['a', 'b']),
             (pd.DataFrame({'age_group_id': [1, 2, 3], 'a': [1, 1, 0], 'b': [2, 3, 0]}), 1, 3, ['a'])]


@pytest.mark.parametrize('data, start, end, val_cols', test_data)
def test_check_age_restrictions_pass(data, start, end, val_cols, gbd_mock):
    utilities.check_age_restrictions(data, start, end, value_columns=val_cols)


test_data = [(pd.DataFrame({'a': [0, 1], 'b': [2, 20]}), 1, 'lower', ['a', 'b'],
              True, 'values below', DataAbnormalError),
             (pd.DataFrame({'a': [0, 1], 'b': [2, 20]}), 0, 'lower', ['a', 'b'],
              False, 'values below or equal to', ValueError),
             (pd.DataFrame({'a': [0], 'b': [10], 'c': [100]}), 10, 'lower', ['a'],
              True, 'values below', ValueError),
             (pd.DataFrame({'a': [0, 1], 'b': [2, 20]}), 20, 'upper', ['a', 'b'],
              False, 'values above', DataAbnormalError),
             (pd.DataFrame({'a': [0, 1], 'b': [2, 20]}, index=[0, 1]), pd.Series([5, 10], index=[0, 1]), 'upper',
              ['a', 'b'], True, 'above the expected', DataFormattingError)]


@pytest.mark.parametrize('data, boundary, boundary_type, val_cols, inclusive, match, error', test_data)
def test_check_value_columns_boundary_fail_warn(data, boundary, boundary_type, val_cols, inclusive, match, error):
    with pytest.raises(error, match=match):
        utilities.check_value_columns_boundary(data, boundary, boundary_type,
                                               val_cols, inclusive=inclusive, error=error)

    with pytest.warns(Warning, match=match):
        utilities.check_value_columns_boundary(data, boundary, boundary_type, val_cols, inclusive=inclusive, error=None)


test_data = [(pd.DataFrame({'a': [0, 1], 'b': [2, 20]}), 0, 'lower', ['a', 'b'], True),
             (pd.DataFrame({'a': [0, 1], 'b': [2, 20]}), -1, 'lower', ['a', 'b'], False),
             (pd.DataFrame({'a': [0], 'b': [10], 'c': [100]}), 10, 'lower', ['b', 'c'], True),
             (pd.DataFrame({'a': [0, 1], 'b': [2, 20]}), 20, 'upper', ['a', 'b'], True),
             (pd.DataFrame({'a': [0, 1], 'b': [2, 20]}, index=[0, 1]), pd.Series([0, 1], index=[0, 1]), 'upper',
              ['a'], True)]


@pytest.mark.parametrize('data, boundary, boundary_type, val_cols, inclusive', test_data)
def test_check_value_columns_boundary_pass(data, boundary, boundary_type, val_cols, inclusive):
    utilities.check_value_columns_boundary(data, boundary, boundary_type, val_cols,
                                           inclusive=inclusive, error=ValueError)


test_data = [(pd.DataFrame({'sex_id': [1, 1], 'a': 0, 'b': 0, 'c': 1}), True, False,
              ['a', 'b'], 'missing data values for males'),
             (pd.DataFrame({'sex_id': [2, 2, 2, 3], 'a': 0, 'b': 0}), False, True,
              ['a', 'b'], 'missing data values for females'),
             (pd.DataFrame({'sex_id': [3], 'a': [0], 'b': [0]}), False, False,
             ['a', 'b'], 'values for both'),
             (pd.DataFrame({'sex_id': [1, 2], 'a': [0, 1], 'b': [0, 1]}), False, False,
              ['a', 'b'], 'values for both')]

@pytest.mark.parametrize('data, male, female, val_cols, match', test_data)
def test_check_sex_restrictions_fail(data, male, female, val_cols, match, gbd_mock):
    with pytest.raises(DataAbnormalError, match=match):
        utilities.check_sex_restrictions(data, male, female, value_columns=val_cols)


test_data = [(pd.DataFrame({'sex_id': [1, 2], 'a': 1, 'b': 0}), True, False,
              ['a', 'b'], 'non-male sex ids'),
             (pd.DataFrame({'sex_id': [1, 2], 'a': 1, 'b': 0}), False, True,
              ['a', 'b'], 'non-female sex ids')]

@pytest.mark.parametrize('data, male, female, val_cols, match', test_data)
def test_check_sex_restrictions_warn(data, male, female, val_cols, match, gbd_mock):
    with pytest.warns(Warning, match=match):
        utilities.check_sex_restrictions(data, male, female, value_columns=val_cols)


test_data = [(pd.DataFrame({'sex_id': [1, 1], 'a': 1, 'b': 0, 'c': 0}), True, False, ['a', 'b']),
             (pd.DataFrame({'sex_id': [2, 2], 'a': 1, 'b': 0}), False, True, ['a', 'b']),
             (pd.DataFrame({'sex_id': [3, 3, 3], 'a': 1, 'b': 0}), False, False, ['a']),
             (pd.DataFrame({'sex_id': [1, 2], 'a': [1, 0], 'b': [0, 1]}), False, False, ['a', 'b']),
             (pd.DataFrame({'sex_id': [3], 'a': [0], 'b': [1]}), False, False, ['a', 'b'])]

@pytest.mark.parametrize('data, male, female, val_cols', test_data)
def test_check_sex_restrictions_pass(data, male, female, val_cols, gbd_mock):
    utilities.check_sex_restrictions(data, male, female, value_columns=val_cols)


@pytest.mark.parametrize('m_ids, expected, match', [([1, 1, 1, 12], ['A'], 'multiple'),
                                                    ([3, 3, 3], ['A', 'B'], 'not in the expected')])
def test_check_measure_id_fail(m_ids, expected, match, measures_mock):
    df = pd.DataFrame({'measure_id': m_ids})
    with pytest.raises(DataAbnormalError, match=match):
        utilities.check_measure_id(df, expected)


@pytest.mark.parametrize('m_ids, expected', [([1], ['A']),
                                             ([2, 2, 2, 2], ['A', 'B'])])
def test_check_measure_id_pass(m_ids, expected, measures_mock):
    df = pd.DataFrame({'measure_id': m_ids})
    utilities.check_measure_id(df, expected)



@pytest.mark.parametrize('m_ids, expected', [([1, 1, 1, 12], 'A'),
                                             ([3, 3, 3], 'A')])
def test_check_metric_id_fail(m_ids, expected, metrics_mock):
    df = pd.DataFrame({'metric_id': m_ids})
    with pytest.raises(DataAbnormalError):
        utilities.check_metric_id(df, expected)


@pytest.mark.parametrize('m_ids, expected', [([1], 'A'),
                                             ([2, 2, 2, 2], 'B')])
def test_check_metric_id_pass(m_ids, expected, metrics_mock):
    df = pd.DataFrame({'metric_id': m_ids})
    utilities.check_metric_id(df, expected)
