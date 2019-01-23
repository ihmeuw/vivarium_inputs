import numpy as np
import pandas as pd
import pytest

from vivarium_inputs.validation import raw
from vivarium_inputs.globals import DataAbnormalError, DataNotExistError


@pytest.fixture
def gbd_mock(mocker):
    gbd_mock = mocker.patch('vivarium_inputs.validation.raw.gbd')
    gbd_mock.get_estimation_years.return_value = list(range(1990, 2015, 5)) + [2017]
    gbd_mock.get_age_group_id.return_value = list(range(1, 6))
    gbd_mock.MALE = 1
    gbd_mock.FEMALE = 2
    gbd_mock.COMBINED = 3
    return gbd_mock


@pytest.fixture
def measures_mock(mocker):
    return mocker.patch('vivarium_inputs.validation.raw.MEASURES', {'A': 1, 'B': 2})


@pytest.fixture
def metrics_mock(mocker):
    return mocker.patch('vivarium_inputs.validation.raw.METRICS', {'A': 1, 'B': 2})


test_data = [(range(1950, 2020), 'annual'), (list(range(1990, 2015, 5)) + [2017], 'binned')]


@pytest.mark.parametrize('years, bin_type', test_data, ids=['annual', 'binned'])
def test_check_years_pass(years, bin_type, gbd_mock):
    df = pd.DataFrame({'year_id': years})
    raw.check_years(df, bin_type)


test_data = [([1990, 1992], 'annual'), ([1990, 1995], 'binned')]


@pytest.mark.parametrize('years, bin_type', test_data, ids=['annual', 'binned'])
def test_check_years_missing(years, bin_type, gbd_mock):
    df = pd.DataFrame({'year_id': years})
    with pytest.raises(DataAbnormalError, match='missing years') as e:
        raw.check_years(df, bin_type)


test_data = [((range(1950, 2020), 'annual'), 'annual'), (list(range(1990, 2015, 5)) + [2017, 2020], 'binned')]


@pytest.mark.parametrize('years, bin_type', test_data, ids=['annual', 'binned'])
def test_check_years_extra(years, bin_type, gbd_mock):
    df = pd.DataFrame({'year_id': years})
    if bin_type == 'annual':
        # extra years should be allowed for annual
        raw.check_years(df, 'annual')
    else:
        with pytest.raises(DataAbnormalError, match='extra years'):
            raw.check_years(df, 'binned')


@pytest.mark.parametrize('location_id', list(range(2, 10)))
def test_check_location_pass(location_id):
    df = pd.DataFrame({'location_id': [location_id] * 5})
    raw.check_location(df, location_id)


@pytest.mark.parametrize('location_id', list(range(2, 10)))
def test_check_location_global_pass(location_id):
    df = pd.DataFrame({'location_id': [1] * 5})
    raw.check_location(df, location_id)


test_data = [([1, 2], 'multiple'), ([2, 2], 'actually has location id')]


@pytest.mark.parametrize('location_ids, match', test_data, ids=['multiple locations', 'wrong location'])
def test_check_location_fail(location_ids, match):
    df = pd.DataFrame({'location_id': location_ids})
    with pytest.raises(DataAbnormalError, match=match):
        raw.check_location(df, -1)


@pytest.mark.parametrize('columns', [['a', 'b'], ['c', 'd', 'e'], ['a'], []])
def test_check_columns_pass(columns):
    raw.check_columns(columns, columns)


@pytest.mark.parametrize('columns', [['a', 'b'], ['c', 'd', 'e'], ['a'], []])
def test_check_columns_extra_fail(columns):
    with pytest.raises(DataAbnormalError, match='extra columns'):
        raw.check_columns(columns, columns+['extra'])


@pytest.mark.parametrize('columns', [['a', 'b'], ['c', 'd', 'e'], ['a']])
def test_check_columns_missing_fail(columns):
    with pytest.raises(DataAbnormalError, match='missing columns'):
        raw.check_columns(columns, columns[:-1])


@pytest.mark.parametrize('data', [pd.DataFrame({'a': [1], 'b': [np.nan], 'c': [0]}),
                                  pd.DataFrame({'a': [np.nan], 'b': [np.nan]}),
                                  pd.DataFrame({'a': [np.inf], 'b': [1], 'c': [0]}),
                                  pd.DataFrame({'a': [np.inf], 'b': [np.inf]}),
                                  pd.DataFrame(columns=['a', 'b'])],
                         ids=['Single NaN', 'All NaN', 'Single Inf', 'All Inf', 'Empty'])
def test_check_data_exist_always_fail(data):
    with pytest.raises(DataNotExistError):
        raw.check_data_exist(data, value_columns=data.columns, zeros_missing=False)

    assert raw.check_data_exist(data, value_columns=data.columns, zeros_missing=False, error=False) is False

    with pytest.raises(DataNotExistError):
        raw.check_data_exist(data, value_columns=data.columns, zeros_missing=True)

    assert raw.check_data_exist(data, value_columns=data.columns, zeros_missing=True, error=False) is False


def test_check_data_exist_fails_only_on_missing_zeros():
    df = pd.DataFrame({'a': [0, 0], 'b': [0, 0]})

    raw.check_data_exist(df, value_columns=df.columns, zeros_missing=False)

    with pytest.raises(DataNotExistError):
        raw.check_data_exist(df, value_columns=df.columns, zeros_missing=True)


def test_check_data_exist_value_columns():
    df = pd.DataFrame({'a': [0, 0], 'b': [0, 0], 'c': [1, 1]})

    assert raw.check_data_exist(df, value_columns=['a', 'b', 'c'], zeros_missing=True, error=False)

    assert raw.check_data_exist(df, value_columns=['a', 'b'], zeros_missing=True, error=False) is False


@pytest.mark.parametrize('data', [pd.DataFrame({'a': [1], 'b': [1], 'c': [0]}),
                                  pd.DataFrame({'a': [0], 'b': [1]}),
                                  pd.DataFrame({'a': [0.0001], 'b': [0], 'c': [0]}),
                                  pd.DataFrame({'a': [1000], 'b': [-1000]})])
def test_check_data_exist_pass(data):
    assert raw.check_data_exist(data, value_columns=data.columns, error=False)


def test_check_age_group_ids(gbd_mock):
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

    df = pd.DataFrame({'age_group_id': range(2, 5)})
    raw.check_age_group_ids(df, 2, 4)


def test_check_sex_ids(gbd_mock):
    df = pd.DataFrame({'sex_id': [1, 2, 12]})
    with pytest.raises(DataAbnormalError, match='invalid sex ids'):
        raw.check_sex_ids(df)

    df = pd.DataFrame({'sex_id': [1, 2, 3]})
    with pytest.warns(Warning, match='extra sex ids'):
        raw.check_sex_ids(df, male_expected=True, female_expected=False, combined_expected=False)

    df = pd.DataFrame({'sex_id': [1, 2]})
    with pytest.warns(Warning, match='missing the following expected sex ids'):
        raw.check_sex_ids(df, male_expected=True, female_expected=True, combined_expected=True)


def test_check_age_restrictions(gbd_mock):
    df = pd.DataFrame({'age_group_id': [1, 2, 3], 'a': 1, 'b': 0})
    with pytest.raises(DataAbnormalError, match='missing the following'):
        raw.check_age_restrictions(df, 1, 4, value_columns=['a', 'b'])

    df = pd.DataFrame({'age_group_id': [1, 2, 3], 'a': [1, 1, 1], 'b': [2, 3, 4]})
    with pytest.raises(DataAbnormalError, match='also included values for age groups'):
        raw.check_age_restrictions(df, 1, 2, value_columns=['a', 'b'])

    df = pd.DataFrame({'age_group_id': [1, 2, 3], 'a': [1, 1, 0], 'b': [2, 3, 0]})
    raw.check_age_restrictions(df, 1, 2, value_columns=['a', 'b'])


def test_check_value_columns_boundary():
    df = pd.DataFrame({'a': [0, 1], 'b': [2, 20]})
    raw.check_value_columns_boundary(df, 0, 'lower', ['a', 'b'], inclusive=True, error=True)

    with pytest.raises(DataAbnormalError, match='values below'):
        raw.check_value_columns_boundary(df, 1, 'lower', ['a', 'b'], inclusive=True, error=True)

    with pytest.warns(Warning, match='values below'):
        raw.check_value_columns_boundary(df, 1, 'lower', ['a', 'b'], inclusive=True, error=False)

    with pytest.raises(DataAbnormalError, match='values below or equal to'):
        raw.check_value_columns_boundary(df, 0, 'lower', ['a', 'b'], inclusive=False, error=True)

    df = pd.DataFrame({'a': [0, 1], 'b': [2, 20]})
    raw.check_value_columns_boundary(df, 20, 'upper', ['a', 'b'], inclusive=True, error=True)

    with pytest.raises(DataAbnormalError, match='values above'):
        raw.check_value_columns_boundary(df, 20, 'upper', ['a', 'b'], inclusive=False, error=True)

    with pytest.warns(Warning, match='values above'):
        raw.check_value_columns_boundary(df, 20, 'upper', ['a', 'b'], inclusive=False, error=False)

    max_vals = pd.Series([5, 20], index=df.index)
    raw.check_value_columns_boundary(df, max_vals, 'upper', ['a', 'b'], inclusive=True, error=True)

    max_vals = pd.Series([5, 10], index=df.index)
    with pytest.raises(DataAbnormalError, match='above the expected boundary values'):
        raw.check_value_columns_boundary(df, max_vals, 'upper', ['a', 'b'], inclusive=True, error=True)


def test_check_sex_restrictions(gbd_mock):
    df = pd.DataFrame({'sex_id': [1, 1], 'a': 0, 'b': 0})
    with pytest.raises(DataAbnormalError, match='missing data values for males'):
        raw.check_sex_restrictions(df, True, False, value_columns=['a', 'b'])

    df = pd.DataFrame({'sex_id': [1, 2], 'a': 1, 'b': 0})
    with pytest.raises(DataAbnormalError, match='contains non-male sex ids for which data values are not all 0.'):
        raw.check_sex_restrictions(df, True, False, value_columns=['a', 'b'])

    df = pd.DataFrame({'sex_id': [2], 'a': 0, 'b': 0})
    with pytest.raises(DataAbnormalError, match='missing data values for females'):
        raw.check_sex_restrictions(df, False, True, value_columns=['a', 'b'])

    df = pd.DataFrame({'sex_id': [1, 2], 'a': 1, 'b': 0})
    with pytest.raises(DataAbnormalError, match='contains non-female sex ids for which data values are not all 0.'):
        raw.check_sex_restrictions(df, False, True, value_columns=['a', 'b'])

    df = pd.DataFrame({'sex_id': [3], 'a': [0], 'b': [0]})
    with pytest.raises(DataAbnormalError, match='not contain non-zero values for both'):
        raw.check_sex_restrictions(df, False, False, value_columns=['a', 'b'])

    df['a'] = 1
    raw.check_sex_restrictions(df, False, False, value_columns=['a', 'b'])

    df = pd.DataFrame({'sex_id': [1, 2], 'a': [0, 1], 'b': [0, 1]})
    with pytest.raises(DataAbnormalError, match='not contain non-zero values for both'):
        raw.check_sex_restrictions(df, False, False, value_columns=['a', 'b'])

    df = pd.DataFrame({'sex_id': [3], 'a': [1], 'b': [1]})
    raw.check_sex_restrictions(df, False, False, value_columns=['a', 'b'])

    df = pd.DataFrame({'sex_id': [1, 2], 'a': [0, 1], 'b': [1, 0]})
    raw.check_sex_restrictions(df, False, False, value_columns=['a', 'b'])


def test_check_measure_id(measures_mock):
    df = pd.DataFrame({'measure_id': [1, 1, 1, 12]})
    with pytest.raises(DataAbnormalError, match='multiple measure ids'):
        raw.check_measure_id(df, ['a'])

    df = pd.DataFrame({'measure_id': [3, 3, 3]})
    with pytest.raises(DataAbnormalError, match='not in the expected measure ids'):
        raw.check_measure_id(df, ['A', 'b'])

    df = pd.DataFrame({'measure_id': [1, 1]})
    raw.check_measure_id(df, ['A', 'b'])


def test_check_metric_id(metrics_mock):
    df = pd.DataFrame({'metric_id': [1, 1, 1, 12]})
    with pytest.raises(DataAbnormalError):
        raw.check_metric_id(df, 'a')

    df = pd.DataFrame({'metric_id': [2, 2]})
    with pytest.raises(DataAbnormalError):
        raw.check_metric_id(df, 'a')

    df = pd.DataFrame({'metric_id': [2, 2]})
    raw.check_metric_id(df, 'B')
