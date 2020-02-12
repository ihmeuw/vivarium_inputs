import numpy as np
import pandas as pd
import pytest

from vivarium_inputs.validation import raw
from vivarium_inputs.globals import DataAbnormalError, DataDoesNotExistError, SEXES

LOCATION_ID = 100
ESTIMATION_YEARS = tuple(list(range(1990, 2015, 5)) + [2017])
AGE_GROUP_IDS = tuple(range(1, 6))
PARENT_LOCATIONS = (1, 5, 10)

@pytest.fixture
def mock_validation_context():
    context = raw.RawValidationContext(
        location_id=LOCATION_ID,
        estimation_years=list(ESTIMATION_YEARS),
        age_group_ids=list(AGE_GROUP_IDS),
        sexes=SEXES,
        parent_locations=list(PARENT_LOCATIONS),
    )
    return context


@pytest.fixture
def measures_mock(mocker):
    return mocker.patch('vivarium_inputs.validation.raw.MEASURES', {'A': 1, 'B': 2})


@pytest.fixture
def metrics_mock(mocker):
    return mocker.patch('vivarium_inputs.validation.raw.METRICS', {'A': 1, 'B': 2})


@pytest.mark.parametrize('mort, morb, yld_only, yll_only, match',
                         [([0, 1, 1], [0, 1, 1], True, False, 'set to 0'),
                          ([1, 1, 0], [1, 1, 1], True, False, 'only one'),
                          ([1, 1, 1], [0, 0, 0], True, False, 'restricted to yld_only'),
                          ([0, 0, 0], [1, 1, 1], False, True, 'restricted to yll_only'),
                          ([1, 0, 0], [0, 1, 1], True, False, 'is restricted to yld_only'),
                          ([1, 0, 2], [0, 1, 1], False, False, 'outside the expected')])
def test_check_mort_morb_flags_fail(mort, morb, yld_only, yll_only, match):
    data = pd.DataFrame({'mortality': mort, 'morbidity': morb})
    with pytest.raises(DataAbnormalError, match=match):
        raw.check_mort_morb_flags(data, yld_only, yll_only)


@pytest.mark.parametrize('mort, morb, yld_only, yll_only',
                         [([1, 1, 1], [1, 1, 1], True, False),
                          ([1, 1, 1], [1, 1, 1], False, False),
                          ([0, 0, 0], [1, 1, 1], True, False,),
                          ([1, 1, 1], [0, 0, 0], False, True,),
                          ([1, 0, 1], [0, 1, 0], False, False)])
def test_check_mort_morb_flags_pass(mort, morb, yld_only, yll_only):
    data = pd.DataFrame({'mortality': mort, 'morbidity': morb})
    raw.check_mort_morb_flags(data, yld_only, yll_only)


@pytest.mark.parametrize('years, bin_type', [(range(1950, 2020), 'annual'),
                                             (range(1990, 2018), 'annual'),
                                             (list(range(1990, 2015, 5)) + [2017], 'binned')],
                         ids=['annual with extra', 'annual', 'binned'])
def test_check_years_pass(mock_validation_context, years, bin_type):
    df = pd.DataFrame({'year_id': years})
    raw.check_years(df, mock_validation_context, bin_type)


@pytest.mark.parametrize('years, bin_type, match', [([1990, 1992], 'annual', 'missing'),
                                                    ([1990, 1995], 'binned', 'missing'),
                                                    (list(range(1990, 2015, 5)) + [2017, 2020], 'binned', 'extra')],
                         ids=['annual with gap', 'binned with gap', 'binned with extra'])
def test_check_years_fail(mock_validation_context, years, bin_type, match):
    df = pd.DataFrame({'year_id': years})
    with pytest.raises(DataAbnormalError, match=match):
        raw.check_years(df, mock_validation_context, bin_type)


@pytest.mark.parametrize('location_id', list(range(2, 10)))
def test_check_location_pass(mock_validation_context, location_id):
    mock_validation_context['location_id'] = location_id
    df = pd.DataFrame({'location_id': [location_id] * 5})
    raw.check_location(df, mock_validation_context)


@pytest.mark.parametrize('location_id', PARENT_LOCATIONS)
def test_check_location_parent_pass(mock_validation_context, location_id):
    df = pd.DataFrame({'location_id': [location_id]})
    raw.check_location(df, mock_validation_context)


@pytest.mark.parametrize('location_ids, match', [([1, 2], 'multiple'),
                                                 ([2, 2], 'actually has location id')],
                         ids=['multiple locations', 'wrong location'])
def test_check_location_fail(mock_validation_context, location_ids, match):
    df = pd.DataFrame({'location_id': location_ids})
    with pytest.raises(DataAbnormalError, match=match):
        raw.check_location(df, mock_validation_context)


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
    with pytest.raises(DataDoesNotExistError):
        raw.check_data_exist(data, value_columns=data.columns, zeros_missing=False)

    assert raw.check_data_exist(data, value_columns=data.columns, zeros_missing=False, error=False) is False

    with pytest.raises(DataDoesNotExistError):
        raw.check_data_exist(data, value_columns=data.columns, zeros_missing=True)

    assert raw.check_data_exist(data, value_columns=data.columns, zeros_missing=True, error=False) is False


def test_check_data_exist_fails_only_on_missing_zeros():
    df = pd.DataFrame({'a': [0, 0], 'b': [0, 0]})

    raw.check_data_exist(df, value_columns=df.columns, zeros_missing=False)

    with pytest.raises(DataDoesNotExistError):
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
    assert raw.check_data_exist(data, zeros_missing=True, value_columns=data.columns, error=False)


@pytest.mark.parametrize('test_age_ids, start, end, match', [([1, 2, 3, 100], None, None, 'invalid'),
                                                             ([1, 3, 4, 5], None, None, 'non-contiguous'),
                                                             ([1, 2, 3, 100], 1, 3, 'invalid'),
                                                             ([1, 2, 3, 5], 1, 3, 'non-contiguous')])
def test_check_age_group_ids_fail(mock_validation_context, test_age_ids, start, end, match):
    df = pd.DataFrame({'age_group_id': test_age_ids})
    with pytest.raises(DataAbnormalError, match=match):
        raw.check_age_group_ids(df, mock_validation_context, start, end)


@pytest.mark.parametrize('test_age_ids, start, end, match',
                         [([2, 3], 2, 4, 'contain all age groups in restriction range'),
                          ([2, 3, 4], 2, 3, 'additional age groups'),
                          ([1, 2, 3, 4, 5], 1, 3, 'additional age groups'),
                          ([1, 2, 3], 1, 5, 'contain all age groups in restriction range')])
def test_check_age_group_ids_warn(mock_validation_context, caplog, test_age_ids, start, end, match):
    df = pd.DataFrame({'age_group_id': test_age_ids})
    raw.check_age_group_ids(df, mock_validation_context, start, end)
    assert match in caplog.text


@pytest.mark.parametrize('test_age_ids, start, end',
                         [([2, 3, 4], 2, 4),
                          ([2, 3, 4], None, None),
                          ([1, 2, 3, 4, 5], 1, 5)])
def test_check_age_group_ids_pass(mock_validation_context, test_age_ids, start, end, recwarn):
    df = pd.DataFrame({'age_group_id': test_age_ids})
    raw.check_age_group_ids(df, mock_validation_context, start, end)

    assert len(recwarn) == 0, 'An unexpected warning was raised.'


@pytest.mark.parametrize('sex_ids, male, female, both', [([1, 2, 12], True, False, False),
                                                         ([1, 2, 3, 3, 0], True, True, True),
                                                         ([5], False, False, False)])
def test_check_sex_ids_fail(mock_validation_context, sex_ids, male, female, both):
    df = pd.DataFrame({'sex_id': sex_ids})
    with pytest.raises(DataAbnormalError):
        raw.check_sex_ids(df, mock_validation_context, male, female, both)


@pytest.mark.parametrize('sex_ids, male, female, both, warn_extra, warn_missing',
                         [([1, 2, 1], True, True, True, False, True),
                          ([1, 2, 3, 2], True, True, False, True, False),
                          ([1], False, True, False, True, True)])
def test_check_sex_ids_warn(mock_validation_context, caplog, sex_ids, male, female, both, warn_extra, warn_missing):
    df = pd.DataFrame({'sex_id': sex_ids})
    raw.check_sex_ids(df, mock_validation_context, male, female, both)
    assert warn_extra == ('extra' in caplog.text), 'The expected warning message was not raised for extra sex ids'
    assert warn_missing == ('missing' in caplog.text), 'The expected warning message was not raised for missing sex ids'


@pytest.mark.parametrize('sex_ids, male, female, both', [([1, 1, 1], True, False, False),
                                                         ([1, 2, 2, 2], True, True, False),
                                                         ([1, 2, 3, 3, 2, 1], True, True, True),
                                                         ([], False, False, False)])
def test_check_sex_ids_pass(mock_validation_context, sex_ids, male, female, both, recwarn):
    df = pd.DataFrame({'sex_id': sex_ids})
    raw.check_sex_ids(df, mock_validation_context, male, female, both)

    assert len(recwarn) == 0, 'An unexpected warning was raised.'


test_data = [
    (pd.DataFrame({'age_group_id': [1, 2, 3], 'a': 1, 'b': 0}), 1, 4, ['a', 'b'], 'missing'),
    (pd.DataFrame({'age_group_id': [1, 2], 'a': [0, 0], 'b': [0, 0]}), 1, 2, ['a', 'b'], 'missing for all age groups')
]


@pytest.mark.parametrize('data, start, end, val_cols, match', test_data)
def test_check_age_restrictions_fail(mock_validation_context, data, start, end, val_cols, match):
    with pytest.raises(DataAbnormalError, match=match):
        raw.check_age_restrictions(data, mock_validation_context, start, end, value_columns=val_cols)


test_data = [(pd.DataFrame({'age_group_id': [1, 2, 3, 4, 5], 'a': 1, 'b': 0}), 1, 4, ['a', 'b']),
             (pd.DataFrame({'age_group_id': [1, 2, 3], 'a': [1, 1, 1], 'b': [2, 3, 0]}), 1, 2, ['a'])]


@pytest.mark.parametrize('data, start, end, val_cols', test_data)
def test_check_age_restrictions_warn(mock_validation_context, caplog, data, start, end, val_cols):
    raw.check_age_restrictions(data, mock_validation_context, start, end, value_columns=val_cols)
    assert 'also included' in caplog.text


test_data = [(pd.DataFrame({'age_group_id': [1, 2, 3], 'a': 1, 'b': 0}), 1, 3, ['a', 'b']),
             (pd.DataFrame({'age_group_id': [1, 2], 'a': [1, 0], 'b': [1, 0.1]}), 1, 2, ['a', 'b']),
             (pd.DataFrame({'age_group_id': [1, 2, 3], 'a': [1, 1, 0], 'b': [2, 3, 0]}), 1, 3, ['a'])]


@pytest.mark.parametrize('data, start, end, val_cols', test_data)
def test_check_age_restrictions_pass(mock_validation_context, data, start, end, val_cols):
    raw.check_age_restrictions(data, mock_validation_context, start, end, value_columns=val_cols)


test_data = [(pd.DataFrame({'sex_id': [1, 1], 'a': 0, 'b': 0, 'c': 1}), True, False,
              ['a', 'b'], 'missing data values for males'),
             (pd.DataFrame({'sex_id': [2, 2, 2, 3], 'a': 0, 'b': 0}), False, True,
              ['a', 'b'], 'missing data values for females'),
             (pd.DataFrame({'sex_id': [3], 'a': [0], 'b': [0]}), False, False,
             ['a', 'b'], 'values for both'),
             (pd.DataFrame({'sex_id': [1, 2], 'a': [0, 1], 'b': [0, 1]}), False, False,
              ['a', 'b'], 'values for both')]


@pytest.mark.parametrize('data, male, female, val_cols, match', test_data)
def test_check_sex_restrictions_fail(mock_validation_context, data, male, female, val_cols, match):
    with pytest.raises(DataAbnormalError, match=match):
        raw.check_sex_restrictions(data, mock_validation_context, male, female, value_columns=val_cols)


test_data = [(pd.DataFrame({'sex_id': [1, 2], 'a': 1, 'b': 0}), True, False,
              ['a', 'b'], 'non-male sex ids'),
             (pd.DataFrame({'sex_id': [1, 2], 'a': 1, 'b': 0}), False, True,
              ['a', 'b'], 'non-female sex ids')]


@pytest.mark.parametrize('data, male, female, val_cols, match', test_data)
def test_check_sex_restrictions_warn(mock_validation_context, caplog, data, male, female, val_cols, match):
    raw.check_sex_restrictions(data, mock_validation_context, male, female, value_columns=val_cols)
    assert match in caplog.text


test_data = [(pd.DataFrame({'sex_id': [1, 1], 'a': 1, 'b': 0, 'c': 0}), True, False, ['a', 'b']),
             (pd.DataFrame({'sex_id': [2, 2], 'a': 1, 'b': 0}), False, True, ['a', 'b']),
             (pd.DataFrame({'sex_id': [3, 3, 3], 'a': 1, 'b': 0}), False, False, ['a']),
             (pd.DataFrame({'sex_id': [1, 2], 'a': [1, 0], 'b': [0, 1]}), False, False, ['a', 'b']),
             (pd.DataFrame({'sex_id': [3], 'a': [0], 'b': [1]}), False, False, ['a', 'b'])]


@pytest.mark.parametrize('data, male, female, val_cols', test_data)
def test_check_sex_restrictions_pass(mock_validation_context, data, male, female, val_cols):
    raw.check_sex_restrictions(data, mock_validation_context, male, female, value_columns=val_cols)


@pytest.mark.parametrize('m_ids, expected, match', [([1, 1, 1, 12], ['A'], 'multiple'),
                                                    ([3, 3, 3], ['A', 'B'], 'not in the expected')])
def test_check_measure_id_fail(m_ids, expected, match, measures_mock):
    df = pd.DataFrame({'measure_id': m_ids})
    with pytest.raises(DataAbnormalError, match=match):
        raw.check_measure_id(df, expected)


@pytest.mark.parametrize('m_ids, expected', [([1], ['A']),
                                             ([2, 2, 2, 2], ['A', 'B'])])
def test_check_measure_id_pass(m_ids, expected, measures_mock):
    df = pd.DataFrame({'measure_id': m_ids})
    raw.check_measure_id(df, expected)


@pytest.mark.parametrize('m_ids, expected', [([1, 1, 1, 12], 'A'),
                                             ([3, 3, 3], 'A')])
def test_check_metric_id_fail(m_ids, expected, metrics_mock):
    df = pd.DataFrame({'metric_id': m_ids})
    with pytest.raises(DataAbnormalError):
        raw.check_metric_id(df, expected)


@pytest.mark.parametrize('m_ids, expected', [([1], 'A'),
                                             ([2, 2, 2, 2], 'B')])
def test_check_metric_id_pass(m_ids, expected, metrics_mock):
    df = pd.DataFrame({'metric_id': m_ids})
    raw.check_metric_id(df, expected)
