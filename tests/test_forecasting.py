import pandas as pd
import pytest

from vivarium_inputs import forecasting
from vivarium_inputs.utilities import gbd


def test_combine_past_future():
    past = pd.DataFrame({'year_start': [2000, 2005], 'value': [1, 2]})
    future = pd.DataFrame({'year_start': [2005, 2005, 2020, 2020], 'draw': [0, 1]*2, 'value': range(6, 10)})

    combo = forecasting.combine_past_future(past, future).set_index(['year_start', 'draw'])

    expected = pd.DataFrame({'year_start': [2000, 2000, 2005, 2005, 2020, 2020], 'draw': [0, 1]*3,
                             'value': [1, 1, 2, 2, 8, 9]}).set_index(['year_start', 'draw'])

    assert combo.equals(expected)


def test_rename_value_columns():
    data = pd.DataFrame({'year_id': [1990, 2000, 2010, 2020], 'scenario': 0, 'rename_me': range(0, 4)})

    assert forecasting.rename_value_columns(data).columns.tolist() == ['year_id', 'scenario', 'value']

    assert forecasting.rename_value_columns(data, 'population').columns.tolist() == ['year_id', 'scenario', 'population']

    data['new_value'] = 5

    with pytest.raises(ValueError):
        forecasting.rename_value_columns(data)


def test_normalize_forecasting():
    data = pd.DataFrame({'year_id': [1990, 2000, 2010, 2020], 'scenario': [0]*4,
                         'value': range(0, 4), 'age_group_id': 22})

    expected = pd.DataFrame({'year_start': [1990, 2000, 2010, 2020],
                             'year_end': [2000, 2010, 2020, 2021],
                             'value': range(0, 4)}).set_index(['year_start', 'year_end'])

    normalized = forecasting.normalize_forecasting(data).set_index(['year_start', 'year_end'])

    assert normalized.equals(expected)


def test_standardize_data():
    data = pd.DataFrame({'year_id': [1990, 2000, 2010, 2020], 'scenario': [0]*4,
                         'sex_id': 1, 'location_id': 102,
                         'value': range(1, 5), 'age_group_id': 2, 'draw': 1})

    standardized = forecasting.standardize_data(data, 0)

    assert 0 in standardized.value

    assert set(standardized.age_group_id) == set(gbd.get_age_group_id())