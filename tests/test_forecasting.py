import pandas as pd
import pytest

from vivarium_inputs import forecasting


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

    expected_base = pd.DataFrame({'year_start': [1990, 2000, 2010, 2020],
                             'year_end': [2000, 2010, 2020, 2021], 'draw': 0,
                             'value': range(0, 4)})

    expected = expected_base.copy()
    for i in range(1, forecasting.NUM_DRAWS):
        draw = expected_base.copy()
        draw['draw'] = i
        expected = expected.append(draw)

    expected = expected.set_index(['year_start', 'year_end', 'draw'])

    normalized = forecasting.normalize_forecasting(data).set_index(['year_start', 'year_end', 'draw'])

    assert normalized.equals(expected)


def test_standardize_data(mocker):
    gbd_mock_utilities = mocker.patch("vivarium_inputs.forecasting.gbd")
    gbd_mock_utilities.get_age_group_id.return_value = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                        20, 30, 31, 32, 235]

    data = pd.DataFrame({'year_id': [1990, 2000, 2010, 2020], 'scenario': [0]*4,
                         'sex_id': 1, 'location_id': 102,
                         'value': range(1, 5), 'age_group_id': 2, 'draw': 1})

    standardized = forecasting.standardize_data(data, 0)

    assert 0 in standardized.value

    assert set(standardized.age_group_id) == set(forecasting.gbd.get_age_group_id())
