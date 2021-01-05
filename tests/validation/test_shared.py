import pandas as pd
import pytest

from vivarium_inputs.globals import DataAbnormalError, DataTransformationError
from vivarium_inputs.validation import shared

test_data = [(pd.DataFrame({'a': [0, 1], 'b': [2, 20]}), 1, 'lower', ['a', 'b'],
              True, 'values below', DataAbnormalError),
             (pd.DataFrame({'a': [0, 1], 'b': [2, 20]}), 0, 'lower', ['a', 'b'],
              False, 'values below or equal to', ValueError),
             (pd.DataFrame({'a': [0], 'b': [10], 'c': [100]}), 10, 'lower', ['a'],
              True, 'values below', ValueError),
             (pd.DataFrame({'a': [0, 1], 'b': [2, 20]}), 20, 'upper', ['a', 'b'],
              False, 'values above', DataAbnormalError),
             (pd.DataFrame({'a': [0, 1], 'b': [2, 20]}, index=[0, 1]), pd.Series([5, 10], index=[0, 1]), 'upper',
              ['a', 'b'], True, 'above the expected', DataTransformationError)]


@pytest.mark.parametrize('data, boundary, boundary_type, val_cols, inclusive, match, error', test_data)
def test_check_value_columns_boundary_fail_warn(caplog, data, boundary, boundary_type, val_cols, inclusive, match, error):
    with pytest.raises(error, match=match):
        shared.check_value_columns_boundary(data, boundary, boundary_type, val_cols, inclusive=inclusive, error=error)

    shared.check_value_columns_boundary(data, boundary, boundary_type, val_cols, inclusive=inclusive, error=None)
    assert match in caplog.text


test_data = [(pd.DataFrame({'a': [0, 1], 'b': [2, 20]}), 0, 'lower', ['a', 'b'], True),
             (pd.DataFrame({'a': [0, 1], 'b': [2, 20]}), -1, 'lower', ['a', 'b'], False),
             (pd.DataFrame({'a': [0], 'b': [10], 'c': [100]}), 10, 'lower', ['b', 'c'], True),
             (pd.DataFrame({'a': [0, 1], 'b': [2, 20]}), 20, 'upper', ['a', 'b'], True),
             (pd.DataFrame({'a': [0, 1], 'b': [2, 20]}, index=[0, 1]), pd.Series([0, 1], index=[0, 1]), 'upper',
              ['a'], True)]


@pytest.mark.parametrize('data, boundary, boundary_type, val_cols, inclusive', test_data)
def test_check_value_columns_boundary_pass(data, boundary, boundary_type, val_cols, inclusive):
    shared.check_value_columns_boundary(data, boundary, boundary_type, val_cols,
                                        inclusive=inclusive, error=ValueError)
