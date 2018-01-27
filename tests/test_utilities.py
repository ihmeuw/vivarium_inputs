import pytest

import pandas as pd
import numpy as np

from itertools import product

from ceam_inputs.utilities import standardize_dimensions


def test_standardize_dimensions__fill():
    expected = pd.MultiIndex.from_product([range(100), ['Male', 'Female'], [1990, 1995, 2000, 2005]], names=['age', 'sex', 'year'])

    actual = list(product([80,81,82], ['Male'], [2000, 2005]))
    actual = pd.DataFrame(actual, columns=['age', 'sex', 'year'])

    actual['value'] = 10.0
    actual['other_value'] = "thing"
    actual = actual.set_index(['age', 'sex', 'year'])

    fills = {'value': -1, 'other_value': 'something else'}

    new_actual = standardize_dimensions(actual, fills, expected)
    new_actual = new_actual.set_index(['age', 'sex', 'year'])
    assert new_actual.index.symmetric_difference(expected).empty
    assert np.all(new_actual.loc[expected.difference(actual.index)].value == -1)
    assert np.all(new_actual.loc[expected.difference(actual.index)].other_value == 'something else')
    assert np.all(new_actual.loc[actual.index] == actual)

def test_standardize_dimensions__missing_dimensions():
    expected = pd.MultiIndex.from_product([range(100), ['Male', 'Female'], [1990, 1995, 2000, 2005]], names=['age', 'sex', 'year'])

    actual = list(product([80,81,82], [2000, 2005]))
    actual = pd.DataFrame(actual, columns=['age', 'year'])

    actual['value'] = 10.0
    actual = actual.set_index(['age', 'year'])
    fills = {'value': -1}


    new_actual = standardize_dimensions(actual, fills, expected)

    expected = expected.droplevel('sex')
    assert 'sex' not in new_actual
    new_actual = new_actual.set_index(['age', 'year'])
    assert new_actual.index.symmetric_difference(expected).empty
    assert np.all(new_actual.loc[expected.difference(actual.index)].value == -1)
    assert np.all(new_actual.loc[actual.index] == actual)


def test_standardize_dimensions__malformed():
    expected = pd.MultiIndex.from_product([range(100), ['Male', 'Female'], [1990, 1995, 2000, 2005]], names=['age', 'sex', 'year'])

    actual = list(product([10, 12, 11, 80,81,82], ['Male'], [2000, 2005]))
    actual = pd.DataFrame(actual, columns=['age', 'sex', 'year'])

    actual['value'] = 10.0
    actual['other_value'] = "thing"
    actual = actual.set_index(['age', 'sex', 'year'])

    fills = {'value': -1, 'other_value': 'something else'}

    with pytest.raises(ValueError):
        new_actual = standardize_dimensions(actual, fills, expected)
