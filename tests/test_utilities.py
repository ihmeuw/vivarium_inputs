import pytest

import pandas as pd
import numpy as np

from itertools import product

from vivarium_inputs.utilities import standardize_dimensions
from vivarium_inputs.core import UnhandledDataError


def test_standardize_dimensions__fill():
    expected = pd.MultiIndex.from_product([range(22), ['Male', 'Female'], [1990, 1995, 2000, 2005]],
                                          names=['age_group_id', 'sex', 'year'])

    actual = list(product([12, 13, 14], ['Male'], [1990, 1995, 2000, 2005], ['value', 'other_value']))
    actual = pd.DataFrame(actual, columns=['age_group_id', 'sex', 'year', 'measure'])

    draws = [f'draw_{i}' for i in range(1000)]
    actual = actual.assign(**{d: 0.0 for d in draws})
    with_val = actual['measure'] == 'value'
    with_other_val = actual['measure'] == 'other_value'
    actual.loc[with_val] = actual.loc[with_val].assign(**{d: 10.0 for d in draws}, )
    actual.loc[with_other_val] = actual.loc[with_other_val].assign(**{d: 1.5 for d in draws})
    actual = actual.set_index(['age_group_id', 'sex', 'year', 'measure'])

    fills = {'value': -1, 'other_value': -2}

    new_actual = standardize_dimensions(actual.reset_index(), expected, fills)
    new_actual = new_actual.set_index(['age_group_id', 'sex', 'year'])
    actual_index = actual.reset_index().set_index(['age_group_id', 'sex', 'year']).index
    assert new_actual.index.symmetric_difference(expected).empty
    assert np.all(new_actual.loc[expected.difference(actual_index)].query('measure == "value"')[draws] == -1)
    assert np.all(new_actual.loc[expected.difference(actual_index)].query('measure == "other_value"')[draws] == -2)
    new_actual = new_actual.reset_index().set_index(['age_group_id', 'sex', 'year', 'measure'])
    assert np.all(new_actual.loc[actual.index].reindex(sorted(new_actual.columns), axis=1)
                  == actual.reindex(sorted(actual.columns), axis=1))


@pytest.mark.skip("Cluster")
def test_standardize_dimensions__interpolate():
    from core_maths.interpolate import interpolate
    expected = pd.MultiIndex.from_product([range(22), ['Male'], [1991, 1995, 2000, 2005]],
                                          names=['age_group_id', 'sex', 'year'])

    actual = list(product(range(22), ['Male'], [1990, 2005], ['value']))
    actual = pd.DataFrame(actual, columns=['age_group_id', 'sex', 'year', 'measure'])

    draws = [f'draw_{i}' for i in range(1000)]
    actual = actual.assign(**{d: 0.0 for d in draws})
    actual.loc[actual['year'] == 1990] = actual.loc[actual['year'] == 1990].assign(**{d: 10.0 for d in draws})
    actual.loc[actual['year'] == 2005] = actual.loc[actual['year'] == 2005].assign(**{d: 100.0 for d in draws})
    actual = actual.set_index(['age_group_id', 'sex', 'year', 'measure'])

    fills = {'value': -1}

    to_interpolate = pd.DataFrame({'year': [1990, 2005], 'sex': ['Male', 'Male'], 'value': [10.0, 100.0]})
    interpolated = interpolate(to_interpolate.query('year == 1990').reset_index(drop=True),
                               to_interpolate.query('year == 2005').reset_index(drop=True),
                               ['sex', 'year'], 'year', ['value'], 1990, 2005)
    interpolated = interpolated.query('year in [1991, 1995, 2000, 2005]')

    new_actual = standardize_dimensions(actual.reset_index(), expected, fills)
    new_actual = new_actual.set_index(['age_group_id', 'sex', 'year'])
    assert new_actual.index.symmetric_difference(expected).empty
    assert np.allclose(
        new_actual.reset_index()[['year', 'draw_0']].drop_duplicates('year').set_index('year').sort_index(),
        interpolated[['year', 'value']].set_index('year').sort_index()
    )


def test_standardize_dimensions__missing_dimensions():
    expected = pd.MultiIndex.from_product([range(100), ['Male', 'Female'], [1990, 1995, 2000, 2005]],
                                          names=['age_group_id', 'sex', 'year'])

    actual = list(product([80,81,82], [1990, 1995, 2000, 2005], ['value']))
    actual = pd.DataFrame(actual, columns=['age_group_id', 'year', 'measure'])

    actual['draw_0'] = 10.0
    actual = actual.set_index(['age_group_id', 'year'])
    fills = {'value': -1}

    new_actual = standardize_dimensions(actual.reset_index(), expected, fills)

    expected = expected.droplevel('sex')
    assert 'sex' not in new_actual
    new_actual = new_actual.set_index(['age_group_id', 'year'])
    assert new_actual.index.symmetric_difference(expected).empty
    assert np.all(new_actual.loc[expected.difference(actual.index)].draw_0 == -1)
    assert np.all(new_actual.loc[actual.index] == actual)


def test_standardize_dimensions__malformed():
    expected = pd.MultiIndex.from_product([range(100), ['Male', 'Female'], [1990, 1995, 2000, 2005]],
                                          names=['age_group_id', 'sex', 'year'])

    actual = list(product([10, 12, 11, 80,81,82], ['Male'], [1990, 2005], ['value']))
    actual = pd.DataFrame(actual, columns=['age_group_id', 'sex', 'year', 'measure'])

    actual['draw_0'] = 10.0
    actual = actual.set_index(['age_group_id', 'sex', 'year'])

    fills = {'value': -1}

    with pytest.raises(UnhandledDataError):
        new_actual = standardize_dimensions(actual.reset_index(), expected, fills)


def test_standardize_dimensions__missing_fills():
    expected = pd.MultiIndex.from_product([range(100), ['Male', 'Female'], [1990, 1995, 2000, 2005]],
                                          names=['age_group_id', 'sex', 'year'])

    actual = list(product([10, 12, 11, 80,81,82], ['Male'], [1990, 2005], ['value', 'other_value']))
    actual = pd.DataFrame(actual, columns=['age_group_id', 'sex', 'year', 'measure'])

    actual['draw_0'] = 10.0
    actual = actual.set_index(['age_group_id', 'sex', 'year'])

    fills = {'value': -1}

    with pytest.raises(AssertionError):
        new_actual = standardize_dimensions(actual.reset_index(), expected, fills)
