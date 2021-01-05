import pytest
import pandas as pd

import gbd_mapping as gm

from vivarium_inputs import utility_data as ud
from tests.extract.check import RUNNING_ON_CI


pytestmark = pytest.mark.skipif(RUNNING_ON_CI, reason="Don't run these tests on the CI server")


def test_get_estimation_years():
    result = ud.get_estimation_years()
    assert(set([1990, 1995, 2000, 2005, 2010, 2015, 2017, 2019]) == set(result))


def test_get_year_block():
    result = ud.get_year_block()
    assert(result.year_start.iloc[0] == 1990)
    assert(result.year_end.iloc[-1] == 2020)
    

def test_get_age_group_ids():
    result = ud.get_age_group_ids()
    truth = list(range(2, 21)) + [30, 31, 32, 235]
    assert(result == truth)


def test_get_location_id():
    result_india = ud.get_location_id('India')
    assert(163 == result_india)


def test_get_location_id_parents():
    assert([1, 158, 159, 163] == ud.get_location_id_parents(163))


def test_get_demographic_dimensions():
    result = ud.get_demographic_dimensions(163)
    assert(len(result))
    assert(set(['location_id', 'sex_id', 'age_group_id', 'year_id']) == set(result.columns))


def test_get_tmrel_category():
    result = ud.get_tmrel_category(gm.risk_factors.smoking)
    assert('cat2' == result)