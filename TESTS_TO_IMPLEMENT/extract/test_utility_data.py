import gbd_mapping as gm
import pandas as pd
import pytest

from tests.conftest import NO_GBD_ACCESS
from vivarium_inputs import utility_data as ud

pytestmark = pytest.mark.skipif(
    NO_GBD_ACCESS, reason="Cannot run these tests without vivarium_gbd_access"
)


def test_get_estimation_years():
    result = ud.get_estimation_years()
    assert set([1990, 1995, 2000, 2005, 2010, 2015, 2019, 2020, 2021, 2022]) == set(result)


def test_get_year_block():
    result = ud.get_year_block()
    assert result.year_start.iloc[0] == 1990
    assert result.year_end.iloc[-1] == 2023


def test_get_age_group_ids():
    result = ud.get_age_group_ids()
    truth = [2, 3, 388, 389, 238, 34] + list(range(6, 21)) + [30, 31, 32, 235]
    assert result == truth


def test_get_location_id():
    result_india = ud.get_location_id("India")
    assert 163 == result_india


@pytest.mark.parametrize(
    "location_id, expected",
    [
        (163, {163: [1, 158, 159, 163]}),
        ([163, 175], {163: [1, 158, 159, 163], 175: [1, 166, 174, 175]}),
    ],
)
def test_get_location_id_parents(location_id, expected):
    assert expected == ud.get_location_id_parents(location_id)


def test_get_demographic_dimensions():
    result = ud.get_demographic_dimensions(163)
    assert len(result)
    assert set(["location_id", "sex_id", "age_group_id", "year_id"]) == set(result.columns)


def test_get_tmrel_category():
    result = ud.get_tmrel_category(gm.risk_factors.smoking)
    assert "cat2" == result
