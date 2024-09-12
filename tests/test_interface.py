"""
Interface tests.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from gbd_mapping import causes
from layered_config_tree import LayeredConfigTree
from pytest_mock import MockerFixture

from tests.conftest import NO_GBD_ACCESS
from vivarium_inputs.globals import (
    DRAW_COLUMNS,
    MEAN_COLUMNS,
    MEASURES,
    SUPPORTED_DATA_TYPES,
)
from vivarium_inputs.interface import get_measure
from vivarium_inputs.utilities import DataTypeNotImplementedError

CAUSE = causes.hiv_aids
LOCATION = 163  # India
YEAR = 2021

SUPPORTED_DATA_TYPES = {
    "mean": MEAN_COLUMNS,
    "draw": DRAW_COLUMNS,
}


def get_mocked_estimation_years() -> list[int]:
    """Mocked estimation years for testing."""
    return [1990, 1995, 2000, 2005, 2010, 2015, 2019, 2020, 2021, 2022]


def get_mocked_age_groups() -> dict[int, str]:
    """Mocked age group ids and names for testing."""
    return {
        2: "Early Neonatal",
        3: "Late Neonatal",
        388: "1-5 months",
        389: "6-11 months",
        238: "12 to 23 months",
        34: "2 to 4",
        6: "5 to 9",
        7: "10 to 14",
        8: "15 to 19",
        9: "20 to 24",
        10: "25 to 29",
        11: "30 to 34",
        12: "35 to 39",
        13: "40 to 44",
        14: "45 to 49",
        15: "50 to 54",
        16: "55 to 59",
        17: "60 to 64",
        18: "65 to 69",
        19: "70 to 74",
        20: "75 to 79",
        30: "80 to 84",
        31: "85 to 89",
        32: "90 to 94",
        235: "95 plus",
    }


def get_mocked_location_path_to_global() -> pd.DataFrame:
    """Mocked location path to global for testing."""
    return pd.read_csv("tests/fixture_data/location_path_to_global.csv")


def get_mocked_age_bins() -> pd.DataFrame:
    """Mocked age bins for testing."""
    return pd.read_csv("tests/fixture_data/age_bins.csv")


def get_mocked_location_ids() -> pd.DataFrame:
    """Mocked location ids for testing."""
    return pd.read_csv("tests/fixture_data/location_ids.csv")


@pytest.fixture
def mocked_hiv_aids_incidence_rate_draws() -> pd.DataFrame:
    """Mocked vivarium_gbd_access draw-level data for testing."""
    age_group_ids = list(get_mocked_age_groups())
    sex_ids = [1, 2]
    measure_ids = [3, 5, 6]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, measure_ids)),
        columns=["age_group_id", "sex_id", "measure_id"],
    )

    # Add on other metadata columns
    df["cause_id"] = CAUSE.gbd_id.real
    df["location_id"] = LOCATION
    df["year_id"] = YEAR
    df["metric_id"] = 3  # for rate
    df["version_id"] = 1471  # not sure where this comes from

    # Add on randomized raw columns
    np.random.seed(42)
    for col in DRAW_COLUMNS:
        df[col] = np.random.uniform(0, 0.0004, size=len(df))

    # Reorder the columns based on what I saw returned from gbd at some point
    df = df[
        ["age_group_id", "cause_id"]
        + DRAW_COLUMNS
        + ["location_id", "measure_id", "sex_id", "year_id", "metric_id", "version_id"]
    ]

    return df


@pytest.fixture
def mocked_hiv_aids_incidence_rate_means() -> pd.DataFrame:
    """Mocked vivarium_gbd_access mean data for testing."""
    age_groups = get_mocked_age_groups()
    age_group_ids = list(age_groups)
    sex_ids = [1, 2]
    measure_ids = [5, 6]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, measure_ids)),
        columns=["age_group_id", "sex_id", "measure_id"],
    )

    # Add on other metadata columns
    df["cause_id"] = CAUSE.gbd_id.real
    df["location_id"] = LOCATION
    df["year_id"] = YEAR
    df["metric_id"] = 3  # for rate
    df["acause"] = "hiv"
    df["age_group_name"] = df["age_group_id"].map(age_groups)
    df["cause_name"] = "HIV/AIDS"
    df["expected"] = False
    df["location_name"] = "India"
    df["location_type"] = "admin0"
    df["measure"] = df["measure_id"].map({5: "incidence", 6: "prevalence"})
    df["measure_name"] = df["measure_id"].map({5: "Incidence", 6: "Prevalence"})
    df["metric_name"] = "Rate"
    df["sex"] = df["sex_id"].map({1: "Male", 2: "Female"})

    # Add on randomized mean columns
    np.random.seed(42)
    for col in MEAN_COLUMNS:
        df[col] = np.random.uniform(0, 0.0004, size=len(df))
    df["upper"] = df["val"] * 1.1
    df["lower"] = df["val"] * 0.9

    # Reorder the columns based on what I saw returned from gbd at some point
    df = df[
        [
            "age_group_id",
            "cause_id",
            "location_id",
            "measure_id",
            "metric_id",
            "sex_id",
            "year_id",
            "acause",
            "age_group_name",
            "cause_name",
            "expected",
            "location_name",
            "location_type",
            "measure",
            "measure_name",
            "metric_name",
            "sex",
        ]
        + MEAN_COLUMNS
        + ["upper", "lower"]
    ]

    return df


@pytest.fixture(autouse=True)
def no_cache(mocker: MockerFixture) -> None:
    """Mock out the cache so that we always pull data."""

    if not NO_GBD_ACCESS:
        mocker.patch(
            "vivarium_gbd_access.utilities.get_input_config",
            return_value=LayeredConfigTree({"input_data": {"cache_data": False}}),
        )


@pytest.mark.parametrize(
    "data_type", ["means", "draws", ["means", "draws"]], ids=("means", "draws", "means_draws")
)
@pytest.mark.parametrize("mock_gbd", [True, False], ids=("mock_gbd", "no_mock_gbd"))
def test_get_incidence_rate(
    data_type: str | list[str],
    mock_gbd: bool,
    mocked_hiv_aids_incidence_rate_draws: pd.DataFrame,
    mocked_hiv_aids_incidence_rate_means: pd.DataFrame,
    runslow: bool,
    mocker: MockerFixture,
) -> None:
    """Test get_measure function.

    If mock_gbd is True, the test will mock vivarium_gbd_access calls with
    dummy data. This allows for pseudo-testing on github actions which
    does not have access to vivarium_gbd_access. If mock_gbd is False, we run
    a full end-to-end test marked as slow (i.e. requires the '--runslow' option
    to run) which will be skipped if there is no access to vivarium_gbd_access
    (i.e. it can run in a Jenkins job).
    """

    if not mock_gbd and not runslow:
        pytest.skip("need --runslow option to run")

    kwargs = {
        "entity": CAUSE,
        "measure": "incidence_rate",
        "location": LOCATION,
        "years": YEAR,
        "data_type": data_type,
    }

    if isinstance(data_type, list):
        with pytest.raises(DataTypeNotImplementedError):
            data = get_measure(**kwargs)
    else:
        if mock_gbd:
            # Test against mocked data instead of actual data retrieval
            mocked_return = {
                "means": mocked_hiv_aids_incidence_rate_means,
                "draws": mocked_hiv_aids_incidence_rate_draws,
            }[data_type]
            mock_extract_incidence_rate = mocker.patch(
                "vivarium_inputs.extract.extract_incidence_rate",
                return_value=mocked_return[
                    mocked_return["measure_id"] == MEASURES["Incidence rate"]
                ],
            )
            mock_extract_prevalence = mocker.patch(
                "vivarium_inputs.extract.extract_prevalence",
                return_value=mocked_return[
                    mocked_return["measure_id"] == MEASURES["Prevalence"]
                ],
            )
            mocker.patch(
                "vivarium_inputs.utility_data.get_estimation_years",
                return_value=get_mocked_estimation_years(),
            )
            mocker.patch(
                "vivarium_inputs.utility_data.get_age_group_ids",
                return_value=list(get_mocked_age_groups()),
            )
            mocker.patch(
                "vivarium_inputs.utility_data.get_location_path_to_global",
                return_value=get_mocked_location_path_to_global(),
            )
            mocker.patch(
                "vivarium_inputs.utility_data.get_raw_age_bins",
                return_value=get_mocked_age_bins(),
            )
            mocker.patch(
                "vivarium_inputs.utility_data.get_raw_location_ids",
                return_value=get_mocked_location_ids(),
            )

            data = get_measure(**kwargs)

            mock_extract_incidence_rate.assert_called_once()
            mock_extract_prevalence.assert_called_once()
            assert not data.empty
            assert all(col in data.columns for col in SUPPORTED_DATA_TYPES[data_type])
            assert all(data.notna())
            assert all(data >= 0)

        if not mock_gbd and runslow:
            # Test actual data retrieval
            if NO_GBD_ACCESS:
                pytest.skip("Need GBD access to run this test")
            data = get_measure(**kwargs)
            assert not data.empty
            assert all(col in data.columns for col in SUPPORTED_DATA_TYPES[data_type])
            assert all(data.notna())
