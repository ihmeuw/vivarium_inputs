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
from vivarium_inputs.globals import DRAW_COLUMNS, MEASURES, DataTypeNotImplementedError
from vivarium_inputs.interface import get_measure
from vivarium_inputs.utility_data import get_age_group_ids

CAUSE = causes.hiv_aids
LOCATION = "India"
YEAR = 2021


@pytest.fixture
def mocked_hiv_aids_incidence_rate() -> pd.DataFrame:
    """Mocked vivarium_gbd_access data for testing."""
    age_group_ids = get_age_group_ids()
    sex_ids = [1, 2]
    measure_ids = [3, 5, 6]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, measure_ids)),
        columns=["age_group_id", "sex_id", "measure_id"],
    )

    # Add on other metadata columns
    df["cause_id"] = CAUSE.gbd_id.real
    df["location_id"] = 163  # for India
    df["year_id"] = YEAR
    df["metric_id"] = 3  # for rate
    df["version_id"] = 1471  # not sure where this comes from

    # Add on randomized raw columns
    np.random.seed(42)
    for col in DRAW_COLUMNS:
        df[col] = np.random.uniform(0, 0.06, size=len(df))

    # Reorder the columns based on what I saw returned from gbd at some point
    df = df[
        ["age_group_id", "cause_id"]
        + DRAW_COLUMNS
        + ["location_id", "measure_id", "sex_id", "year_id", "metric_id", "version_id"]
    ]

    return df


@pytest.fixture(autouse=True)
def no_cache(mocker: MockerFixture):
    """Mock out the cache so that we always pull data."""

    if not NO_GBD_ACCESS:
        mocker.patch(
            "vivarium_gbd_access.utilities.get_input_config",
            return_value=LayeredConfigTree({"input_data": {"cache_data": False}}),
        )


@pytest.mark.parametrize(
    "data_type", ["mean", "draw", ["mean", "draw"]], ids=("mean", "draw", "mean_draw")
)
@pytest.mark.parametrize("mock_gbd", [True, False], ids=("mock_gbd", "no_mock_gbd"))
def test_get_incidence_rate(
    data_type: str | list[str],
    mock_gbd: bool,
    mocked_hiv_aids_incidence_rate: pd.DataFrame,
    runslow: bool,
    mocker: MockerFixture,
):
    """Test get_measure function.

    If mock_gbd is True, the test will mock vivarium_gbd_access calls with a
    dummy data fixutre. This allows for pseodu-testing on github actions which
    does not have access to vivarium_gbd_access. If mock_gb is False, the test
    is a full end-to-end test, is marked as slow (requires --runslow option to
    run), and will be skipped if there is not access to vivarium_gbd_access
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

    if mock_gbd:
        # Test against mocked data instead of actual data retrieval
        if isinstance(data_type, list):
            with pytest.raises(DataTypeNotImplementedError):
                data = get_measure(**kwargs)
        elif data_type == "mean":
            with pytest.raises(DataTypeNotImplementedError):
                data = get_measure(**kwargs)
        else:
            mock_extract_incidence_rate = mocker.patch(
                "vivarium_inputs.extract.extract_incidence_rate",
                return_value=mocked_hiv_aids_incidence_rate[
                    mocked_hiv_aids_incidence_rate["measure_id"] == MEASURES["Incidence rate"]
                ],
            )
            mock_extract_prevalence = mocker.patch(
                "vivarium_inputs.extract.extract_prevalence",
                return_value=mocked_hiv_aids_incidence_rate[
                    mocked_hiv_aids_incidence_rate["measure_id"] == MEASURES["Prevalence"]
                ],
            )
            data = get_measure(**kwargs)
            mock_extract_incidence_rate.assert_called_once()
            mock_extract_prevalence.assert_called_once()
            assert not data.empty
            assert all(col in data.columns for col in DRAW_COLUMNS)

    if not mock_gbd and runslow:
        if NO_GBD_ACCESS:
            pytest.skip("Need GBD access to run this test")

        # Test actual data retrieval
        if isinstance(data_type, list):
            with pytest.raises(DataTypeNotImplementedError):
                data = get_measure(**kwargs)
        elif data_type == "mean":
            with pytest.raises(DataTypeNotImplementedError):
                data = get_measure(**kwargs)
        else:
            data = get_measure(**kwargs)
            assert not data.empty
            assert all(col in data.columns for col in DRAW_COLUMNS)
