"""
Interface tests.
"""

import itertools
from typing import Union

import numpy as np
import pandas as pd
import pytest
from gbd_mapping import causes
from layered_config_tree import LayeredConfigTree
from pytest_mock import MockerFixture
from vivarium_gbd_access.gbd import DataTypeNotImplementedError

from tests.conftest import RUNNING_ON_CI
from vivarium_inputs.globals import DRAW_COLUMNS
from vivarium_inputs.interface import get_measure
from vivarium_inputs.utility_data import get_age_group_ids

CAUSE = causes.hiv_aids
LOCATION = "India"
YEAR = 2021


@pytest.fixture
def mocked_raw_gbd_draws() -> pd.DataFrame:
    """Mocked raw GBD data for testing."""
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


MOCKED_DATA = pd.DataFrame()


@pytest.fixture(autouse=True)
def no_cache(mocker: MockerFixture):
    """Mock out the cache so that we always pull data."""

    mocker.patch(
        "vivarium_gbd_access.utilities.get_input_config",
        return_value=LayeredConfigTree({"input_data": {"cache_data": False}}),
    )


@pytest.mark.parametrize(
    "data_type", ["mean", "draw", ["mean", "draw"]], ids=("mean", "draw", "mean_draw")
)
@pytest.mark.parametrize(
    "mock_gbd_call", [True, False], ids=("mock_gbd_call", "no_mock_gbd_call")
)
def test_get_incidence_rate(
    data_type: Union[str, list[str]],
    mock_gbd_call: bool,
    mocked_raw_gbd_draws: pd.DataFrame,
    runslow: bool,
    mocker: MockerFixture,
):

    if not mock_gbd_call and not runslow:
        pytest.skip("need --runslow option to run")

    kwargs = {
        "entity": CAUSE,
        "measure": "incidence_rate",
        "location": LOCATION,
        "years": YEAR,
        "data_type": data_type,
    }

    if mock_gbd_call:
        # Test against mocked data instead of actual data retrieval
        if isinstance(data_type, list):
            with pytest.raises(
                DataTypeNotImplementedError,
                match="A list of requested data types is not yet supported",
            ):
                data = get_measure(**kwargs)
        elif data_type == "mean":
            with pytest.raises(
                DataTypeNotImplementedError,
                match="Getting mean values is not yet supported",
            ):
                data = get_measure(**kwargs)
        else:
            mock_call = mocker.patch(
                "vivarium_gbd_access.gbd.get_draws", return_value=mocked_raw_gbd_draws
            )
            data = get_measure(**kwargs)
            # get_draws is called twice, once for raw_incidence and once for prevalence
            assert mock_call.call_count == 2
            assert not data.empty
            assert all(col in data.columns for col in DRAW_COLUMNS)

    if not mock_gbd_call and runslow:
        if RUNNING_ON_CI:
            pytest.skip("Need GBD access to run this test")

        # Test actual data retrieval
        if isinstance(data_type, list):
            with pytest.raises(
                DataTypeNotImplementedError,
                match="A list of requested data types is not yet supported",
            ):
                data = get_measure(**kwargs)
        elif data_type == "mean":
            with pytest.raises(
                DataTypeNotImplementedError,
                match="Getting mean values is not yet supported",
            ):
                data = get_measure(**kwargs)
        else:
            data = get_measure(**kwargs)
            assert not data.empty
            assert all(col in data.columns for col in DRAW_COLUMNS)
