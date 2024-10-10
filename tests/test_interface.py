"""
Interface tests.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from gbd_mapping import Cause, Sequela, causes, sequelae
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

#############
# CONSTANTS #
#############

HIV_AIDS = causes.hiv_aids
HIV_AIDS_DRUG_SUSCEPTIBLE_TB_WO_ANEMIA = (
    sequelae.hiv_aids_drug_susceptible_tuberculosis_without_anemia
)
LOCATION = 163  # India
YEAR = 2021


###############
# MOCKED DATA #
###############


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


MOCKED_AGE_GROUP_ENDPOINTS = {
    "age_start": [
        0.0,
        0.07671233,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        0.01917808,
        35.0,
        40.0,
        45.0,
        50.0,
        55.0,
        60.0,
        65.0,
        70.0,
        75.0,
        80.0,
        85.0,
        90.0,
        95.0,
    ],
    "age_end": [
        0.07671233,
        0.5,
        2.0,
        1.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        0.01917808,
        35.0,
        40.0,
        45.0,
        50.0,
        55.0,
        60.0,
        65.0,
        70.0,
        75.0,
        80.0,
        85.0,
        90.0,
        95.0,
        125.0,
    ],
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


def get_mocked_incidence_rate_get_draws_common_data() -> pd.DataFrame:
    """Common dataset for mocked get_draws() incidence_rate data."""
    age_group_ids = list(get_mocked_age_groups())
    sex_ids = [1, 2]
    measure_ids = [3, 5, 6]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, measure_ids)),
        columns=["age_group_id", "sex_id", "measure_id"],
    )

    # Add on other metadata columns
    df["location_id"] = LOCATION
    df["year_id"] = YEAR
    df["metric_id"] = 3  # for rate
    df["version_id"] = 1471  # not sure where this comes from

    # Add on randomized raw columns
    np.random.seed(42)
    for col in DRAW_COLUMNS:
        df[col] = np.random.uniform(0, 0.0004, size=len(df))

    return df


def get_mocked_incidence_rate_get_outputs_common_data() -> pd.DataFrame:
    """Common dataset for mocked get_outputs() incidence_rate data."""

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
    df["location_id"] = LOCATION
    df["year_id"] = YEAR
    df["metric_id"] = 3  # for rate
    df["age_group_name"] = df["age_group_id"].map(age_groups)
    df["expected"] = False
    df["location_name"] = "India"
    df["location_type"] = "admin0"
    df["measure"] = df["measure_id"].map({5: "prevalence", 6: "incidence"})
    df["measure_name"] = df["measure_id"].map({5: "Prevalence", 6: "Incidence"})
    df["metric_name"] = "Rate"
    df["sex"] = df["sex_id"].map({1: "Male", 2: "Female"})

    # Add on randomized mean columns
    np.random.seed(42)
    for col in MEAN_COLUMNS:
        df[col] = np.random.uniform(0, 0.0004, size=len(df))
    df["upper"] = df["val"] * 1.1
    df["lower"] = df["val"] * 0.9

    return df


############
# FIXTURES #
############


@pytest.fixture
def mocked_cause_incidence_rate_get_draws() -> pd.DataFrame:
    """Mocked vivarium_gbd_access data for testing."""

    df = get_mocked_incidence_rate_get_draws_common_data().copy()

    # Add on specific columns
    df["cause_id"] = int(HIV_AIDS.gbd_id)

    # Reorder the columns based on what I saw returned from gbd at some point
    df = df[
        ["age_group_id", "cause_id"]
        + DRAW_COLUMNS
        + ["location_id", "measure_id", "sex_id", "year_id", "metric_id", "version_id"]
    ]

    return df


@pytest.fixture
def mocked_sequela_incidence_rate_get_draws() -> pd.DataFrame:
    """Mocked vivarium_gbd_access data for testing."""

    df = get_mocked_incidence_rate_get_draws_common_data().copy()

    # Add on specific columns
    df["sequela_id"] = int(HIV_AIDS_DRUG_SUSCEPTIBLE_TB_WO_ANEMIA.gbd_id)

    # Reorder the columns based on what I saw returned from gbd at some point
    df = df[
        ["age_group_id"]
        + DRAW_COLUMNS
        + [
            "location_id",
            "measure_id",
            "sequela_id",
            "sex_id",
            "year_id",
            "metric_id",
            "version_id",
        ]
    ]

    return df


@pytest.fixture
def mocked_cause_incidence_rate_get_outputs() -> pd.DataFrame:
    """Mocked vivarium_gbd_access data for testing."""

    df = get_mocked_incidence_rate_get_outputs_common_data().copy()

    # Add on other metadata columns
    df["cause_id"] = int(HIV_AIDS.gbd_id)
    df["acause"] = "hiv"
    df["cause_name"] = "HIV/AIDS"

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


@pytest.fixture
def mocked_sequela_incidence_rate_get_outputs() -> pd.DataFrame:
    """Mocked vivarium_gbd_access data for testing."""

    df = get_mocked_incidence_rate_get_outputs_common_data().copy()

    # Add on other metadata columns
    df["sequela_id"] = int(HIV_AIDS_DRUG_SUSCEPTIBLE_TB_WO_ANEMIA.gbd_id)
    df["sequela_name"] = "HIV/AIDS -  Drug-susceptible Tuberculosis without anemia"

    # Reorder the columns based on what I saw returned from gbd at some point
    df = df[
        [
            "age_group_id",
            "location_id",
            "measure_id",
            "metric_id",
            "sequela_id",
            "sex_id",
            "year_id",
            "age_group_name",
            "expected",
            "location_name",
            "location_type",
            "measure",
            "measure_name",
            "metric_name",
            "sequela_name",
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


#########
# TESTS #
#########


@pytest.mark.slow
@pytest.mark.parametrize(
    "entity", [HIV_AIDS, HIV_AIDS_DRUG_SUSCEPTIBLE_TB_WO_ANEMIA], ids=("cause", "sequela")
)
@pytest.mark.parametrize(
    "data_type", ["means", "draws", ["means", "draws"]], ids=("means", "draws", "means_draws")
)
def test_get_incidence_rate(
    entity: Cause | Sequela,
    data_type: str | list[str],
) -> None:
    """Test get_measure function.

    Run a full end-to-end test marked as slow (i.e. requires the '--runslow' option
    to run) which will be skipped if there is no access to vivarium_gbd_access
    (i.e. it can run in a Jenkins job).
    """

    kwargs = {
        "entity": entity,
        "measure": "incidence_rate",
        "location": LOCATION,
        "years": YEAR,
        "data_type": data_type,
    }

    if isinstance(data_type, list):
        with pytest.raises(DataTypeNotImplementedError):
            data = get_measure(**kwargs)
    else:
        if NO_GBD_ACCESS:
            pytest.skip("Need GBD access to run this test")
        data = get_measure(**kwargs)

        check_data(data, data_type)


@pytest.mark.parametrize(
    "entity", [HIV_AIDS, HIV_AIDS_DRUG_SUSCEPTIBLE_TB_WO_ANEMIA], ids=("cause", "sequela")
)
@pytest.mark.parametrize(
    "data_type", ["means", "draws", ["means", "draws"]], ids=("means", "draws", "means_draws")
)
def test_get_incidence_rate_mocked(
    entity: Cause | Sequela,
    data_type: str | list[str],
    mocker: MockerFixture,
    request: pytest.FixtureRequest,
) -> None:
    """Test the mocked get_measure function.

    This mocks the vivarium_gbd_access calls with dummy data. This allows for
    pseudo-testing on github actions which does not have access to vivarium_gbd_access.
    """

    kwargs = {
        "entity": entity,
        "measure": "incidence_rate",
        "location": LOCATION,
        "years": YEAR,
        "data_type": data_type,
    }

    if isinstance(data_type, list):
        with pytest.raises(DataTypeNotImplementedError):
            data = get_measure(**kwargs)
    else:
        # Use the parameter ID combination to request the correct fixture data
        # to use as a mocked return
        parametrized_id = request.node.name.split(request.function.__name__)[-1]
        mocked_data = {
            "[means-cause]": request.getfixturevalue(
                "mocked_cause_incidence_rate_get_outputs"
            ),
            "[draws-cause]": request.getfixturevalue("mocked_cause_incidence_rate_get_draws"),
            "[means-sequela]": request.getfixturevalue(
                "mocked_sequela_incidence_rate_get_outputs"
            ),
            "[draws-sequela]": request.getfixturevalue(
                "mocked_sequela_incidence_rate_get_draws"
            ),
        }[parametrized_id]
        mock_extract_incidence_rate = mocker.patch(
            "vivarium_inputs.extract.extract_incidence_rate",
            return_value=mocked_data[mocked_data["measure_id"] == MEASURES["Incidence rate"]],
        )
        mock_extract_prevalence = mocker.patch(
            "vivarium_inputs.extract.extract_prevalence",
            return_value=mocked_data[mocked_data["measure_id"] == MEASURES["Prevalence"]],
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

        check_data(data, data_type)


####################
# HELPER FUNCTIONS #
####################


def check_data(data: pd.DataFrame, data_type: str) -> None:
    """Check the data returned."""
    assert not data.empty
    assert all(col in data.columns for col in SUPPORTED_DATA_TYPES[data_type])
    value_cols = SUPPORTED_DATA_TYPES[data_type]
    assert all(col in data.columns for col in value_cols)
    assert all(data.notna())
    assert all(data >= 0)
    # Check metadata index values (note that there may be other metadata returned)
    expected_metadata = {
        "location": {"India"},
        "sex": {"Male", "Female"},
        "age_start": set(MOCKED_AGE_GROUP_ENDPOINTS["age_start"]),
        "age_end": set(MOCKED_AGE_GROUP_ENDPOINTS["age_end"]),
        "year_start": {2021},
        "year_end": {2022},
    }
    for idx, expected in expected_metadata.items():
        assert set(data.index.get_level_values(idx)) == expected
