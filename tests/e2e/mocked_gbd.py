"""
Various mocked GBD objects for testing purposes.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from pytest_mock import MockerFixture

from vivarium_inputs import utility_data
from vivarium_inputs.globals import DRAW_COLUMNS, MEAN_COLUMNS, MEASURES

LOCATION = "India"
YEAR = 2021  # most recent year (used when default None is provided to get_measure())
DUMMY_INT = 1234
DUMMY_STR = "foo"
DUMMY_FLOAT = 123.4


def mock_vivarium_gbd_access(
    entity,
    measure: str,
    data_type: str | list[str],
    mocker: MockerFixture,
) -> list["mocker.Mock"]:
    """Mock calls as appropriate.

    For simplicity, we mock several smaller functions that may or may not be called
    by a specific test. However, for mocked 'utilities.extract' data, we only mock
    if it is required for a given entity-measure-data_type combination.

    Returns
    -------
    A list of the mocked extract functions.
    """

    # TODO [MIC-5454]: Move mocked gbd access stuff to vivarium_testing_utilities

    # Generic/small mocks that may or may not actually be called for this specific test
    mocker.patch(
        "vivarium_inputs.utility_data.get_estimation_years",
        return_value=get_mocked_estimation_years(),
    )
    age_groups = get_mocked_age_bins()
    # NOTE: the list of age group IDs must be sorted by their corresponding ages
    mocker.patch(
        "vivarium_inputs.utility_data.get_age_group_ids",
        return_value=list(age_groups.sort_values("age_group_years_start")["age_group_id"]),
    )
    mocker.patch(
        "vivarium_inputs.utility_data.get_location_path_to_global",
        return_value=get_mocked_location_path_to_global(),
    )
    mocker.patch(
        "vivarium_inputs.utility_data.get_raw_age_bins",
        return_value=age_groups,
    )
    mocker.patch(
        "vivarium_inputs.utility_data.get_raw_location_ids",
        return_value=get_mocked_location_ids(),
    )

    # Mock the relevant test extract function w/ dummy return_value data
    mocked_data_func = {
        "means": mocked_get_outputs,
        "draws": mocked_get_draws,
    }[data_type]

    entity_specific_metadata = {
        "Cause": {
            "cause_id": int(entity.gbd_id),
            "acause": DUMMY_STR,
            "cause_name": DUMMY_STR,
        },
        "Sequela": {
            "sequela_id": int(entity.gbd_id),
            "sequela_name": DUMMY_STR,
        },
    }[entity.__class__.__name__]

    mocked_funcs = []
    if measure == "incidence_rate":
        mocked_extract_incidence_rate = mocker.patch(
            "vivarium_inputs.extract.extract_incidence_rate",
            return_value=mocked_data_func(measure, **entity_specific_metadata),
        )
        mocked_extract_prevalence = mocker.patch(
            "vivarium_inputs.extract.extract_prevalence",
            return_value=mocked_data_func("prevalence", **entity_specific_metadata),
        )
        mocked_funcs = [mocked_extract_incidence_rate, mocked_extract_prevalence]
    elif measure == "prevalence":
        mocked_extract_prevalence = mocker.patch(
            "vivarium_inputs.extract.extract_prevalence",
            return_value=mocked_data_func(measure, **entity_specific_metadata),
        )
        mocked_funcs = [mocked_extract_prevalence]
    elif measure == "disability_weight":
        mocked_extract_disability_weight = mocker.patch(
            "vivarium_inputs.extract.extract_disability_weight",
            return_value=mocked_data_func(measure, **entity_specific_metadata),
        )
        mocked_funcs = [mocked_extract_disability_weight]
    elif measure == "remission_rate":
        mocked_extract_remission_rate = mocker.patch(
            "vivarium_inputs.extract.extract_remission_rate",
            return_value=mocked_data_func(measure, **entity_specific_metadata),
        )
        mocked_funcs = [mocked_extract_remission_rate]
    elif measure == "cause_specific_mortality_rate":
        del entity_specific_metadata["acause"]
        del entity_specific_metadata["cause_name"]
        mocked_extract_deaths = mocker.patch(
            "vivarium_inputs.extract.extract_deaths",
            return_value=mocked_data_func("deaths", **entity_specific_metadata),
        )
        mocked_extract_structure = mocker.patch(
            "vivarium_inputs.extract.extract_structure",
            return_value=mocked_data_func("structure"),
        )
        mocked_funcs = [mocked_extract_deaths, mocked_extract_structure]
    elif measure == "excess_mortality_rate":
        mocked_prevalence_data = mocked_data_func("prevalence", **entity_specific_metadata)
        mocked_extract_prevalence = mocker.patch(
            "vivarium_inputs.extract.extract_prevalence",
            return_value=mocked_prevalence_data[
                mocked_prevalence_data["measure_id"] == MEASURES["Prevalence"]
            ],
        )
        death_kwargs = entity_specific_metadata.copy()
        del death_kwargs["acause"]
        del death_kwargs["cause_name"]
        mocked_extract_deaths = mocker.patch(
            "vivarium_inputs.extract.extract_deaths",
            return_value=mocked_data_func("deaths", **entity_specific_metadata),
        )
        mocked_extract_structure = mocker.patch(
            "vivarium_inputs.extract.extract_structure",
            return_value=mocked_data_func("structure"),
        )
        mocked_funcs = [
            mocked_extract_prevalence,
            mocked_extract_deaths,
            mocked_extract_structure,
        ]
    else:
        raise NotImplementedError(f"Unexpected measure: {measure}")
    return mocked_funcs


#################
# VARIOUS MOCKS #
#################


def get_mocked_estimation_years() -> list[int]:
    """Mocked estimation years for testing."""
    return [1990, 1995, 2000, 2005, 2010, 2015, 2019, 2020, 2021, 2022]


def get_mocked_location_path_to_global() -> pd.DataFrame:
    """Mocked location path to global for testing."""
    return pd.read_csv("tests/fixture_data/location_path_to_global.csv")


def get_mocked_age_bins() -> pd.DataFrame:
    """Mocked age bins for testing."""
    return pd.read_csv("tests/fixture_data/age_bins.csv")


def get_mocked_location_ids() -> pd.DataFrame:
    """Mocked location ids for testing."""
    return pd.read_csv("tests/fixture_data/location_ids.csv")


#########################
# MOCKED GET_DRAWS DATA #
#########################


def mocked_get_draws(measure: str, **entity_specific_metadata) -> pd.DataFrame:
    """Mocked vivarium_gbd_access get_draws() data for testing."""

    # Get the common data for the specific measure (regardless of entity type)
    df = {
        "incidence_rate": get_mocked_data_incidence_rate_get_draws,
        "prevalence": get_mocked_data_prevalence_get_draws,
        "disability_weight": get_mocked_data_dw_get_draws,
        "remission_rate": get_mocked_data_remission_rate_get_draws,
        "deaths": get_mocked_data_deaths_get_draws,
        "structure": get_mocked_data_structure_get_draws,
    }[measure]()

    # Add on common metadata (note that this may overwrite existing columns, e.g.
    # from loading a population static file)
    df["location_id"] = utility_data.get_location_id(LOCATION)
    df["year_id"] = YEAR

    # Add on other entity-specific metadata columns
    for key, value in entity_specific_metadata.items():
        df[key] = value

    return df


def get_mocked_data_incidence_rate_get_draws() -> pd.DataFrame:
    age_group_ids = get_mocked_age_bins()["age_group_id"]
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids)),
        columns=["age_group_id", "sex_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 6
    df["metric_id"] = 3  # for rate
    df["version_id"] = DUMMY_INT

    _add_value_columns(df, DRAW_COLUMNS, 0.3, 3.6)

    return df


def get_mocked_data_prevalence_get_draws() -> pd.DataFrame:
    age_group_ids = get_mocked_age_bins()["age_group_id"]
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids)),
        columns=["age_group_id", "sex_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 5
    df["metric_id"] = 3  # for rate
    df["version_id"] = DUMMY_INT

    _add_value_columns(df, DRAW_COLUMNS, 0.004, 0.06)

    return df


def get_mocked_data_dw_get_draws() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "age_group_id": 22,
            "sex_id": 3,
            "measure": "disability_weight",
            "healthstate_id": DUMMY_FLOAT,
            "healthstate": DUMMY_STR,
        },
        index=[0],
    )

    _add_value_columns(df, DRAW_COLUMNS, 0.1, 0.7)

    return df


def get_mocked_data_remission_rate_get_draws() -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids)),
        columns=["age_group_id", "sex_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 7
    df["metric_id"] = 3
    df["model_version_id"] = DUMMY_INT
    df["modelable_entity_id"] = DUMMY_INT

    _add_value_columns(df, DRAW_COLUMNS, 50.0, 80.0)

    return df


def get_mocked_data_deaths_get_draws() -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids)),
        columns=["age_group_id", "sex_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 1
    df["metric_id"] = 1
    df["version_id"] = DUMMY_INT

    # Note: The maximum deaths cannot be too high because we have a validation
    # cap on CSMR and EMR values; we thus use an unrealistically small value here.
    _add_value_columns(df, DRAW_COLUMNS, 0.0, 10.0)

    return df


def get_mocked_data_structure_get_draws() -> pd.DataFrame:
    # Populations is difficult to mock at the age-group level so just load it
    return pd.read_csv(f"tests/fixture_data/population_{LOCATION.lower()}_{YEAR}.csv")


###########################
# MOCKED GET_OUTPUTS DATA #
###########################


def mocked_get_outputs(measure: str, **entity_specific_metadata) -> pd.DataFrame:
    """Mocked vivarium_gbd_access get_outputs() data for testing."""

    # Get the common data for the specific measure (regardless of entity type)
    df = {
        "incidence_rate": get_mocked_data_incidence_rate_get_outputs,
        "prevalence": get_mocked_data_prevalence_get_outputs,
    }[measure]()

    # Add on common metadata (note that this may overwrite existing columns, e.g.
    # from loading a population static file)
    df["location_id"] = utility_data.get_location_id(LOCATION)
    df["year_id"] = YEAR
    df["expected"] = False
    df["location_name"] = "India"
    df["location_type"] = "admin0"
    age_bins = get_mocked_age_bins()
    df["age_group_name"] = df["age_group_id"].map(
        dict(age_bins[["age_group_id", "age_group_name"]].values)
    )
    df["sex"] = df["sex_id"].map({1: "Male", 2: "Female"})

    # Add on other metadata columns
    for key, value in entity_specific_metadata.items():
        df[key] = value

    # get_outputs() also returns 'upper' and 'lower' columns
    df["upper"] = df["val"] * 1.1
    df["lower"] = df["val"] * 0.9

    return df


def get_mocked_data_incidence_rate_get_outputs() -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids)),
        columns=["age_group_id", "sex_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 6
    df["metric_id"] = 3
    df["measure"] = "incidence"
    df["measure_name"] = "Incidence"
    df["metric_name"] = "Rate"

    _add_value_columns(df, MEAN_COLUMNS, 1.4, 1.8)

    return df


def get_mocked_data_prevalence_get_outputs() -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids)),
        columns=["age_group_id", "sex_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 5
    df["metric_id"] = 3
    df["measure"] = "prevalence"
    df["measure_name"] = "Prevalence"
    df["metric_name"] = "Rate"

    _add_value_columns(df, MEAN_COLUMNS, 0.22, 0.027)

    return df


####################
# HELPER FUNCTIONS #
####################


def _add_value_columns(
    df: pd.DataFrame, value_columns: list[str], min: float, max: float
) -> None:
    np.random.seed(42)
    for col in value_columns:
        df[col] = np.random.uniform(min, max, size=len(df))
