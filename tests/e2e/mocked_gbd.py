"""
Various mocked GBD objects for testing purposes.
"""

from __future__ import annotations

import itertools
from functools import partial

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
    # TODO [MIC-5461]: Speed up mocked data generation

    # Generic/small mocks that may or may not actually be called for this specific test
    mocker.patch("vivarium_inputs.utility_data.get_most_recent_year", return_value=YEAR)
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

    entity_specific_metadata_mapper = {
        "Cause": {
            "cause_id": int(entity.gbd_id),
            "acause": DUMMY_STR,
            "cause_name": DUMMY_STR,
        },
        "Sequela": {
            "sequela_id": int(entity.gbd_id),
            "sequela_name": DUMMY_STR,
        },
        "RiskFactor": {
            "rei_id": int(entity.gbd_id),
        },
    }
    entity_specific_metadata = entity_specific_metadata_mapper[entity.__class__.__name__]

    mocked_funcs = []
    if measure == "incidence_rate":
        mocked_extract_incidence_rate = mocker.patch(
            "vivarium_inputs.extract.extract_incidence_rate",
            return_value=mocked_data_func(measure, entity, **entity_specific_metadata),
        )
        mocked_extract_prevalence = mocker.patch(
            "vivarium_inputs.extract.extract_prevalence",
            return_value=mocked_data_func("prevalence", entity, **entity_specific_metadata),
        )
        mocked_funcs = [mocked_extract_incidence_rate, mocked_extract_prevalence]
    elif measure == "prevalence":
        mock = mocker.patch(
            "vivarium_inputs.extract.extract_prevalence",
            return_value=mocked_data_func(measure, entity, **entity_specific_metadata),
        )
        mocked_funcs = [mock]
    elif measure == "disability_weight":
        mock = mocker.patch(
            "vivarium_inputs.extract.extract_disability_weight",
            return_value=mocked_data_func(measure, entity, **entity_specific_metadata),
        )
        combined_metadata = entity_specific_metadata_mapper["Cause"]
        combined_metadata.update(entity_specific_metadata_mapper["Sequela"])
        mocked_extract_prevalence = mocker.patch(
            "vivarium_inputs.extract.extract_prevalence",
            return_value=mocked_data_func("prevalence", entity, **combined_metadata),
        )
        mocked_funcs = [mock]
    elif measure == "remission_rate":
        mock = mocker.patch(
            "vivarium_inputs.extract.extract_remission_rate",
            return_value=mocked_data_func(measure, entity, **entity_specific_metadata),
        )
        mocked_funcs = [mock]
    elif measure == "cause_specific_mortality_rate":
        del entity_specific_metadata["acause"]
        del entity_specific_metadata["cause_name"]
        mocked_extract_deaths = mocker.patch(
            "vivarium_inputs.extract.extract_deaths",
            return_value=mocked_data_func("deaths", entity, **entity_specific_metadata),
        )
        mocked_extract_structure = mocker.patch(
            "vivarium_inputs.extract.extract_structure",
            return_value=mocked_data_func("structure", entity),
        )
        mocked_funcs = [mocked_extract_deaths, mocked_extract_structure]
    elif measure == "excess_mortality_rate":
        mocked_prevalence_data = mocked_data_func(
            "prevalence", entity, **entity_specific_metadata
        )
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
            return_value=mocked_data_func("deaths", entity, **entity_specific_metadata),
        )
        mocked_extract_structure = mocker.patch(
            "vivarium_inputs.extract.extract_structure",
            return_value=mocked_data_func("structure", entity),
        )
        mocked_funcs = [
            mocked_extract_prevalence,
            mocked_extract_deaths,
            mocked_extract_structure,
        ]
    elif measure == "exposure":
        mock = mocker.patch(
            "vivarium_inputs.extract.extract_exposure",
            return_value=mocked_data_func(measure, entity, **entity_specific_metadata),
        )
        mocked_funcs = [mock]
    elif measure == "exposure_standard_deviation":
        mock = mocker.patch(
            "vivarium_inputs.extract.extract_exposure_standard_deviation",
            return_value=mocked_data_func(measure, entity, **entity_specific_metadata),
        )
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


def mocked_get_draws(measure: str, entity, **entity_specific_metadata) -> pd.DataFrame:
    """Mocked vivarium_gbd_access get_draws() data for testing."""

    # Get the common data for the specific measure (regardless of entity type)
    df = {
        "incidence_rate": get_mocked_incidence_rate_get_draws,
        "prevalence": get_mocked_prevalence_get_draws,
        "disability_weight": get_mocked_dw_get_draws,
        "remission_rate": get_mocked_remission_rate_get_draws,
        "deaths": get_mocked_deaths_get_draws,
        "structure": get_mocked_structure_get_draws,
        "exposure": partial(get_mocked_exposure_get_draws, entity),
        "exposure_standard_deviation": get_mocked_exposure_sd_get_draws,
    }[measure]()
    # Add on common metadata (note that this may overwrite existing columns, e.g.
    # from loading a population static file)
    df["location_id"] = utility_data.get_location_id(LOCATION)
    df["year_id"] = YEAR

    # Add on other entity-specific metadata columns
    for key, value in entity_specific_metadata.items():
        df[key] = value

    return df


def get_mocked_incidence_rate_get_draws() -> pd.DataFrame:
    age_group_ids = get_mocked_age_bins()["age_group_id"]
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids)),
        columns=["age_group_id", "sex_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 6  # incidence
    df["metric_id"] = 3  # rate
    df["version_id"] = DUMMY_INT

    _add_value_columns(df, DRAW_COLUMNS, 0.3, 3.6)

    return df


def get_mocked_prevalence_get_draws() -> pd.DataFrame:
    age_group_ids = get_mocked_age_bins()["age_group_id"]
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids)),
        columns=["age_group_id", "sex_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 5  # prevalence
    df["metric_id"] = 3  # rate
    df["version_id"] = DUMMY_INT

    _add_value_columns(df, DRAW_COLUMNS, 0.004, 0.06)

    return df


def get_mocked_dw_get_draws() -> pd.DataFrame:
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
    # We set the values here very low to avoid validation errors
    _add_value_columns(df, DRAW_COLUMNS, 0.0, 0.1)

    return df


def get_mocked_remission_rate_get_draws() -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids)),
        columns=["age_group_id", "sex_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 7  # remission
    df["metric_id"] = 3  # rate
    df["model_version_id"] = DUMMY_INT
    df["modelable_entity_id"] = DUMMY_INT

    _add_value_columns(df, DRAW_COLUMNS, 50.0, 80.0)

    return df


def get_mocked_deaths_get_draws() -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids)),
        columns=["age_group_id", "sex_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 1  # deaths
    df["metric_id"] = 1  # number
    df["version_id"] = DUMMY_INT

    # Note: The maximum deaths cannot be too high because we have a validation
    # cap on CSMR and EMR values; we thus use an unrealistically small value here.
    _add_value_columns(df, DRAW_COLUMNS, 0.0, 10.0)

    return df


def get_mocked_structure_get_draws() -> pd.DataFrame:
    # Populations is difficult to mock at the age-group level so just load it
    # return pd.read_csv(f"tests/fixture_data/population_{LOCATION.lower()}_{YEAR}.csv")
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2, 3]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids)),
        columns=["age_group_id", "sex_id"],
    )

    # Add on other metadata columns
    df["run_id"] = DUMMY_INT

    _add_value_columns(df, ["population"], 1.0e6, 100.0e6)

    return df


def get_mocked_exposure_get_draws(entity) -> pd.DataFrame:
    if entity.name == "low_birth_weight_and_short_gestation":
        df = pd.read_csv("tests/fixture_data/lbwsg_exposure_metadata.csv")
        _add_value_columns(df, DRAW_COLUMNS, 0.0, 1.0)
        #
    else:
        age_bins = get_mocked_age_bins()
        age_group_ids = list(age_bins["age_group_id"])
        sex_ids = [1, 2]

        # Initiate df with all possible combinations of variable metadata columns
        df = pd.DataFrame(
            list(itertools.product(age_group_ids, sex_ids)),
            columns=["age_group_id", "sex_id"],
        )

        # Add on other metadata columns
        df["modelable_entity_id"] = DUMMY_INT
        df["measure_id"] = 19  # continuous
        df["metric_id"] = 3  # rate
        df["parameter"] = "continuous"

        _add_value_columns(df, DRAW_COLUMNS, 100.0, 200.0)

    return df


def get_mocked_exposure_sd_get_draws() -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids)),
        columns=["age_group_id", "sex_id"],
    )

    # Add on other metadata columns
    df["modelable_entity_id"] = DUMMY_INT
    df["measure_id"] = 19  # continuous
    df["metric_id"] = 3  # rate
    df["model_version_id"] = DUMMY_INT

    _add_value_columns(df, DRAW_COLUMNS, 6.0, 14.0)

    return df


###########################
# MOCKED GET_OUTPUTS DATA #
###########################


def mocked_get_outputs(measure: str, entity, **entity_specific_metadata) -> pd.DataFrame:
    """Mocked vivarium_gbd_access get_outputs() data for testing."""

    # Get the common data for the specific measure (regardless of entity type)
    df = {
        "incidence_rate": get_mocked_incidence_rate_get_outputs,
        "prevalence": get_mocked_prevalence_get_outputs,
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


def get_mocked_incidence_rate_get_outputs() -> pd.DataFrame:
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


def get_mocked_prevalence_get_outputs() -> pd.DataFrame:
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
