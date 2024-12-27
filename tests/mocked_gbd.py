"""
Various mocked GBD objects for testing purposes.
"""

from __future__ import annotations

import itertools
from functools import partial

import numpy as np
import pandas as pd
from gbd_mapping import ModelableEntity
from pytest_mock import MockerFixture

from vivarium_inputs import utility_data
from vivarium_inputs.globals import DRAW_COLUMNS, MEAN_COLUMNS, MEASURES

MOST_RECENT_YEAR = 2021
DUMMY_INT = 1234
DUMMY_STR = "foo"
DUMMY_FLOAT = 123.4


def mock_vivarium_gbd_access(
    entity,
    measure: str,
    locations: str | int | list[str | int],
    years: int | list[int] | str | None,
    data_type: str | list[str],
    mocker: MockerFixture,
) -> list["mocker.Mock"]:
    """Mock calls as appropriate.

    For simplicity, we mock several smaller functions that may or may not be called
    by a specific test. However, for mocked 'utilities.extract' data, we only mock
    if it is required for a given entity-measure-data_type combination.

    Notes
    -----
    For simplicity, we mock the 'utilities.extract` functions rather than the
    actual gbd calls. This does leave a bit of a coverage gap since the
    'utilities.extract' functions have a bit of logic in them once the raw
    data from gbd is returned. This seems like a reasonable tradeoff given that
    we are still running the full unmocked tests on Jenkins as well.

    Returns
    -------
    A list of the mocked extract functions.
    """

    # TODO [MIC-5454]: Move mocked gbd access stuff to vivarium_testing_utilities
    # TODO [MIC-5461]: Speed up mocked data generation

    # Generic/small mocks that may or may not actually be called for this specific test
    mocker.patch(
        "vivarium_inputs.utility_data.get_most_recent_year", return_value=MOST_RECENT_YEAR
    )
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

    gbd_id = int(entity.gbd_id) if entity.gbd_id else None
    entity_specific_metadata_mapper = {
        "Cause": {
            "cause_id": gbd_id,
            "acause": DUMMY_STR,
            "cause_name": DUMMY_STR,
        },
        "Sequela": {
            "sequela_id": gbd_id,
            "sequela_name": DUMMY_STR,
        },
        "RiskFactor": {
            "rei_id": gbd_id,
        },
        "Covariate": {
            "covariate_id": DUMMY_INT,
            "covariate_name_short": DUMMY_STR,
        },
    }

    entity_specific_metadata = entity_specific_metadata_mapper.get(
        entity.__class__.__name__, {}
    )

    # Convert years and locations to lists of IDs
    if not years:
        year_ids = [MOST_RECENT_YEAR]
    elif years == "all":
        estimation_years = get_mocked_estimation_years()
        year_ids = list(range(min(estimation_years), max(estimation_years) + 1))
    elif not isinstance(years, list):
        year_ids = [int(years)]
    else:
        year_ids = years

    location_ids = locations if isinstance(locations, list) else [locations]
    location_ids = [
        utility_data.get_location_id(loc) if isinstance(loc, str) else loc
        for loc in location_ids
    ]

    if measure in ["incidence_rate", "raw_incidence_rate"]:
        mocked_extract_incidence_rate = mocker.patch(
            "vivarium_inputs.extract.extract_incidence_rate",
            return_value=mocked_data_func(
                measure, entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_extract_prevalence = mocker.patch(
            "vivarium_inputs.extract.extract_prevalence",
            return_value=mocked_data_func(
                "prevalence", entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_funcs = [mocked_extract_incidence_rate, mocked_extract_prevalence]
    elif measure == "prevalence":
        mock = mocker.patch(
            "vivarium_inputs.extract.extract_prevalence",
            return_value=mocked_data_func(
                measure, entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_funcs = [mock]
    elif measure == "disability_weight":
        mock = mocker.patch(
            "vivarium_inputs.extract.extract_disability_weight",
            return_value=mocked_data_func(
                measure, entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        combined_metadata = entity_specific_metadata_mapper["Cause"]
        combined_metadata.update(entity_specific_metadata_mapper["Sequela"])
        mocked_extract_prevalence = mocker.patch(
            "vivarium_inputs.extract.extract_prevalence",
            return_value=mocked_data_func(
                "prevalence", entity, location_ids, year_ids, **combined_metadata
            ),
        )
        mocked_funcs = [mock]
    elif measure == "remission_rate":
        mock = mocker.patch(
            "vivarium_inputs.extract.extract_remission_rate",
            return_value=mocked_data_func(
                measure, entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_funcs = [mock]
    elif measure == "cause_specific_mortality_rate":
        del entity_specific_metadata["acause"]
        del entity_specific_metadata["cause_name"]
        mocked_extract_deaths = mocker.patch(
            "vivarium_inputs.extract.extract_deaths",
            return_value=mocked_data_func(
                "deaths", entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_extract_structure = mocker.patch(
            "vivarium_inputs.extract.extract_structure",
            return_value=mocked_data_func("structure", entity, location_ids, year_ids),
        )
        mocked_funcs = [mocked_extract_deaths, mocked_extract_structure]
    elif measure == "excess_mortality_rate":
        mocked_prevalence_data = mocked_data_func(
            "prevalence", entity, location_ids, year_ids, **entity_specific_metadata
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
            return_value=mocked_data_func(
                "deaths", entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_extract_structure = mocker.patch(
            "vivarium_inputs.extract.extract_structure",
            return_value=mocked_data_func("structure", entity, location_ids, year_ids),
        )
        mocked_funcs = [
            mocked_extract_prevalence,
            mocked_extract_deaths,
            mocked_extract_structure,
        ]
    elif measure == "exposure":
        mock = mocker.patch(
            "vivarium_inputs.extract.extract_exposure",
            return_value=mocked_data_func(
                measure, entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_funcs = [mock]
    elif measure == "exposure_standard_deviation":
        mocked_exposure_sd = mocker.patch(
            "vivarium_inputs.extract.extract_exposure_standard_deviation",
            return_value=mocked_data_func(
                measure, entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_exposure = mocker.patch(
            "vivarium_inputs.extract.extract_exposure",
            return_value=mocked_data_func(
                "exposure", entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_funcs = [mocked_exposure_sd, mocked_exposure]
    elif measure == "exposure_distribution_weights":
        mocked_exposure_distribution_weights = mocker.patch(
            "vivarium_inputs.extract.extract_exposure_distribution_weights",
            return_value=mocked_data_func(
                measure, entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_exposure = mocker.patch(
            "vivarium_inputs.extract.extract_exposure",
            return_value=mocked_data_func(
                "exposure", entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_funcs = [mocked_exposure_distribution_weights, mocked_exposure]
    elif measure == "relative_risk":
        mocked_rr = mocker.patch(
            "vivarium_inputs.extract.extract_relative_risk",
            return_value=mocked_data_func(
                "relative_risk", entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_exposure = mocker.patch(
            "vivarium_inputs.extract.extract_exposure",
            return_value=mocked_data_func(
                "exposure", entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_funcs = [mocked_rr, mocked_exposure]
    elif measure == "population_attributable_fraction":
        mocked_pafs = mocker.patch(
            "vivarium_inputs.extract.extract_population_attributable_fraction",
            return_value=mocked_data_func(
                measure, entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_exposure = mocker.patch(
            "vivarium_inputs.extract.extract_exposure",
            return_value=mocked_data_func(
                "exposure", entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_rr = mocker.patch(
            "vivarium_inputs.extract.extract_relative_risk",
            return_value=mocked_data_func(
                "relative_risk", entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_funcs = [mocked_pafs, mocked_exposure, mocked_rr]
    elif measure == "estimate":
        mock = mocker.patch(
            "vivarium_inputs.extract.extract_estimate",
            return_value=mocked_data_func(
                measure, entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_funcs = [mock]
    elif measure == "structure":
        mock = mocker.patch(
            "vivarium_inputs.extract.extract_structure",
            return_value=mocked_data_func(
                "structure", entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_funcs = [mock]
    elif measure == "demographic_dimensions":
        mocked_funcs = []
    elif measure == "deaths":
        mocked_extract_deaths = mocker.patch(
            "vivarium_inputs.extract.extract_deaths",
            return_value=mocked_data_func(
                "deaths", entity, location_ids, year_ids, **entity_specific_metadata
            ),
        )
        mocked_extract_structure = mocker.patch(
            "vivarium_inputs.extract.extract_structure",
            return_value=mocked_data_func("structure", entity, location_ids, year_ids),
        )
        mocked_funcs = [mocked_extract_deaths, mocked_extract_structure]
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


def mocked_get_draws(
    measure: str,
    entity: ModelableEntity,
    locations: list[int],
    years: list[int],
    **entity_specific_metadata,
) -> pd.DataFrame:
    """Mocked vivarium_gbd_access get_draws() data for testing."""

    # Get the common data for the specific measure (regardless of entity type)
    df = {
        "incidence_rate": get_mocked_incidence_rate_get_draws,
        "raw_incidence_rate": get_mocked_incidence_rate_get_draws,  # mock same as not-raw
        "prevalence": get_mocked_prevalence_get_draws,
        "disability_weight": get_mocked_dw_get_draws,
        "remission_rate": get_mocked_remission_rate_get_draws,
        "deaths": get_mocked_deaths_get_draws,
        "structure": get_mocked_structure_get_draws,
        "exposure": partial(get_mocked_exposure_get_draws, entity),
        "exposure_standard_deviation": get_mocked_exposure_sd_get_draws,
        "exposure_distribution_weights": get_mocked_exposure_distribution_weights_get_draws,
        "population_attributable_fraction": get_mocked_pafs_get_draws,
        "relative_risk": partial(get_mocked_rr_get_draws, entity),
        "estimate": get_mocked_estimate_get_draws,
    }[measure](locations, years)

    # Add on entity-specific metadata columns
    for key, value in entity_specific_metadata.items():
        df[key] = value

    return df


def get_mocked_incidence_rate_get_draws(
    locations: list[int], years: list[int]
) -> pd.DataFrame:
    age_group_ids = get_mocked_age_bins()["age_group_id"]
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, locations, years)),
        columns=["age_group_id", "sex_id", "location_id", "year_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 6  # incidence
    df["metric_id"] = 3  # rate
    df["version_id"] = DUMMY_INT

    _add_value_columns(df, DRAW_COLUMNS, 0.3, 3.6)

    return df


def get_mocked_prevalence_get_draws(locations: list[int], years: list[int]) -> pd.DataFrame:
    age_group_ids = get_mocked_age_bins()["age_group_id"]
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, locations, years)),
        columns=["age_group_id", "sex_id", "location_id", "year_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 5  # prevalence
    df["metric_id"] = 3  # rate
    df["version_id"] = DUMMY_INT

    _add_value_columns(df, DRAW_COLUMNS, 0.004, 0.06)

    return df


def get_mocked_dw_get_draws(locations: list[int], years: list[int]) -> pd.DataFrame:
    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(locations, years)),
        columns=["location_id", "year_id"],
    )

    # Add on other metadata columns
    df["age_group_id"] = 22
    df["sex_id"] = 3
    df["measure"] = "disability_weight"
    df["healthstate_id"] = DUMMY_FLOAT
    df["healthstate"] = DUMMY_STR

    # We set the values here very low to avoid validation errors
    _add_value_columns(df, DRAW_COLUMNS, 0.0, 0.1)

    return df


def get_mocked_remission_rate_get_draws(
    locations: list[int], years: list[int]
) -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2]
    # HACK: Remission rates are binned by year. If `years`` is a full continuous range of
    # the estimation years, then likely the user requested data for "all" years.
    # If this is the case, we set years to the estimation years.
    estimation_years = get_mocked_estimation_years()
    if years == list(range(min(estimation_years), max(estimation_years) + 1)):
        years = estimation_years

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, locations, years)),
        columns=["age_group_id", "sex_id", "location_id", "year_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 7  # remission
    df["metric_id"] = 3  # rate
    df["model_version_id"] = DUMMY_INT
    df["modelable_entity_id"] = DUMMY_INT

    _add_value_columns(df, DRAW_COLUMNS, 50.0, 80.0)

    return df


def get_mocked_deaths_get_draws(locations: list[int], years: list[int]) -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, locations, years)),
        columns=["age_group_id", "sex_id", "location_id", "year_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 1  # deaths
    df["metric_id"] = 1  # number
    df["version_id"] = DUMMY_INT

    # Note: The maximum deaths cannot be too high because we have a validation
    # cap on CSMR and EMR values; we thus use an unrealistically small value here.
    _add_value_columns(df, DRAW_COLUMNS, 0.0, 10.0)

    return df


def get_mocked_structure_get_draws(locations: list[int], years: list[int]) -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2, 3]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, locations, years)),
        columns=["age_group_id", "sex_id", "location_id", "year_id"],
    )

    # Add on other metadata columns
    df["run_id"] = DUMMY_INT

    _add_value_columns(df, ["population"], 1.0e6, 100.0e6)

    return df


def get_mocked_exposure_get_draws(
    entity, locations: list[int], years: list[int]
) -> pd.DataFrame:
    if entity.name == "low_birth_weight_and_short_gestation":
        age_group_ids = [2, 3]
        sex_ids = [1, 2]
        parameters = list(entity.categories.to_dict())
        # Initiate df with all possible combinations of variable metadata columns
        df = pd.DataFrame(
            list(itertools.product(age_group_ids, sex_ids, parameters, locations, years)),
            columns=["age_group_id", "sex_id", "parameter", "location_id", "year_id"],
        )
        # Add on other metadata columns
        df["modelable_entity_id"] = DUMMY_FLOAT  # b/c nans come in
        _add_value_columns(df, DRAW_COLUMNS, 0.0, 1.0)
    elif entity.name == "high_systolic_blood_pressure":
        age_bins = get_mocked_age_bins()
        age_group_ids = list(age_bins["age_group_id"])
        sex_ids = [1, 2]
        # Initiate df with all possible combinations of variable metadata columns
        df = pd.DataFrame(
            list(itertools.product(age_group_ids, sex_ids, locations, years)),
            columns=["age_group_id", "sex_id", "location_id", "year_id"],
        )
        # Add on other metadata columns
        df["modelable_entity_id"] = DUMMY_INT
        df["parameter"] = "continuous"
        _add_value_columns(df, DRAW_COLUMNS, 100.0, 200.0)
    else:
        raise NotImplementedError(f"{entity.name} not implemented in mocked_gbd.py")

    df["measure_id"] = 19  # continuous
    df["metric_id"] = 3  # rate

    return df


def get_mocked_exposure_sd_get_draws(locations: list[int], years: list[int]) -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, locations, years)),
        columns=["age_group_id", "sex_id", "location_id", "year_id"],
    )

    # Add on other metadata columns
    df["modelable_entity_id"] = DUMMY_INT
    df["measure_id"] = 19  # continuous
    df["metric_id"] = 3  # rate
    df["model_version_id"] = DUMMY_INT

    _add_value_columns(df, DRAW_COLUMNS, 6.0, 14.0)

    return df


def get_mocked_exposure_distribution_weights_get_draws(
    locations: list[int], years: list[int]
) -> pd.DataFrame:

    # We simply copy/paste the data from the call here (year 2021, location 163)
    return pd.DataFrame(
        {
            "exp": 0.0012511270939698,
            "gamma": 0.0118578481457336,
            "invgamma": 0.0307299056293708,
            "llogis": 0.244326256508881,
            "gumbel": 0.515477963465818,
            "weibull": 0.0085853963221991,
            "lnorm": 0.0162367872410608,
            "norm": 0.00494120859787,
            "glnorm": 0,
            "betasr": 0.0350911796340715,
            "mgamma": 0.0064136247118684,
            "mgumbel": 0.125088702649157,
            "invweibull": 0,
            "rei_id": 107,
            "location_id": 163,
            "sex_id": 3,
            "age_group_id": 22,
            "measure": "ensemble_distribution_weight",
        },
        index=[0],
    )


def get_mocked_pafs_get_draws(locations: list[int], years: list[int]) -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2]
    measure_ids = [3, 4]  # only allowables for PAFs

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, measure_ids, locations, years)),
        columns=["age_group_id", "sex_id", "measure_id", "location_id", "year_id"],
    )

    # Add on other metadata columns
    df["cause_id"] = 495  # Needs to be a valid cause_id
    df["metric_id"] = 2  # percent
    df["version_id"] = DUMMY_INT

    _add_value_columns(df, DRAW_COLUMNS, 0.0, 1.0)

    return df


def get_mocked_rr_get_draws(entity, locations: list[int], years: list[int]) -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    if entity.name == "high_systolic_blood_pressure":
        # high sbp is only for >=25 years
        age_bins = age_bins[age_bins["age_group_years_start"] >= 25]
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2]
    # HACK: Relative risks are binned by year. If `years`` is a full continuous range of
    # the estimation years, then likely the user requested data for "all" years.
    # If this is the case, we set years to the estimation years.
    estimation_years = get_mocked_estimation_years()
    if years == list(range(min(estimation_years), max(estimation_years) + 1)):
        years = estimation_years

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, years)),
        columns=["age_group_id", "sex_id", "year_id"],
    )

    # Add on other metadata columns
    df["location_id"] = 1  # Most relative risks are global
    df["modelable_entity_id"] = DUMMY_INT
    df["cause_id"] = 495  # Needs to be a valid cause_id
    df["mortality"] = 1
    df["morbidity"] = 1
    df["metric_id"] = 3  # rate
    df["parameter"] = DUMMY_FLOAT
    df["exposure"] = np.nan

    _add_value_columns(df, DRAW_COLUMNS, 0.0, 1000.0)

    return df


def get_mocked_estimate_get_draws(locations: list[int], years: list[int]) -> pd.DataFrame:
    age_group_ids = [27]
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, locations, years)),
        columns=["age_group_id", "sex_id", "location_id", "year_id"],
    )

    # Add on other metadata columns
    location_ids = (
        get_mocked_location_ids()[["location_id", "location_name"]]
        .set_index("location_id")
        .squeeze()
    )
    df["location_name"] = df["location_id"].map(location_ids)
    df["covariate_id"] = DUMMY_INT
    df["model_version_id"] = DUMMY_INT
    df["age_group_name"] = DUMMY_STR
    df["sex"] = DUMMY_STR

    # Estimates don't play by the rules
    df["mean_value"] = DUMMY_FLOAT
    df["lower_value"] = 0.9 * df["mean_value"]
    df["upper_value"] = 1.1 * df["mean_value"]

    return df


###########################
# MOCKED GET_OUTPUTS DATA #
###########################


def mocked_get_outputs(
    measure: str,
    entity: ModelableEntity,
    locations: list[int],
    years: list[int],
    **entity_specific_metadata,
) -> pd.DataFrame:
    """Mocked vivarium_gbd_access get_outputs() data for testing."""

    # Get the common data for the specific measure (regardless of entity type)
    df = {
        "incidence_rate": get_mocked_incidence_rate_get_outputs,
        "raw_incidence_rate": get_mocked_incidence_rate_get_outputs,  # Same as non-raw
        "prevalence": get_mocked_prevalence_get_outputs,
    }[measure](locations, years)

    # Add on common metadata (note that this may overwrite existing columns, e.g.
    # from loading a population static file)
    df["expected"] = False
    location_ids = (
        get_mocked_location_ids()[["location_id", "location_name"]]
        .set_index("location_id")
        .squeeze()
    )
    df["location_name"] = df["location_id"].map(location_ids)
    df["location_type"] = "admin0"  # brittle
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


def get_mocked_incidence_rate_get_outputs(
    locations: list[int], years: list[int]
) -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, locations, years)),
        columns=["age_group_id", "sex_id", "location_id", "year_id"],
    )

    # Add on other metadata columns
    df["measure_id"] = 6
    df["metric_id"] = 3
    df["measure"] = "incidence"
    df["measure_name"] = "Incidence"
    df["metric_name"] = "Rate"

    _add_value_columns(df, MEAN_COLUMNS, 1.4, 1.8)

    return df


def get_mocked_prevalence_get_outputs(locations: list[int], years: list[int]) -> pd.DataFrame:
    age_bins = get_mocked_age_bins()
    age_group_ids = list(age_bins["age_group_id"])
    sex_ids = [1, 2]

    # Initiate df with all possible combinations of variable metadata columns
    df = pd.DataFrame(
        list(itertools.product(age_group_ids, sex_ids, locations, years)),
        columns=["age_group_id", "sex_id", "location_id", "year_id"],
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
