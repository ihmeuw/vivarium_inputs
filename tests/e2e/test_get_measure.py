"""
End-to-end tests for interface.get_measure().

Due to the fact that pulling data can take quite some time as well as that
vivarium_gbd_access is required but not always available, some unique
precautions and guards have been taken:
- All tests are automatically skipped if vivarium_gbd_access is not installed 
    (e.g. from github actions).
- Tests are parametrized by a 'mock_gbd' boolean which, if true, uses mocked
    data instead of pulling from the GBD. This allows for testing on github actions
    as well as testing on every push/pull (via Jenkins pipelines).
- Unmocked tests are marked as slow and thus require the --runslow flag to run.
- Particularly slow tests are datetime-checked to only weekly.
"""

from __future__ import annotations

from functools import partial

import pandas as pd
import pytest
from gbd_mapping import Cause, Sequela, causes, covariates, risk_factors, sequelae
from layered_config_tree import LayeredConfigTree
from pytest_mock import MockerFixture

from tests.conftest import NO_GBD_ACCESS
from tests.e2e.mocked_gbd import (
    LOCATION,
    YEAR,
    get_mocked_age_bins,
    mock_vivarium_gbd_access,
)
from vivarium_inputs import utility_data
from vivarium_inputs.globals import SUPPORTED_DATA_TYPES, DataAbnormalError
from vivarium_inputs.interface import get_measure
from vivarium_inputs.utilities import DataTypeNotImplementedError


# TODO [MIC-5448]: Move to vivarium_testing_utilties
@pytest.fixture(autouse=True)
def no_cache(mocker: MockerFixture) -> None:
    """Mock out the cache so that we always pull data."""

    if not NO_GBD_ACCESS:
        mocker.patch(
            "vivarium_gbd_access.utilities.get_input_config",
            return_value=LayeredConfigTree({"input_data": {"cache_data": False}}),
        )


CAUSES = [
    # (entity, applicable_measures)
    # NOTE: 'raw_incidence_rate' and 'deaths' should not be called directly from `get_measure()`
    (
        causes.measles,
        [
            "incidence_rate",
            "prevalence",
            "disability_weight",
            "cause_specific_mortality_rate",
            "excess_mortality_rate",
        ],
    ),
    (
        causes.diarrheal_diseases,
        [
            "incidence_rate",
            "prevalence",
            "disability_weight",
            "remission_rate",
            "cause_specific_mortality_rate",
            "excess_mortality_rate",
        ],
    ),
    (
        causes.diabetes_mellitus_type_2,
        [
            "incidence_rate",
            "prevalence",
            "disability_weight",
            "cause_specific_mortality_rate",
            "excess_mortality_rate",
        ],
    ),
]
CAUSE_MEASURES = [
    "incidence_rate",
    "prevalence",
    "birth_prevalence",
    "disability_weight",
    "remission_rate",
    "cause_specific_mortality_rate",
    "excess_mortality_rate",
]


@pytest.mark.parametrize("entity_details", CAUSES, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", CAUSE_MEASURES, ids=lambda x: x)
@pytest.mark.parametrize(
    "data_type", ["means", "draws", ["means", "draws"]], ids=("means", "draws", "means_draws")
)
@pytest.mark.parametrize("mock_gbd", [True, False], ids=("mocked", "unmocked"))
def test_get_measure_causelike(
    entity_details: tuple[Cause, list[str]],
    measure: str,
    data_type: str | list[str],
    mock_gbd: bool,
    runslow: bool,
    mocker: MockerFixture,
):
    # Test-specific fixme skips
    if measure == "birth_prevalence":
        pytest.skip("FIXME: need to find causes with birth prevalence")

    # Handle not implemented
    is_unimplemented_means = data_type == "means" and measure in [
        "disability_weight",
        "remission_rate",
        "cause_specific_mortality_rate",
        "excess_mortality_rate",
    ]
    is_unimplemented = isinstance(data_type, list) or is_unimplemented_means

    run_test(entity_details, measure, data_type, mock_gbd, runslow, mocker, is_unimplemented)


SEQUELAE = [
    (
        sequelae.hiv_aids_drug_susceptible_tuberculosis_without_anemia,
        [
            "incidence_rate",
            "prevalence",
            "birth_prevalence",
            "disability_weight",
        ],
    ),
]
SEQUELA_MEASURES = [
    "incidence_rate",
    "prevalence",
    "birth_prevalence",
    "disability_weight",
]


@pytest.mark.slow
@pytest.mark.parametrize("entity_details", SEQUELAE, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", SEQUELA_MEASURES, ids=lambda x: x)
@pytest.mark.parametrize(
    "data_type", ["means", "draws", ["means", "draws"]], ids=("means", "draws", "means_draws")
)
@pytest.mark.parametrize("mock_gbd", [True, False], ids=("mocked", "unmocked"))
def test_get_measure_sequelalike(
    entity_details: tuple[Sequela, list[str]],
    measure: str,
    data_type: str | list[str],
    mock_gbd: bool,
    runslow: bool,
    mocker: MockerFixture,
):

    # Test-specific fixme skips
    if measure == "birth_prevalence":
        pytest.skip("FIXME: need to find sequelae with birth prevalence")

    # Handle not implemented
    is_unimplemented_means = data_type == "means" and measure in [
        "disability_weight",
    ]
    is_unimplemented = isinstance(data_type, list) or is_unimplemented_means

    run_test(entity_details, measure, data_type, mock_gbd, runslow, mocker, is_unimplemented)


ENTITIES_R = [
    (
        risk_factors.high_systolic_blood_pressure,
        [
            "exposure",
            "exposure_standard_deviation",
            "exposure_distribution_weights",
            # "relative_risk",  # TODO: Add back in once Mic-4936 is resolved
            "population_attributable_fraction",
        ],
    ),
    (
        risk_factors.low_birth_weight_and_short_gestation,
        [
            "exposure",
            "relative_risk",
            "population_attributable_fraction",
        ],
    ),
]
MEASURES_R = [
    "exposure",
    "exposure_standard_deviation",
    "exposure_distribution_weights",
    # "relative_risk",  # TODO: Add back in once Mic-4936 is resolved
    "population_attributable_fraction",
]


@pytest.mark.skip("TODO: [mic-5456]")
@pytest.mark.parametrize("entity_details", ENTITIES_R, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_R, ids=lambda x: x[0])
def test_get_measure_risklike(entity_details, measure):
    entity, entity_expected_measures = entity_details
    tester = success_expected if measure in entity_expected_measures else fail_expected
    _df = tester(entity, measure, utility_data.get_location_id(LOCATION))


ENTITIES_COV = [
    covariates.systolic_blood_pressure_mmhg,
]
MEASURES_COV = ["estimate"]


@pytest.mark.skip("TODO: [mic-5456]")
@pytest.mark.parametrize("entity", ENTITIES_COV, ids=lambda x: x.name)
@pytest.mark.parametrize("measure", MEASURES_COV, ids=lambda x: x)
def test_get_measure_covariatelike(entity, measure):
    _df = get_measure(entity, measure, utility_data.get_location_id(LOCATION))


# TODO: Remove with Mic-4936
ENTITIES_R = [
    (risk_factors.high_systolic_blood_pressure, ["relative_risk"]),
]
MEASURES_R = ["relative_risk"]


@pytest.mark.skip("TODO: [mic-5456]")
@pytest.mark.parametrize("entity_details", ENTITIES_R, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_R, ids=lambda x: x[0])
def test_get_failing_relative_risk(entity_details, measure):
    entity, _entity_expected_measures = entity_details
    with pytest.raises(DataAbnormalError):
        _df = get_measure(entity, measure, LOCATION)


ENTITIES_R = [
    (risk_factors.iron_deficiency, ["relative_risk"]),
]
MEASURES_R = ["relative_risk"]


@pytest.mark.skip("TODO: [mic-5456]")
@pytest.mark.parametrize("entity_details", ENTITIES_R, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_R, ids=lambda x: x[0])
def test_get_working_relative_risk(entity_details, measure):
    entity, _entity_expected_measures = entity_details
    _df = success_expected(entity, measure, utility_data.get_location_id(LOCATION))


####################
# HELPER FUNCTIONS #
####################


def run_test(
    entity_details: tuple[Cause, list[str]],
    measure: str,
    data_type: str | list[str],
    mock_gbd: bool,
    runslow: bool,
    mocker: MockerFixture,
    is_unimplemented: bool,
):
    entity, entity_expected_measures = entity_details

    if not runslow and not mock_gbd:
        pytest.skip("Need --runslow option to run unmocked tests")
    if NO_GBD_ACCESS and not mock_gbd:
        pytest.skip("Need GBD access to run unmocked tests")

    tester = success_expected if measure in entity_expected_measures else fail_expected
    if is_unimplemented:
        tester = partial(fail_expected, raise_type=DataTypeNotImplementedError)

    if mock_gbd:
        # TODO: Reduce duplicate testing. Since this data is mocked, it is doing the
        # same test for a given measure regardless of the entity.
        if is_unimplemented:
            pytest.skip("Cannot mock data for unimplemented features.")
        if tester == fail_expected:
            pytest.skip("Do mock data for expected failed calls.")
        mocked_funcs = mock_vivarium_gbd_access(entity, measure, data_type, mocker)

    tester(entity, measure, utility_data.get_location_id(LOCATION), data_type)
    if mock_gbd:
        for mocked_func in mocked_funcs:
            assert mocked_func.called_once()


def success_expected(entity, measure, location, data_type):
    df = get_measure(entity, measure, location, years=None, data_type=data_type)
    check_data(df, data_type)


def fail_expected(entity, measure, location, data_type, raise_type=Exception):
    with pytest.raises(raise_type):
        _df = get_measure(entity, measure, location, years=None, data_type=data_type)


def check_data(data: pd.DataFrame, data_type: str) -> None:
    """Check the data returned."""
    assert all(col in data.columns for col in SUPPORTED_DATA_TYPES[data_type])
    assert all(data.notna())
    assert all(data >= 0)
    # Check metadata index values (note that there may be other metadata returned)
    age_bins = get_mocked_age_bins()
    expected_metadata = {
        "location": {LOCATION},
        "sex": {"Male", "Female"},
        "age_start": set(age_bins["age_group_years_start"]),
        "age_end": set(age_bins["age_group_years_end"]),
        "year_start": {YEAR},
        "year_end": {YEAR + 1},
    }
    for idx, expected in expected_metadata.items():
        assert set(data.index.get_level_values(idx)) == expected
