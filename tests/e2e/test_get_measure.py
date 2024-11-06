"""End-to-end tests for interface.get_measure()."""

from __future__ import annotations

import datetime
from functools import partial

import pandas as pd
import pytest
from gbd_mapping import (
    Cause,
    RiskFactor,
    Sequela,
    causes,
    covariates,
    risk_factors,
    sequelae,
)
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
from vivarium_inputs.globals import (
    NON_STANDARD_MEASURES,
    SUPPORTED_DATA_TYPES,
    DataAbnormalError,
)
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
    """Test get_measure().

    We are parametrizing over various things.
    - entity_details: A tuple of an entity and a list of measures that are applicable to that entity.
    - measure: The possible measures for that entity type (e.g. cause, sequela, etc)
    - data_type: The data type request (means or draws)
    - mock_gbd: Whether to mock calls to vivarium_gbd_access or not. We do this because
        getting the real data tends to take a long time and so we want to only run that
        when requesting a --runslow test. Note that we do not attempt to get more granular
        with slow runs; mocked tests will always run and unmocked tests will only run
        if --runslow is passed

    Notes
    -----
    The --runslow flag is automatically passed into these tests as an argument; we do
    not mark the tests themselves as slow (because we want more granularity, i.e.
    mocked tests are not slow but unmocked tests are).
    """
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
    """Test get_measure().

    We are parametrizing over various things.
    - entity_details: A tuple of an entity and a list of measures that are applicable to that entity.
    - measure: The possible measures for that entity type (e.g. cause, sequela, etc)
    - data_type: The data type request (means or draws)
    - mock_gbd: Whether to mock calls to vivarium_gbd_access or not. We do this because
        getting the real data tends to take a long time and so we want to only run that
        when requesting a --runslow test. Note that we do not attempt to get more granular
        with slow runs; mocked tests will always run and unmocked tests will only run
        if --runslow is passed

    Notes
    -----
    The --runslow flag is automatically passed into these tests as an argument; we do
    not mark the tests themselves as slow (because we want more granularity, i.e.
    mocked tests are not slow but unmocked tests are).
    """

    # Test-specific fixme skips
    if measure == "birth_prevalence":
        pytest.skip("FIXME: need to find sequelae with birth prevalence")

    # Handle not implemented
    is_unimplemented_means = data_type == "means" and measure in [
        "disability_weight",
    ]
    is_unimplemented = isinstance(data_type, list) or is_unimplemented_means

    run_test(entity_details, measure, data_type, mock_gbd, runslow, mocker, is_unimplemented)


RISK_FACTORS = [
    (
        risk_factors.high_systolic_blood_pressure,
        [
            "exposure",
            "exposure_standard_deviation",
            "exposure_distribution_weights",
            # "relative_risk",  # TODO: Add back in once Mic-4936 is resolved
            "population_attributable_fraction",  # Very slow
        ],
    ),
    (
        risk_factors.low_birth_weight_and_short_gestation,
        [
            "exposure",
            # "relative_risk",  # TODO: Add back in once Mic-4936 is resolved
            "population_attributable_fraction",  # Very slow
        ],
    ),
]
RISK_FACTOR_MEASURES = [
    "exposure",
    "exposure_standard_deviation",
    "exposure_distribution_weights",
    "relative_risk",
    "population_attributable_fraction",
    # "mediation_factors",
    #   QUESTION: are we supposed to support mediation_factors? There is no mapping
    #   for interface.get_measure(), but there is an extraction method
    #   (i.e. interface.get_raw_data())
]


@pytest.mark.parametrize("entity_details", RISK_FACTORS, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", RISK_FACTOR_MEASURES, ids=lambda x: x)
@pytest.mark.parametrize(
    "data_type", ["means", "draws", ["means", "draws"]], ids=("means", "draws", "means_draws")
)
@pytest.mark.parametrize("mock_gbd", [False], ids=("unmocked",))  # TODO: mock_id=True
def test_get_measure_risklike(
    entity_details: RiskFactor,
    measure: str,
    data_type: str | list[str],
    mock_gbd: bool,
    runslow: bool,
    mocker: MockerFixture,
):
    """Test get_measure().

    We are parametrizing over various things.
    - entity_details: A tuple of an entity and a list of measures that are applicable to that entity.
    - measure: The possible measures for that entity type (e.g. cause, sequela, etc)
    - data_type: The data type request (means or draws)
    - mock_gbd: Whether to mock calls to vivarium_gbd_access or not. We do this because
        getting the real data tends to take a long time and so we want to only run that
        when requesting a --runslow test. Note that we do not attempt to get more granular
        with slow runs; mocked tests will always run and unmocked tests will only run
        if --runslow is passed

    Notes
    -----
    The --runslow flag is automatically passed into these tests as an argument; we do
    not mark the tests themselves as slow (because we want more granularity, i.e.
    mocked tests are not slow but unmocked tests are).
    """

    # Test-specific fixme skips
    if measure == "relative_risk":
        pytest.skip("FIXME: [mic-4936]")

    # Handle not implemented
    is_unimplemented = isinstance(data_type, list) or data_type == "means"

    run_test(entity_details, measure, data_type, mock_gbd, runslow, mocker, is_unimplemented)


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
    entity_details: tuple[Cause | Sequela | RiskFactor, list[str]],
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
    # Only run PAF tests on Sundays since it's so slow. Note that this
    # could indadvertently be skipped if there is a significant delay between when
    # a jenkins pipeline is kicked off and when the test itself is run.
    if (
        measure in entity_expected_measures
        and measure == "population_attributable_fraction"
        and not mock_gbd
        and datetime.datetime.today().weekday() != 6
    ):
        pytest.skip("Only run full PAF tests on Sundays")

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
    check_data(measure, df, data_type)


def fail_expected(entity, measure, location, data_type, raise_type=Exception):
    with pytest.raises(raise_type):
        _df = get_measure(entity, measure, location, years=None, data_type=data_type)


def check_data(measure: str, data: pd.DataFrame, data_type: str) -> None:
    """Check the data returned."""
    if measure in NON_STANDARD_MEASURES:
        assert list(data.columns) == ["value"]
    else:
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
