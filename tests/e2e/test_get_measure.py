from functools import partial

import pandas as pd
import pytest
from gbd_mapping import causes, covariates, risk_factors, sequelae
from layered_config_tree import LayeredConfigTree
from pytest_mock import MockerFixture

from tests.conftest import NO_GBD_ACCESS
from vivarium_inputs import utility_data
from vivarium_inputs.globals import SUPPORTED_DATA_TYPES, DataAbnormalError
from vivarium_inputs.interface import get_measure
from vivarium_inputs.utilities import DataTypeNotImplementedError

pytestmark = pytest.mark.skipif(
    NO_GBD_ACCESS, reason="Cannot run these tests without vivarium_gbd_access"
)

LOCATION = "India"


@pytest.fixture(autouse=True)
def no_cache(mocker: MockerFixture) -> None:
    """Mock out the cache so that we always pull data."""

    if not NO_GBD_ACCESS:
        mocker.patch(
            "vivarium_gbd_access.utilities.get_input_config",
            return_value=LayeredConfigTree({"input_data": {"cache_data": False}}),
        )


def success_expected(entity, measure, location, data_type):
    df = get_measure(entity, measure, location, years=None, data_type=data_type)
    check_data(df, data_type)


def fail_expected(entity, measure, location, data_type, raise_type=Exception):
    with pytest.raises(raise_type):
        _df = get_measure(entity, measure, location, years=None, data_type=data_type)


ENTITIES_C = [
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
MEASURES_C = [
    "incidence_rate",
    "prevalence",
    "birth_prevalence",
    "disability_weight",
    "remission_rate",
    "cause_specific_mortality_rate",
    "excess_mortality_rate",
]


@pytest.mark.slow
@pytest.mark.parametrize("entity_details", ENTITIES_C, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_C, ids=lambda x: x)
@pytest.mark.parametrize(
    "data_type", ["means", "draws", ["means", "draws"]], ids=("means", "draws", "means_draws")
)
def test_get_measure_causelike(entity_details, measure, data_type):

    entity, entity_expected_measures = entity_details
    location_id = utility_data.get_location_id(LOCATION)

    if NO_GBD_ACCESS:
        pytest.skip("Need GBD access to run this test")
    if measure == "birth_prevalence":
        pytest.skip("FIXME: need to find causes with birth prevalence")

    tester = success_expected if measure in entity_expected_measures else fail_expected
    # Handle not implemented
    if isinstance(data_type, list):
        tester = partial(fail_expected, raise_type=DataTypeNotImplementedError)
    if data_type == "means" and measure in [
        "disability_weight",
        "remission_rate",
        "cause_specific_mortality_rate",
        "excess_mortality_rate",
    ]:
        tester = partial(fail_expected, raise_type=DataTypeNotImplementedError)

    tester(entity, measure, location_id, data_type)


ENTITIES_S = [
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
MEASURES_S = [
    "incidence_rate",
    "prevalence",
    "birth_prevalence",
    "disability_weight",
]


@pytest.mark.slow
@pytest.mark.parametrize("entity_details", ENTITIES_S, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_S, ids=lambda x: x)
@pytest.mark.parametrize(
    "data_type", ["means", "draws", ["means", "draws"]], ids=("means", "draws", "means_draws")
)
def test_get_measure_sequelalike(entity_details, measure, data_type):

    entity, entity_expected_measures = entity_details
    location_id = utility_data.get_location_id(LOCATION)

    if NO_GBD_ACCESS:
        pytest.skip("Need GBD access to run this test")
    if measure == "birth_prevalence":
        pytest.skip("FIXME: need to find sequelae with birth prevalence")

    tester = success_expected if measure in entity_expected_measures else fail_expected
    # Handle not implemented
    if isinstance(data_type, list):
        tester = partial(fail_expected, raise_type=DataTypeNotImplementedError)
    if data_type == "means" and measure in [
        "disability_weight",
    ]:
        tester = partial(fail_expected, raise_type=DataTypeNotImplementedError)

    tester(entity, measure, location_id, data_type)


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


@pytest.mark.skip("TODO: [mic-5407]")
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


@pytest.mark.skip("TODO: [mic-5407]")
@pytest.mark.parametrize("entity", ENTITIES_COV, ids=lambda x: x.name)
@pytest.mark.parametrize("measure", MEASURES_COV, ids=lambda x: x)
def test_get_measure_covariatelike(entity, measure):
    _df = get_measure(entity, measure, utility_data.get_location_id(LOCATION))


# TODO: Remove with Mic-4936
ENTITIES_R = [
    (risk_factors.high_systolic_blood_pressure, ["relative_risk"]),
]
MEASURES_R = ["relative_risk"]


@pytest.mark.skip("TODO: [mic-5407]")
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


@pytest.mark.skip("TODO: [mic-5407]")
@pytest.mark.parametrize("entity_details", ENTITIES_R, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_R, ids=lambda x: x[0])
def test_get_working_relative_risk(entity_details, measure):
    entity, _entity_expected_measures = entity_details
    _df = success_expected(entity, measure, utility_data.get_location_id(LOCATION))


####################
# HELPER FUNCTIONS #
####################


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
        "year_start": {2021},
        "year_end": {2022},
    }
    for idx, expected in expected_metadata.items():
        assert set(data.index.get_level_values(idx)) == expected


###############
# MOCKED DATA #
###############


def get_mocked_age_bins() -> pd.DataFrame:
    """Mocked age bins for testing."""
    return pd.read_csv("tests/fixture_data/age_bins.csv")
