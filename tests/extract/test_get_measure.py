import pytest
from gbd_mapping import causes, covariates, risk_factors

from tests.extract.check import RUNNING_ON_CI
from vivarium_inputs import utility_data
from vivarium_inputs.globals import DataAbnormalError
from vivarium_inputs.interface import get_measure

pytestmark = pytest.mark.skipif(
    RUNNING_ON_CI, reason="Don't run these tests on the CI server"
)


def success_expected(entity_name, measure_name, location):
    df = get_measure(entity_name, measure_name, location)
    return df


def fail_expected(entity_name, measure_name, location):
    with pytest.raises(Exception):
        _df = get_measure(entity_name, measure_name, location)


ENTITIES_C = [
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
LOCATIONS_C = ["India"]


@pytest.mark.parametrize("entity_details", ENTITIES_C, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_C, ids=lambda x: x[0])
@pytest.mark.parametrize("location", LOCATIONS_C)
def test_get_measure_causelike(entity_details, measure, location):
    entity, entity_expected_measures = entity_details
    tester = success_expected if measure in entity_expected_measures else fail_expected
    _df = tester(entity, measure, utility_data.get_location_id(location))


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
LOCATIONS_R = ["India"]


@pytest.mark.parametrize("entity_details", ENTITIES_R, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_R, ids=lambda x: x[0])
@pytest.mark.parametrize("location", LOCATIONS_R)
def test_get_measure_risklike(entity_details, measure, location):
    entity, entity_expected_measures = entity_details
    tester = success_expected if measure in entity_expected_measures else fail_expected
    _df = tester(entity, measure, utility_data.get_location_id(location))


ENTITIES_COV = [
    covariates.systolic_blood_pressure_mmhg,
]
MEASURES_COV = ["estimate"]
LOCATIONS_COV = ["India"]


@pytest.mark.parametrize("entity", ENTITIES_COV, ids=lambda x: x.name)
@pytest.mark.parametrize("measure", MEASURES_COV, ids=lambda x: x)
@pytest.mark.parametrize("location", LOCATIONS_COV)
def test_get_measure_covariatelike(entity, measure, location):
    _df = get_measure(entity, measure, utility_data.get_location_id(location))


# TODO: Remove with Mic-4936
ENTITIES_R = [
    (risk_factors.high_systolic_blood_pressure, ["relative_risk"]),
]
MEASURES_R = ["relative_risk"]
LOCATIONS_R = ["India"]


@pytest.mark.parametrize("entity_details", ENTITIES_R, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_R, ids=lambda x: x[0])
@pytest.mark.parametrize("location", LOCATIONS_R)
def test_get_failing_relative_risk(entity_details, measure, location):
    entity, _entity_expected_measures = entity_details
    with pytest.raises(DataAbnormalError):
        _df = get_measure(entity, measure, location)


ENTITIES_R = [
    (risk_factors.iron_deficiency, ["relative_risk"]),
]
MEASURES_R = ["relative_risk"]
LOCATIONS_R = ["India"]


@pytest.mark.parametrize("entity_details", ENTITIES_R, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_R, ids=lambda x: x[0])
@pytest.mark.parametrize("location", LOCATIONS_R)
def test_get_working_relative_risk(entity_details, measure, location):
    entity, _entity_expected_measures = entity_details
    _df = success_expected(entity, measure, utility_data.get_location_id(location))
