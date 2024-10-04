import pytest
from gbd_mapping import ModelableEntity, causes, covariates, risk_factors

from tests.conftest import NO_GBD_ACCESS
from vivarium_inputs import extract, utility_data

pytestmark = pytest.mark.skipif(
    NO_GBD_ACCESS, reason="Cannot run these tests without vivarium_gbd_access"
)


VALIDATE_FLAG = False


def success_expected(entity_name, measure_name, location):
    df = extract.extract_data(entity_name, measure_name, location, validate=VALIDATE_FLAG)
    return df


def fail_expected(entity_name, measure_name, location):
    with pytest.raises(Exception):
        _df = extract.extract_data(
            entity_name, measure_name, location, validate=VALIDATE_FLAG
        )


ENTITIES_C = [
    (
        causes.hiv_aids,
        ["incidence_rate", "prevalence", "birth_prevalence", "remission_rate", "deaths"],
    ),
    (
        causes.neural_tube_defects,
        ["incidence_rate", "prevalence", "birth_prevalence", "deaths"],
    ),
]
MEASURES_C = [
    "incidence_rate",
    "prevalence",
    "birth_prevalence",
    "disability_weight",
    "remission_rate",
    "deaths",
]
LOCATIONS_C = ["India"]


@pytest.mark.parametrize("entity", ENTITIES_C, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_C, ids=lambda x: x[0])
@pytest.mark.parametrize("location", LOCATIONS_C)
def test_extract_causelike(entity, measure, location):
    entity_name, entity_expected_measures = entity
    tester = success_expected if measure in entity_expected_measures else fail_expected
    _df = tester(entity_name, measure, utility_data.get_location_id(location))


ENTITIES_R = [
    (
        risk_factors.high_fasting_plasma_glucose,
        [
            "exposure",
            "exposure_standard_deviation",
            "exposure_distribution_weights",
            "relative_risk",
            "population_attributable_fraction",
            "etiology_population_attributable_fraction",
            "mediation_factors",
        ],
    ),
    (
        risk_factors.low_birth_weight_and_short_gestation,
        [
            "exposure",
            "relative_risk",
            "population_attributable_fraction",
            "etiology_population_attributable_fraction",
        ],
    ),
]
MEASURES_R = [
    "exposure",
    "exposure_standard_deviation",
    "exposure_distribution_weights",
    # "relative_risk",  # TODO: Add back in with Mic-4936
    "population_attributable_fraction",
    "etiology_population_attributable_fraction",
    "mediation_factors",
]
LOCATIONS_R = ["India"]


@pytest.mark.parametrize("entity", ENTITIES_R, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_R, ids=lambda x: x[0])
@pytest.mark.parametrize("location", LOCATIONS_R)
def test_extract_risklike(entity, measure, location):
    entity_name, entity_expected_measures = entity
    tester = success_expected if measure in entity_expected_measures else fail_expected
    _df = tester(entity_name, measure, utility_data.get_location_id(location))


entity_cov = [
    covariates.systolic_blood_pressure_mmhg,
]
measures_cov = ["estimate"]
locations_cov = ["India"]


@pytest.mark.parametrize("entity", entity_cov)
@pytest.mark.parametrize("measure", measures_cov)
@pytest.mark.parametrize("location", locations_cov)
def test_extract_covariatelike(entity, measure, location):
    _df = extract.extract_data(
        entity, measure, utility_data.get_location_id(location), validate=VALIDATE_FLAG
    )


@pytest.mark.parametrize(
    "measures", ["structure", "theoretical_minimum_risk_life_expectancy"]
)
def test_extract_population(measures):
    pop = ModelableEntity("ignored", "population", None)
    _df = extract.extract_data(
        pop, measures, utility_data.get_location_id("India"), validate=VALIDATE_FLAG
    )


# TODO: Remove with Mic-4936
@pytest.mark.parametrize("entity", ENTITIES_R, ids=lambda x: x[0].name)
@pytest.mark.parametrize("location", LOCATIONS_R)
@pytest.mark.xfail(reason="New relative risk data is not set up for processing yet")
def test_extract_relative_risk(entity, location):
    measure_name = "relative_risk"
    entity_name, _entity_expected_measures = entity
    _df = extract.extract_data(entity_name, measure_name, location)
