import pytest
from gbd_mapping import ModelableEntity, causes, covariates, risk_factors

from tests.conftest import NO_GBD_ACCESS
from vivarium_inputs import core, utility_data
from vivarium_inputs.mapping_extension import healthcare_entities

pytestmark = pytest.mark.skipif(
    NO_GBD_ACCESS, reason="Cannot run these tests without vivarium_gbd_access"
)


def success_expected(entity_name, measure_name, location):
    df = core.get_data(entity_name, measure_name, location)
    return df


def fail_expected(entity_name, measure_name, location):
    with pytest.raises(Exception):
        _df = core.get_data(entity_name, measure_name, location)


def check_year_in_data(entity, measure, location, years):
    if isinstance(years, list):
        df = core.get_data(entity, measure, location, years=years)
        assert set(df.reset_index()["year_id"]) == set(years)
    # years expected to be 1900, 2019, None, or "all"
    elif years != 1900:
        df = core.get_data(entity, measure, location, years=years)
        if years == None:
            assert set(df.reset_index()["year_id"]) == set([2021])
        elif years == 2019:
            assert set(df.reset_index()["year_id"]) == set([2019])
        elif years == "all":
            assert set(df.reset_index()["year_id"]) == set(range(1990, 2023))
    else:
        with pytest.raises(ValueError, match="years must be in"):
            df = core.get_data(entity, measure, location, years=years)


ENTITIES_C = [
    (
        causes.measles,
        [
            "incidence_rate",
            "raw_incidence_rate",
            "prevalence",
            "disability_weight",
            "cause_specific_mortality_rate",
            "excess_mortality_rate",
            "deaths",
        ],
    ),
    (
        causes.diarrheal_diseases,
        [
            "incidence_rate",
            "raw_incidence_rate",
            "prevalence",
            "disability_weight",
            "remission_rate",
            "cause_specific_mortality_rate",
            "excess_mortality_rate",
            "deaths",
        ],
    ),
    (
        causes.diabetes_mellitus_type_2,
        [
            "incidence_rate",
            "raw_incidence_rate",
            "prevalence",
            "disability_weight",
            "cause_specific_mortality_rate",
            "excess_mortality_rate",
            "deaths",
        ],
    ),
]
MEASURES_C = [
    "incidence_rate",
    "raw_incidence_rate",
    "prevalence",
    "birth_prevalence",
    "disability_weight",
    "remission_rate",
    "cause_specific_mortality_rate",
    "excess_mortality_rate",
    "deaths",
]
LOCATIONS_C = ["India"]


@pytest.mark.parametrize("entity_details", ENTITIES_C, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_C, ids=lambda x: x[0])
@pytest.mark.parametrize("location", LOCATIONS_C)
def test_core_causelike(entity_details, measure, location):
    entity, entity_expected_measures = entity_details
    tester = success_expected if measure in entity_expected_measures else fail_expected
    _df = tester(entity, measure, utility_data.get_location_id(location))


@pytest.mark.parametrize("entity_details", ENTITIES_C, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_C, ids=lambda x: x[0])
@pytest.mark.parametrize("location", LOCATIONS_C)
@pytest.mark.parametrize("years", [None, 2019, 1900, [2019], [2019, 2020, 2021], "all"])
def test_year_id_causelike(entity_details, measure, location, years):
    entity, entity_expected_measures = entity_details
    if measure in entity_expected_measures:
        check_year_in_data(entity, measure, location, years=years)


ENTITIES_R = [
    (
        risk_factors.high_systolic_blood_pressure,
        [
            "exposure",
            "exposure_standard_deviation",
            "exposure_distribution_weights",
            "relative_risk",
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
    "relative_risk",
    "population_attributable_fraction",
]
LOCATIONS_R = ["India"]


@pytest.mark.parametrize("entity_details", ENTITIES_R, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_R, ids=lambda x: x[0])
@pytest.mark.parametrize("location", LOCATIONS_R)
def test_core_risklike(entity_details, measure, location):
    entity, entity_expected_measures = entity_details
    if (
        entity.name == risk_factors.high_systolic_blood_pressure.name
        and measure == "population_attributable_fraction"
    ):
        pytest.skip("MIC-4891")
    tester = success_expected if measure in entity_expected_measures else fail_expected
    _df = tester(entity, measure, utility_data.get_location_id(location))


@pytest.mark.parametrize("entity_details", ENTITIES_R, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_R, ids=lambda x: x[0])
@pytest.mark.parametrize("location", LOCATIONS_R)
@pytest.mark.parametrize("years", [None, 2019, 1900, [2019], [2019, 2020, 2021], "all"])
def test_year_id_risklike(entity_details, measure, location, years):
    entity, entity_expected_measures = entity_details
    # exposure-parametrized RRs for all years requires a lot of time and memory to process
    if (
        entity == risk_factors.high_systolic_blood_pressure
        and measure == "relative_risk"
        and years == "all"
    ):
        pytest.skip(reason="need --runslow option to run")
    if measure in entity_expected_measures:
        check_year_in_data(entity, measure, location, years=years)


@pytest.mark.slow  # this test requires a lot of time and memory to run
@pytest.mark.parametrize("location", LOCATIONS_R)
def test_slow_year_id_risklike(location):
    check_year_in_data(
        risk_factors.high_systolic_blood_pressure, "relative_risk", location, years="all"
    )


ENTITIES_COV = [
    covariates.systolic_blood_pressure_mmhg,
]
MEASURES_COV = ["estimate"]
LOCATIONS_COV = ["India"]


@pytest.mark.parametrize("entity", ENTITIES_COV, ids=lambda x: x.name)
@pytest.mark.parametrize("measure", MEASURES_COV, ids=lambda x: x)
@pytest.mark.parametrize("location", LOCATIONS_COV)
def test_core_covariatelike(entity, measure, location):
    _df = core.get_data(entity, measure, utility_data.get_location_id(location))


@pytest.mark.parametrize("entity", ENTITIES_COV, ids=lambda x: x.name)
@pytest.mark.parametrize("measure", MEASURES_COV, ids=lambda x: x)
@pytest.mark.parametrize("location", LOCATIONS_COV)
@pytest.mark.parametrize("years", [None, 2019, 1900, [2019], [2019, 2020, 2021], "all"])
def test_year_id_covariatelike(entity, measure, location, years):
    check_year_in_data(entity, measure, location, years=years)


@pytest.mark.parametrize(
    "measures",
    [
        "structure",
        "age_bins",
        "demographic_dimensions",
        "theoretical_minimum_risk_life_expectancy",
    ],
)
def test_core_population(measures):
    pop = ModelableEntity("ignored", "population", None)
    _df = core.get_data(pop, measures, utility_data.get_location_id("India"))


@pytest.mark.parametrize("measure", ["structure", "demographic_dimensions"])
@pytest.mark.parametrize("years", [None, 2019, 1900, [2019], [2019, 2020, 2021], "all"])
def test_year_id_population(measure, years):
    pop = ModelableEntity("ignored", "population", None)
    location = utility_data.get_location_id("India")
    check_year_in_data(pop, measure, location, years=years)


# TODO - Underlying problem with gbd access. Remove when corrected.
ENTITIES_HEALTH_SYSTEM = [
    healthcare_entities.outpatient_visits,
]
MEASURES_HEALTH_SYSTEM = ["utilization_rate"]
LOCATIONS_HEALTH_SYSTEM = ["India"]


@pytest.mark.skip(reason="Underlying problem with gbd access. Remove when corrected.")
@pytest.mark.parametrize("entity", ENTITIES_HEALTH_SYSTEM, ids=lambda x: x.name)
@pytest.mark.parametrize("measure", MEASURES_HEALTH_SYSTEM, ids=lambda x: x)
@pytest.mark.parametrize("location", LOCATIONS_HEALTH_SYSTEM)
def test_core_healthsystem(entity, measure, location):
    _df = core.get_data(entity, measure, utility_data.get_location_id(location))


@pytest.mark.parametrize("entity_details", ENTITIES_C, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_C, ids=lambda x: x[0])
@pytest.mark.parametrize(
    "locations",
    [
        [164, 165, 175],
        ["Ethiopia", "Nigeria"],
        [164, "Nigeria"],
    ],
)
def test_pulling_multiple_locations(entity_details, measure, locations):
    entity, entity_expected_measures = entity_details
    measure_name, measure_id = measure
    tester = success_expected if (entity_expected_measures & measure_id) else fail_expected
    _df = tester(entity, measure_name, locations)
