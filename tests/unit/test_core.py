import pytest
from gbd_mapping import ModelableEntity, causes, covariates, risk_factors

from tests.conftest import NO_GBD_ACCESS
from vivarium_inputs import core, utility_data
from vivarium_inputs.utilities import DataType, DataTypeNotImplementedError

pytestmark = pytest.mark.skipif(
    NO_GBD_ACCESS, reason="Cannot run these tests without vivarium_gbd_access"
)


def check_year_in_data(entity, measure, location, years, data_type):
    if _is_not_implemented(data_type, measure):
        with pytest.raises(DataTypeNotImplementedError):
            data_type = DataType(measure, data_type)
            core.get_data(entity, measure, location, years, data_type)
    else:
        data_type = DataType(measure, data_type)
        if isinstance(years, list):
            df = core.get_data(entity, measure, location, years, data_type)
            assert set(df.index.get_level_values("year_id")) == set(years)
        # years expected to be 1900, 2019, None, or "all"
        elif years != 1900:
            df = core.get_data(entity, measure, location, years, data_type)
            if years == None:
                assert set(df.index.get_level_values("year_id")) == set([2021])
            elif years == "all":
                assert set(df.index.get_level_values("year_id")) == set(range(1990, 2023))
            else:  # a single (non-1900) year
                assert set(df.index.get_level_values("year_id")) == set([years])
        else:
            with pytest.raises(ValueError, match="years must be in"):
                core.get_data(entity, measure, location, years, data_type)


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
@pytest.mark.parametrize("measure", MEASURES_C, ids=lambda x: x)
@pytest.mark.parametrize("location", LOCATIONS_C, ids=lambda x: x)
@pytest.mark.parametrize(
    "years", [None, 2019, 1900, [2019], [2019, 2020], "all"], ids=lambda x: str(x)
)
@pytest.mark.parametrize(
    "data_type", ["draws", "means", ["draws", "means"]], ids=lambda x: str(x)
)
def test_year_id_causelike(entity_details, measure, location, years, data_type):
    entity, entity_expected_measures = entity_details
    if measure in entity_expected_measures:
        check_year_in_data(entity, measure, location, years, data_type)


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
@pytest.mark.parametrize("measure", MEASURES_R, ids=lambda x: x)
@pytest.mark.parametrize("location", LOCATIONS_R, ids=lambda x: x)
@pytest.mark.parametrize(
    "years", [None, 2019, 1900, [2019], [2019, 2020], "all"], ids=lambda x: str(x)
)
@pytest.mark.parametrize(
    "data_type", ["draws", "means", ["draws", "means"]], ids=lambda x: str(x)
)
def test_year_id_risklike(entity_details, measure, location, years, data_type):
    if measure in ["relative_risk", "population_attributable_fraction"]:
        pytest.skip("TODO: mic-5245 punting until later b/c these are soooo slow")
    entity, entity_expected_measures = entity_details
    if measure in entity_expected_measures:
        check_year_in_data(entity, measure, location, years, data_type)


ENTITIES_COV = [
    covariates.systolic_blood_pressure_mmhg,
]
MEASURES_COV = ["estimate"]
LOCATIONS_COV = ["India"]


@pytest.mark.parametrize("entity", ENTITIES_COV, ids=lambda x: x.name)
@pytest.mark.parametrize("measure", MEASURES_COV, ids=lambda x: x)
@pytest.mark.parametrize("location", LOCATIONS_COV, ids=lambda x: x)
@pytest.mark.parametrize(
    "years", [None, 2019, 1900, [2019], [2019, 2020], "all"], ids=lambda x: str(x)
)
@pytest.mark.parametrize(
    "data_type", ["draws", "means", ["draws", "means"]], ids=lambda x: str(x)
)
def test_year_id_covariatelike(entity, measure, location, years, data_type):
    check_year_in_data(entity, measure, location, years, data_type)


@pytest.mark.parametrize("measure", ["structure", "demographic_dimensions"], ids=lambda x: x)
@pytest.mark.parametrize(
    "years", [None, 2019, 1900, [2019], [2019, 2020], "all"], ids=lambda x: str(x)
)
@pytest.mark.parametrize(
    "data_type", ["draws", "means", ["draws", "means"]], ids=lambda x: str(x)
)
def test_year_id_population(measure, years, data_type):
    pop = ModelableEntity("ignored", "population", None)
    location = utility_data.get_location_id("India")
    check_year_in_data(pop, measure, location, years, data_type)


@pytest.mark.parametrize("entity_details", ENTITIES_C, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", MEASURES_C, ids=lambda x: x)
@pytest.mark.parametrize(
    "locations",
    [
        "Ethiopia",  # 179
        179,
        ["Ethiopia", "Nigeria"],  # [179, 214]
        [179, 214],
        [179, "Nigeria"],
    ],
    ids=lambda x: str(x),
)
@pytest.mark.parametrize(
    "data_type", ["draws", "means", ["draws", "means"]], ids=lambda x: str(x)
)
def test_multiple_locations_causelike(entity_details, measure, locations, data_type):
    year = 2021
    location_id_mapping = {
        "Ethiopia": 179,
        "Nigeria": 214,
    }
    entity, entity_expected_measures = entity_details
    if _is_not_implemented(data_type, measure):
        with pytest.raises(DataTypeNotImplementedError):
            data_type = DataType(measure, data_type)
            core.get_data(entity, measure, locations, year, data_type)
    else:
        data_type = DataType(measure, data_type)
        if measure not in entity_expected_measures:
            with pytest.raises(Exception):
                core.get_data(entity, measure, locations, year, data_type)
        else:
            df = core.get_data(entity, measure, locations, year, data_type)
            if not isinstance(locations, list):
                locations = [locations]
            location_ids = {
                (
                    location_id_mapping[item]
                    if isinstance(item, str) and item in location_id_mapping
                    else item
                )
                for item in locations
            }
            assert set(df.index.get_level_values("location_id")) == set(location_ids)


# TODO: Should we add the location tests for other entity types?


def _is_not_implemented(data_type: str | list[str], measure: str) -> bool:
    return isinstance(data_type, list) or (
        data_type == "means"
        and measure
        in [
            "disability_weight",
            "remission_rate",
            "cause_specific_mortality_rate",
            "excess_mortality_rate",
            "deaths",
            "exposure",
            "low_birth_weight_and_short_gestation",
            "exposure_standard_deviation",
            "exposure_distribution_weights",
            "estimate",
            "structure",
        ]
    )
