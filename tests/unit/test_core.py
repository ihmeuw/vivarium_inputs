import pytest
from gbd_mapping import Cause, ModelableEntity, causes, covariates, risk_factors
from pytest_mock import MockerFixture
from tests.mocked_gbd import (
    LOCATION,
    get_mocked_age_bins,
    mock_vivarium_gbd_access,
)
from functools import partial
from vivarium_inputs import core, utility_data
from vivarium_inputs.utilities import DataType

CAUSES = [
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
    # (
    #     causes.diarrheal_diseases,
    #     [
    #         "incidence_rate",
    #         "raw_incidence_rate",
    #         "prevalence",
    #         "disability_weight",
    #         "remission_rate",
    #         "cause_specific_mortality_rate",
    #         "excess_mortality_rate",
    #         "deaths",
    #     ],
    # ),
    # (
    #     causes.diabetes_mellitus_type_2,
    #     [
    #         "incidence_rate",
    #         "raw_incidence_rate",
    #         "prevalence",
    #         "disability_weight",
    #         "cause_specific_mortality_rate",
    #         "excess_mortality_rate",
    #         "deaths",
    #     ],
    # ),
]
CAUSE_MEASURES = [
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


@pytest.mark.parametrize("entity_details", CAUSES, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", CAUSE_MEASURES, ids=lambda x: x)
# @pytest.mark.parametrize("years", [None, 2019, 1900, [2019], [2019, 2020, 2021], "all"], ids=lambda x: x if not isinstance(x, list) else "list")
# @pytest.mark.parametrize(
#     "data_type", ["means", "draws", ["means", "draws"]], ids=("means", "draws", "means_draws")
# )
@pytest.mark.parametrize("years", ["all"], ids=lambda x: x if not isinstance(x, list) else "list")
@pytest.mark.parametrize(
    "data_type", ["means", "draws"]
)
def test_year_id_causelike(
    entity_details: tuple[Cause, list[str]],
    measure: str,
    years: int | list[int] | None | str,
    data_type: str | list[str],
    mocker: MockerFixture,
):
    # Handle not implemented
    is_unimplemented_means = data_type == "means" and measure in [
        "disability_weight",
        "remission_rate",
        "cause_specific_mortality_rate",
        "excess_mortality_rate",
        "deaths",
    ]
    is_unimplemented = isinstance(data_type, list) or is_unimplemented_means

    check_year_in_data(entity_details, measure, LOCATION, years, data_type, is_unimplemented, mocker)


RISKS = [
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
RISK_MEASURES = [
    "exposure",
    "exposure_standard_deviation",
    "exposure_distribution_weights",
    "relative_risk",
    "population_attributable_fraction",
]


@pytest.mark.skip("TODO - MIC-5245")
@pytest.mark.parametrize("entity_details", RISKS, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", RISK_MEASURES, ids=lambda x: x[0])
@pytest.mark.parametrize("years", [None, 2019, 1900, [2019], [2019, 2020, 2021], "all"])
def test_year_id_risklike(entity_details, measure, years):
    entity, entity_expected_measures = entity_details
    # exposure-parametrized RRs for all years requires a lot of time and memory to process
    if (
        entity == risk_factors.high_systolic_blood_pressure
        and measure == "relative_risk"
        and years == "all"
    ):
        pytest.skip(reason="need --runslow option to run")
    if measure in entity_expected_measures:
        check_year_in_data(entity, measure, LOCATION, years)


@pytest.mark.skip("TODO - MIC-5245")
@pytest.mark.slow  # this test requires a lot of time and memory to run
def test_slow_year_id_risklike(location):
    check_year_in_data(
        risk_factors.high_systolic_blood_pressure, "relative_risk", LOCATION, years="all"
    )


COVARIATES = [
    covariates.systolic_blood_pressure_mmhg,
]
COVARIATE_MEASURES = ["estimate"]


@pytest.mark.skip("TODO - MIC-5245")
@pytest.mark.parametrize("entity", COVARIATES, ids=lambda x: x.name)
@pytest.mark.parametrize("measure", COVARIATE_MEASURES, ids=lambda x: x)
@pytest.mark.parametrize("years", [None, 2019, 1900, [2019], [2019, 2020, 2021], "all"])
def test_year_id_covariatelike(entity, measure, years):
    check_year_in_data(entity, measure, LOCATION, years)


@pytest.mark.skip("TODO - MIC-5245")
@pytest.mark.parametrize("measure", ["structure", "demographic_dimensions"])
@pytest.mark.parametrize("years", [None, 2019, 1900, [2019], [2019, 2020, 2021], "all"])
def test_year_id_population(measure, years):
    pop = ModelableEntity("ignored", "population", None)
    location = utility_data.get_location_id(LOCATION)
    check_year_in_data(pop, measure, location, years=years)


@pytest.mark.skip("TODO - MIC-5245")
@pytest.mark.parametrize("entity_details", CAUSES, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", CAUSE_MEASURES, ids=lambda x: x[0])
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


####################
# Helper functions #
####################


def success_expected(entity_name, measure_name, location):
    df = core.get_data(entity_name, measure_name, location)
    return df


def fail_expected(entity_name, measure_name, location):
    with pytest.raises(Exception):
        _df = core.get_data(entity_name, measure_name, location)


def check_year_in_data(
    entity_details: tuple[ModelableEntity, list[str]],
    measure: str,
    location,
    years: int | list[int] | None | str,
    data_type: str | list[str],
    is_unimplemented: bool,
    mocker: MockerFixture,
):
    if is_unimplemented:
        pytest.skip("Cannot mock data for unimplemented features.")

    entity, entity_expected_measures = entity_details
    data_type = DataType(measure, data_type)

    if measure not in entity_expected_measures:
        tester = fail_expected
    elif isinstance(years, list) or years != 1900:
        tester = success_expected
    else:
        tester = partial(fail_expected, raise_type=ValueError, match="years must be in")

    # mocked_funcs = mock_vivarium_gbd_access(entity, measure, data_type.type, mocker)
    tester(entity, measure, location, years, data_type)
    # for mocked_func in mocked_funcs:
    #     assert mocked_func.called_once()


def success_expected(entity, measure, location, years, data_type):
    df = core.get_data(entity, measure, location, years, data_type)
    if isinstance(years, list):
        assert set(df.reset_index()["year_id"]) == set(years)
    else:  # years != 1900:
        if years == None:
            assert set(df.reset_index()["year_id"]) == set([2021])
        elif years == 2019:
            assert set(df.reset_index()["year_id"]) == set([2019])
        elif years == "all":
            assert set(df.reset_index()["year_id"]) == set(range(1990, 2023))

def fail_expected(entity, measure, location, years, data_type, raise_type=Exception, match=None):
    with pytest.raises(raise_type, match=match):
        _df = core.get_data(entity, measure, location, years, data_type)