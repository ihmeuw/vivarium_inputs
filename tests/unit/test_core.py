import pytest
from gbd_mapping import ModelableEntity, causes, covariates, risk_factors
from pytest_mock import MockerFixture

from tests.conftest import is_not_implemented
from tests.mocked_gbd import (
    MOST_RECENT_YEAR,
    get_mocked_location_ids,
    mock_vivarium_gbd_access,
)
from vivarium_inputs import core
from vivarium_inputs.globals import Population
from vivarium_inputs.utilities import DataType, DataTypeNotImplementedError

LOCATION = "India"

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
@pytest.mark.parametrize(
    "years", [None, 2019, 1900, [2019], [2019, 2020], "all"], ids=lambda x: str(x)
)
@pytest.mark.parametrize(
    "data_type", ["draws", "means", ["draws", "means"]], ids=lambda x: str(x)
)
def test_year_id_causelike(
    entity_details: tuple[ModelableEntity, list[str]],
    measure: str,
    years: int | list[int] | str | None,
    data_type: str | list[str],
    mocker: MockerFixture,
):
    if years == "all" and measure == "remission_rate":
        # remission rates and relative risks have binned years and for
        # "all" years will use central comp's `core-maths` library which, like
        # vivarium_gbd_access, is hosted on bitbucket and so cannot be accessed
        # from github-actions.
        pytest.skip("Expected to fail - see test_xfailed test")

    entity, entity_expected_measures = entity_details
    if measure in entity_expected_measures:
        check_year_in_data(entity, measure, LOCATION, years, data_type, mocker)


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


@pytest.mark.parametrize("entity_details", RISKS, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", RISK_MEASURES, ids=lambda x: x)
@pytest.mark.parametrize(
    "years", [None, 2019, 1900, [2019], [2019, 2020], "all"], ids=lambda x: str(x)
)
@pytest.mark.parametrize(
    "data_type", ["draws", "means", ["draws", "means"]], ids=lambda x: str(x)
)
def test_year_id_risklike(
    entity_details: tuple[ModelableEntity, list[str]],
    measure: str,
    years: int | list[int] | str | None,
    data_type: str | list[str],
    mocker: MockerFixture,
):
    if years == "all" and measure == "relative_risk":
        # remission rates and relative risks have binned years and for
        # "all" years will use central comp's `core-maths` library which, like
        # vivarium_gbd_access, is hosted on bitbucket and so cannot be accessed
        # from github-actions.
        pytest.skip("Expected to fail - see test_xfailed test")
    if (
        measure == "relative_risk"
        and entity_details[0].name == "high_systolic_blood_pressure"
        and data_type == "draws"
    ):
        pytest.skip("FIXME: [mic-5542] continuous rrs cannot validate")
    entity, entity_expected_measures = entity_details
    if measure in entity_expected_measures:
        check_year_in_data(entity, measure, LOCATION, years, data_type, mocker)


@pytest.mark.xfail(raises=ModuleNotFoundError, reason="Cannot import core-maths", strict=True)
@pytest.mark.parametrize(
    "entity, measure",
    [
        [causes.diarrheal_diseases, "remission_rate"],
        [risk_factors.high_systolic_blood_pressure, "relative_risk"],
        [risk_factors.low_birth_weight_and_short_gestation, "relative_risk"],
    ],
)
def test_xfailed(
    entity: ModelableEntity,
    measure: str,
    mocker: MockerFixture,
):
    """We expect failures when trying to interpolate 'all' years for binned measures

    Notes
    -----
    These test parameterizations are a subset of others in this test module
    that are simply marked as 'skip'.
    """
    if measure == "relative_risk" and entity.name == "high_systolic_blood_pressure":
        pytest.skip("FIXME: [mic-5542] continuous rrs cannot validate")
    check_year_in_data(entity, measure, LOCATION, "all", "draws", mocker)


COVARIATES = [
    covariates.systolic_blood_pressure_mmhg,
]
COVARIATE_MEASURES = ["estimate"]


@pytest.mark.parametrize("entity", COVARIATES, ids=lambda x: x.name)
@pytest.mark.parametrize("measure", COVARIATE_MEASURES, ids=lambda x: x)
@pytest.mark.parametrize(
    "years", [None, 2019, 1900, [2019], [2019, 2020], "all"], ids=lambda x: str(x)
)
@pytest.mark.parametrize(
    "data_type", ["draws", "means", ["draws", "means"]], ids=lambda x: str(x)
)
def test_year_id_covariatelike(
    entity: ModelableEntity,
    measure: str,
    years: int | list[int] | str | None,
    data_type: str | list[str],
    mocker: MockerFixture,
):
    check_year_in_data(entity, measure, LOCATION, years, data_type, mocker)


@pytest.mark.parametrize("measure", ["structure", "demographic_dimensions"], ids=lambda x: x)
@pytest.mark.parametrize(
    "years", [None, 2019, 1900, [2019], [2019, 2020], "all"], ids=lambda x: str(x)
)
@pytest.mark.parametrize(
    "data_type", ["draws", "means", ["draws", "means"]], ids=lambda x: str(x)
)
def test_year_id_population(
    measure: str,
    years: int | list[int] | str | None,
    data_type: str | list[str],
    mocker: MockerFixture,
):
    pop = Population()
    check_year_in_data(pop, measure, LOCATION, years, data_type, mocker)


@pytest.mark.parametrize("entity_details", CAUSES, ids=lambda x: x[0].name)
@pytest.mark.parametrize("measure", CAUSE_MEASURES, ids=lambda x: x)
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
def test_multiple_locations_causelike(
    entity_details: tuple[ModelableEntity, list[str]],
    measure: str,
    locations: str | int | list[str | int],
    data_type: str | list[str],
    mocker: MockerFixture,
):
    year = MOST_RECENT_YEAR
    location_id_mapping = {
        "Ethiopia": 179,
        "Nigeria": 214,
    }
    entity, entity_expected_measures = entity_details
    if is_not_implemented(data_type, measure):
        with pytest.raises(DataTypeNotImplementedError):
            data_type = DataType(measure, data_type)
            mocker.patch(
                "vivarium_inputs.utility_data.get_raw_location_ids",
                return_value=get_mocked_location_ids(),
            )
            core.get_data(entity, measure, locations, year, data_type)
    else:
        if measure not in entity_expected_measures:
            with pytest.raises(Exception):
                data_type = DataType(measure, data_type)
                core.get_data(entity, measure, locations, year, data_type)
        else:
            mocked_funcs = mock_vivarium_gbd_access(
                entity, measure, locations, year, data_type, mocker
            )
            data_type = DataType(measure, data_type)
            df = core.get_data(entity, measure, locations, year, data_type)
            for mocked_func in mocked_funcs:
                assert mocked_func.called_once()
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


####################
# Helper functions #
####################


def check_year_in_data(
    entity: ModelableEntity,
    measure: str,
    location: str,
    years: int | list[int] | str | None,
    data_type: str | list[str],
    mocker: MockerFixture,
):
    if is_not_implemented(data_type, measure):
        with pytest.raises(DataTypeNotImplementedError):
            data_type = DataType(measure, data_type)
            mocker.patch(
                "vivarium_inputs.utility_data.get_raw_location_ids",
                return_value=get_mocked_location_ids(),
            )
            core.get_data(entity, measure, location, years, data_type)
    else:
        mocked_funcs = mock_vivarium_gbd_access(
            entity, measure, location, years, data_type, mocker
        )
        data_type = DataType(measure, data_type)
        if isinstance(years, list):
            df = core.get_data(entity, measure, location, years, data_type)
            assert set(df.index.get_level_values("year_id")) == set(years)
            for mocked_func in mocked_funcs:
                assert mocked_func.called_once()
        # years expected to be 1900, 2019, None, or "all"
        elif years != 1900:
            df = core.get_data(entity, measure, location, years, data_type)
            if years == None:
                assert set(df.index.get_level_values("year_id")) == set([MOST_RECENT_YEAR])
            elif years == "all":
                assert set(df.index.get_level_values("year_id")) == set(range(1990, 2023))
            else:  # a single (non-1900) year
                assert set(df.index.get_level_values("year_id")) == set([years])
            for mocked_func in mocked_funcs:
                assert mocked_func.called_once()
        else:
            with pytest.raises(ValueError, match="years must be in"):
                core.get_data(entity, measure, location, years, data_type)
