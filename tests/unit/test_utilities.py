import re

import pandas as pd
import pytest

from vivarium_inputs import utilities
from vivarium_inputs.globals import (
    DRAW_COLUMNS,
    MEAN_COLUMNS,
    NON_STANDARD_MEASURES,
    SUPPORTED_DATA_TYPES,
)

POSSIBLE_MEASURES = [
    "incidence_rate",
    "raw_incidence_rate",
    "prevalence",
    "birth_prevalence",
    "disability_weight",
    "remission_rate",
    "cause_specific_mortality_rate",
    "excess_mortality_rate",
    "deaths",
    # Risk-like measures
    "exposure",
    "exposure_standard_deviation",
    "exposure_distribution_weights",
    "relative_risk",
    "population_attributable_fraction",
    # Covariate measures
    "estimate",
    # Population measures
    "structure",
    "theoretical_minimum_risk_life_expectancy",
    "age_bins",
    "demographic_dimensions",
]


@pytest.mark.parametrize(
    "sex_ids",
    [(1, 1, 1, 2, 2, 2), (1, 1, 2, 2, 3, 3), (1, 1, 1), (2, 2, 2), (3, 3, 3)],
    ids=["male_female", "male_female_both", "male", "female", "both"],
)
def test_normalize_sex(sex_ids):
    df = pd.DataFrame({"sex_id": sex_ids, "value": [1] * len(sex_ids)})
    normalized = utilities.normalize_sex(df, fill_value=0.0, cols_to_fill=["value"])
    assert {1, 2} == set(normalized.sex_id)


def test_normalize_sex_copy_3():
    values = [1, 2, 3, 4]
    df = pd.DataFrame({"sex_id": [3] * len(values), "value": values})
    normalized = utilities.normalize_sex(df, fill_value=0.0, cols_to_fill=["value"])
    assert (normalized.loc[normalized.sex_id == 1, "value"] == values).all()
    assert (normalized.loc[normalized.sex_id == 2, "value"] == values).all()


def test_normalize_sex_fill_value():
    values = [1, 2, 3, 4]
    fill = 0.0
    for sex in [1, 2]:
        missing_sex = 1 if sex == 2 else 2
        df = pd.DataFrame({"sex_id": [sex] * len(values), "value": values})
        normalized = utilities.normalize_sex(df, fill_value=fill, cols_to_fill=["value"])
        assert (normalized.loc[normalized.sex_id == sex, "value"] == values).all()
        assert (
            normalized.loc[normalized.sex_id == missing_sex, "value"] == [fill] * len(values)
        ).all()


def test_normalize_sex_no_sex_id():
    df = pd.DataFrame({"ColumnA": [1, 2, 3], "ColumnB": [1, 2, 3]})
    normalized = utilities.normalize_sex(df, fill_value=0.0, cols_to_fill=["value"])
    pd.testing.assert_frame_equal(df, normalized)


@pytest.mark.parametrize(
    "data_type, should_raise",
    [
        ("means", False),
        ("draws", False),
        ("foo", True),
        (["means", "draws"], True),  # temporarily; not implemented
        (["means", "draws", "foo"], True),
        ({"not": "a list"}, True),
    ],
)
def test_validate_data_type(data_type, should_raise):
    if should_raise:
        if not isinstance(data_type, (list, str)):
            match = re.escape(
                f"'data_type' must be a string or a list of strings. Got {type(data_type)}."
            )
        elif isinstance(data_type, list):
            match = "Lists of data types are not yet supported."
        else:
            match = re.escape(
                f"Data type(s) {set(['foo'])} are not supported. Supported types are {list(SUPPORTED_DATA_TYPES)}."
            )
        with pytest.raises(utilities.DataTypeNotImplementedError, match=match):
            utilities.DataType("foo", data_type)
    else:
        utilities.DataType("foo", data_type)


@pytest.mark.parametrize("measure", POSSIBLE_MEASURES, ids=lambda x: x)
@pytest.mark.parametrize(
    "data_type",
    ("means", "draws", None, ["means", "draws"]),
    ids=["means", "draws", "None", "means_draws"],
)
def test_get_value_columns(measure, data_type):
    if isinstance(data_type, list):
        with pytest.raises(
            utilities.DataTypeNotImplementedError,
            match="Lists of data types are not yet supported.",
        ):
            utilities.DataType(measure, data_type).value_columns
    else:
        if data_type == None:  # Hacky: this goes first in the DataType class
            expected_returned_cols = []
        elif measure in NON_STANDARD_MEASURES:
            expected_returned_cols = ["value"]
        elif data_type == "means":
            expected_returned_cols = MEAN_COLUMNS
        else:  # draws
            expected_returned_cols = DRAW_COLUMNS
        assert utilities.DataType(measure, data_type).value_columns == expected_returned_cols
