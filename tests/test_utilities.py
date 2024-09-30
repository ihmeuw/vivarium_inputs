import re

import pandas as pd
import pytest

from vivarium_inputs import utilities
from vivarium_inputs.globals import DRAW_COLUMNS, MEAN_COLUMNS, SUPPORTED_DATA_TYPES


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
    "data_type_and_expected",
    [
        ("mean", "mean"),
        ("draw", "draw"),
        ("foo", "raises"),
        ("MEANS", "mean"),
        ("DRAWS", "draw"),
        (["mean", "draw"], ["mean", "draw"]),
        (["MEANS", "DRAWS"], ["mean", "draw"]),
        (["mean", "draw", "foo"], "raises"),
        ({"not": "a list"}, "raises"),
    ],
)
def test_process_data_type(data_type_and_expected):
    data_type, expected = data_type_and_expected
    if expected == "raises":
        if not isinstance(data_type, (list, str)):
            match = re.escape(
                f"'data_type' must be a string or a list of strings. Got {type(data_type)}."
            )
        else:
            match = re.escape(
                f"Data type 'foo' is not supported. Supported types are {list(SUPPORTED_DATA_TYPES)}."
            )
        with pytest.raises(ValueError, match=match):
            utilities.process_data_type(data_type)
    else:
        assert utilities.process_data_type(data_type) == expected


@pytest.mark.parametrize(
    "data_type_and_returned_cols",
    [
        ("mean", MEAN_COLUMNS),
        ("draw", DRAW_COLUMNS),
        (["mean", "draw"], MEAN_COLUMNS + DRAW_COLUMNS),
    ],
)
def test_get_value_columns(data_type_and_returned_cols):
    data_type, returned_cols = data_type_and_returned_cols
    assert utilities.get_value_columns(data_type) == returned_cols
