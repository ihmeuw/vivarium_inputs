import pytest

import pandas as pd

from vivarium_inputs import utilities


@pytest.mark.parametrize("sex_ids", [
    (1, 1, 1, 2, 2, 2),
    (1, 1, 2, 2, 3, 3),
    (1, 1, 1),
    (2, 2, 2),
    (3, 3, 3)
], ids=['male_female', 'male_female_both', 'male', 'female', 'both'])
def test_normalize_sex(sex_ids):
    df = pd.DataFrame({'sex_id': sex_ids, 'value': [1] * len(sex_ids)})
    normalized = utilities.normalize_sex(df, fill_value=0.0, cols_to_fill=['value'])
    assert {1, 2} == set(normalized.sex_id)


def test_normalize_sex_copy_3():
    values = [1, 2, 3, 4]
    df = pd.DataFrame({'sex_id': [3] * len(values), 'value': values})
    normalized = utilities.normalize_sex(df, fill_value=0.0, cols_to_fill=['value'])
    assert (normalized.loc[normalized.sex_id == 1, 'value'] == values).all()
    assert (normalized.loc[normalized.sex_id == 2, 'value'] == values).all()


def test_normalize_sex_fill_value():
    values = [1, 2, 3, 4]
    fill = 0.0
    for sex in [1, 2]:
        missing_sex = 1 if sex == 2 else 2
        df = pd.DataFrame({'sex_id': [sex] * len(values), 'value': values})
        normalized = utilities.normalize_sex(df, fill_value=fill, cols_to_fill=['value'])
        assert (normalized.loc[normalized.sex_id == sex, 'value'] == values).all()
        assert (normalized.loc[normalized.sex_id == missing_sex, 'value'] == [fill] * len(values)).all()


def test_normalize_sex_no_sex_id():
    df = pd.DataFrame({"ColumnA": [1, 2, 3], "ColumnB": [1, 2, 3]})
    normalized = utilities.normalize_sex(df, fill_value=0.0, cols_to_fill=['value'])
    pd.testing.assert_frame_equal(df, normalized)

