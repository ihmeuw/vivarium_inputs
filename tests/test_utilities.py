import pytest

import pandas as pd

from vivarium_inputs import utilities


# assert no sex 3
# assert sex 1 and 2 present
@pytest.mark.parametrize("sex_ids", [
    (1, 1, 1, 2, 2, 2),
    (1, 1, 2, 2, 3, 3),
    (1, 1, 1),
    (2, 2, 2),
    (3, 3, 3)
], ids=['male_female', 'male_female_both', 'male', 'female', 'both'])
def test_normalize_sex(sex_ids):
    df = pd.DataFrame({'sex_id': sex_ids, 'value': [1] * len(sex_ids)})

    normalized = utilities.normalize_sex(df, fill_value=0.0)

    assert {1, 2} == set(normalized.sex_id)


def test_normalize sex_fill(sex_ids):
    pass



# assert dataframe is unchanged
def test_normalize_sex_no_sex_id():
    pass


# # get bins, add and subtract years from
# def test_normalize_year_annual():
#     pass
#


# test age 22 passed in, results in all ages
# test age deficient, fills in ages
# test extra ages, should subset like year?
def test_normalize_age():
    pass


# assert dataframe is unchanged
def test_normalize_age_no_age_id():
    pass


def test_reshape():
    pass
