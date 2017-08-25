# 1. set_age_year_index
from ceam_inputs.gbd_ms_auxiliary_functions import get_age_group_midpoint_from_age_group_id

from ceam_inputs.gbd_ms_auxiliary_functions import normalize_for_simulation
from ceam_inputs.gbd_ms_auxiliary_functions import expand_grid

import pandas as pd
import numpy as np


def test_normalize_for_simulation():
    df = pd.DataFrame({'sex_id': [1, 2], 'year_id': [1990, 1995]})
    df = normalize_for_simulation(df)

    assert df.columns.tolist() == ['year', 'sex'], 'normalize_for_simulation column names should be year and sex'
    assert df.sex.tolist() == ['Male', 'Female'], 'sex values should take categorical values Male, Female, or Both'


def test_get_age_group_midpoint_from_age_group_id():
    df = pd.DataFrame({'age_group_id': list(np.arange(2, 22))})
    df = get_age_group_midpoint_from_age_group_id(df, 3)
    assert 'age' in df.columns, "get_age_group_midpoint_from_age_group_id should create an age column"
    test_ages= ([(.01917808 / 2), ((0.01917808 + 0.07671233) / 2), ((0.07671233 + 1) / 2), 3]
                + [x for x in np.arange(7.5, 78, 5)] + [82.5])
    assert df.age.tolist() == test_ages, "get_age_group_midpoint_from_age_group_id should return age group midpoints for the gbd age groups"


def test_expand_grid():
    ages = pd.Series([0, 1, 2, 3, 4, 5])
    years = pd.Series([1990, 1991, 1992])

    df = expand_grid(ages, years)

    assert df.year_id.tolist() == np.repeat([1990, 1991, 1992], 6).tolist(), "expand_grid should expand a df to get row for each age/year combo"
    assert df.age.tolist() == [0, 1, 2, 3, 4, 5] + [0, 1, 2, 3, 4, 5] + [0, 1, 2, 3, 4, 5], "expand_grid should expand a df to get row for each age/year combo"
