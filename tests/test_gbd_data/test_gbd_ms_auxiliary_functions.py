# 1. set_age_year_index
from ceam_inputs.gbd_ms_auxiliary_functions import get_age_group_midpoint_from_age_group_id
from ceam_inputs.gbd_ms_auxiliary_functions import create_age_column
from ceam_inputs.gbd_ms_auxiliary_functions import get_populations
from ceam_inputs.gbd_ms_auxiliary_functions import normalize_for_simulation
from ceam_inputs.gbd_ms_auxiliary_functions import expand_grid
from ceam_inputs.gbd_ms_auxiliary_functions import assign_sex_id
from ceam_inputs.gbd_ms_auxiliary_functions import get_healthstate_id
from scipy import stats
from ceam_tests.util import setup_simulation, generate_test_population
from ceam_public_health.components.diarrhea_disease_model import diarrhea_factory
from ceam_public_health.components.risks.categorical_risk_handler import CategoricalRiskHandler
from datetime import datetime
import pandas as pd
import numpy as np


# 1. create_age_column
def test_create_age_column():

    pop = pd.DataFrame({'age': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89], 'proportion_of_total_pop': [.1]*8 + [.15]*1 + [.05]})
    simulants = pd.DataFrame({'simulant_id': range(0, 500000)})
    simulants = create_age_column(simulants, pop, 500000)
    simulants['count'] = 1
    simulants = simulants[['age', 'count']].groupby('age').sum()
    simulants['proportion'] = simulants['count'] / 500000

    # now check that the output proportions are close to the input proportions (em arbitrarily picked .1)
    # might need to revamp py test to allow for periodic transient failures since this function is based on randomness
    assert np.allclose(simulants.proportion.values, pop.proportion_of_total_pop.values, .1), 'create_age_column needs to assign ages according to input proportions'


# 2. normalize_for_simulation
def test_normalize_for_simulation():
    df = pd.DataFrame({'sex_id': [1, 2], 'year_id': [1990, 1995]})
    df = normalize_for_simulation(df)

    assert df.columns.tolist() == ['year', 'sex'], 'normalize_for_simulation column names should be year and sex'
    assert df.sex.tolist() == ['Male', 'Female'], 'sex values should take categorical values Male, Female, or Both'


# 3. get_age_group_midpoint_from_age_group_id
def test_get_age_group_midpoint_from_age_group_id():
    df = pd.DataFrame({'age_group_id': list(np.arange(2, 22)) + [235]})
    df = get_age_group_midpoint_from_age_group_id(df)
    assert 'age' in df.columns, "get_age_group_midpoint_from_age_group_id should create an age column"
    assert df.age.tolist() == [(.01917808/2), ((0.01917808+0.07671233)/2), ((0.07671233+1)/2), 3] + [x for x in np.arange(7.5, 78, 5)] + [82.5] + [97.5], "get_age_group_midpoint_from_age_group_id should return age group midpoints for the gbd age groups"


# 4. assign_sex_id
# create fake populations of men/women and assign sex id while making sure it's correlated with age
def test_assign_sex_id():
    male_pop = pd.DataFrame({'age': [0, 5, 10, 15, 20], 'pop_scaled': [0, 25000, 50000, 75000, 10000]})
    female_pop = pd.DataFrame({'age': [0, 5, 10, 15, 20], 'pop_scaled': [100000, 75000, 50000, 25000, 0]})

    simulants = pd.DataFrame({'simulant_id': range(0, 500000), 'age': np.repeat([0, 5, 10, 15, 20], 100000)})

    df = assign_sex_id(simulants, male_pop, female_pop)

    # age 0 should be all women, 5 should be 75%, 10 should be 50%, and so on
    df['count'] = 1
    grouped = df[['sex_id', 'age', 'count']].groupby(['sex_id', 'age']).sum()
    grouped['proportion'] = grouped['count'] / 100000

    assert np.allclose(grouped.proportion.tolist(), [x for x in np.arange(.25, 1.25, .25)] + [x for x in np.arange(0, 1.25, .25)][4:0:-1], .1), 'assign_sex_id needs to assign sexes so that they are correlated with age'


# 5. get_healthstate_id
def test_get_healthstate_id():
    # modelable entity id 1823 = severe heart failure
    val = get_healthstate_id(dis_weight_modelable_entity_id=1823)

    assert val == 383, "modelable entity id 1823 should have a healthstate of 383 as of 9/30"


#6. expand_grid
def test_expand_grid():
    ages = pd.Series([0, 1, 2, 3, 4, 5])
    years = pd.Series([1990, 1991, 1992])

    df = expand_grid(ages, years)
    
    assert df.year_id.tolist() == np.repeat([1990, 1991, 1992], 6).tolist(), "expand_grid should expand a df to get row for each age/year combo"
    assert df.age.tolist() == [0, 1, 2, 3, 4, 5] + [0, 1, 2, 3, 4, 5] + [0, 1, 2, 3, 4, 5], "expand_grid should expand a df to get row for each age/year combo"
