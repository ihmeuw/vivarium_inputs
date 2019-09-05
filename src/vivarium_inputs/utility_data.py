from typing import List

import pandas as pd

from vivarium_inputs.globals import gbd, SEXES, NON_MAX_TMREL
from gbd_mapping import RiskFactor


def get_estimation_years(*_, **__) -> pd.Series:
    data = gbd.get_estimation_years()
    return data


def get_year_block(*_, **__) -> pd.DataFrame:
    estimation_years = get_estimation_years()
    year_block = pd.DataFrame({'year_start': range(min(estimation_years), max(estimation_years) + 1)})
    year_block['year_end'] = year_block['year_start'] + 1
    return year_block


def get_age_group_ids(*_, **__) -> List[int]:
    data = gbd.get_age_group_id()
    return data


def get_age_bins(*_, **__) -> pd.DataFrame:
    age_bins = (
        gbd.get_age_bins()[['age_group_id', 'age_group_name', 'age_group_years_start', 'age_group_years_end']]
            .rename(columns={'age_group_years_start': 'age_start',
                             'age_group_years_end': 'age_end'})
    )
    return age_bins


def get_location_id(location_name):
    return {r.location_name: r.location_id for _, r in gbd.get_location_ids().iterrows()}[location_name]


def get_location_id_parents(location_id: int) -> List[int]:
    location_metadata = gbd.get_location_path_to_global().set_index('location_id')
    parent_ids = [int(loc) for loc in location_metadata.at[location_id, 'path_to_top_parent'].split(',')]
    return parent_ids


def get_demographic_dimensions(location_id: int, draws: bool = False, value: float = None) -> pd.DataFrame:
    ages = get_age_group_ids()
    estimation_years = get_estimation_years()
    years = range(min(estimation_years), max(estimation_years) + 1)
    sexes = [SEXES['Male'], SEXES['Female']]
    location = [location_id]
    values = [location, sexes, ages, years]
    names = ['location_id', 'sex_id', 'age_group_id', 'year_id']

    data = (pd.MultiIndex
            .from_product(values, names=names)
            .to_frame(index=False))
    if draws:
        for i in range(1000):
            data[f'draw_{i}'] = value
    return data


def get_tmrel_category(entity: RiskFactor) -> str:
    if entity.name in NON_MAX_TMREL:
        tmrel_cat = NON_MAX_TMREL[entity.name]
    else:
        tmrel_cat = sorted(list(entity.categories.to_dict()), key=lambda x: int(x[3:]))[-1]
    return tmrel_cat
