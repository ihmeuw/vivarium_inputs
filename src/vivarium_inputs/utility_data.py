from typing import List

import pandas as pd

from vivarium_inputs.globals import gbd


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
            .rename(columns={'age_group_years_start': 'age_group_start',
                             'age_group_years_end': 'age_group_end'})
    )
    return age_bins
