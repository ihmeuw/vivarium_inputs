import pandas as pd

from vivarium_inputs.globals import gbd


def get_estimation_years(*_, **__) -> pd.Series:
    data = gbd.get_estimation_years()
    return data


def get_year_block() -> pd.DataFrame:
    estimation_years = get_estimation_years()
    year_block = pd.DataFrame({'year_start': range(min(estimation_years), max(estimation_years) + 1)})
    year_block['year_end'] = estimation_years['year_start'] + 1
    return year_block
