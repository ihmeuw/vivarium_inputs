import pandas as pd

from vivarium_inputs.utilities import normalize_for_simulation, get_age_group_midpoint_from_age_group_id
from vivarium_inputs import core


def normalize(data: pd.DataFrame) -> pd.DataFrame:
    """Remove GBD specific column names and concepts and make the dataframe long over draws."""
    assert not data.empty
    data = normalize_for_simulation(data)
    if "age_group_id" in data:
        data = get_age_group_midpoint_from_age_group_id(data)
    draw_columns = [c for c in data.columns if "draw_" in c]
    index_columns = [c for c in data.columns if "draw_" not in c]
    data = pd.melt(data, id_vars=index_columns, value_vars=draw_columns, var_name="draw")
    data["draw"] = data.draw.str.partition("_")[2].astype(int)

    estimation_years = core.get_estimation_years()
    year_start, year_end = min(estimation_years), max(estimation_years)
    if "year" in data:
        data = data.loc[(data.year >= year_start) & (data.year <= year_end)]
    return data
