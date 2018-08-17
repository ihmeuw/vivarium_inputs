import pandas as pd

from vivarium_inputs.utilities import normalize_for_simulation, get_age_group_midpoint_from_age_group_id


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
    return data
