import pandas as pd
from gbd_mapping import RiskFactor

from vivarium_inputs.globals import NON_MAX_TMREL, NUM_DRAWS, SEXES, gbd


def get_estimation_years(*_, **__) -> pd.Series:
    return gbd.get_estimation_years()


def get_most_recent_year(*_, **__) -> int:
    return gbd.get_most_recent_year()


def get_year_block(*_, **__) -> pd.DataFrame:
    estimation_years = get_estimation_years()
    year_block = pd.DataFrame(
        {"year_start": range(min(estimation_years), max(estimation_years) + 1)}
    )
    year_block["year_end"] = year_block["year_start"] + 1
    return year_block


def get_age_group_ids(*_, **__) -> list[int]:
    return gbd.get_age_group_id()


def get_age_bins(*_, **__) -> pd.DataFrame:
    age_bins = get_raw_age_bins()[
        ["age_group_id", "age_group_name", "age_group_years_start", "age_group_years_end"]
    ].rename(columns={"age_group_years_start": "age_start", "age_group_years_end": "age_end"})
    age_bins = age_bins.sort_values("age_start").reset_index(drop=True)
    return age_bins


def get_raw_age_bins() -> pd.DataFrame:
    """Wrapper for gbd call for test mocking purposes."""
    return gbd.get_age_bins()


def get_location_id(location_name: str) -> int:
    return {r.location_name: r.location_id for _, r in get_raw_location_ids().iterrows()}[
        location_name
    ]


def get_location_name(location_id: int) -> str:
    return {
        row.location_id: row.location_name for _, row in get_raw_location_ids().iterrows()
    }[location_id]


def get_raw_location_ids() -> pd.DataFrame:
    """Wrapper for gbd call for test mocking purposes."""
    return gbd.get_location_ids()


def get_location_id_parents(location_id: int | list[int]) -> dict[int, list]:
    if isinstance(location_id, int):
        location_id = [location_id]
    location_metadata = get_location_path_to_global()
    parent_ids = (
        location_metadata.loc[location_metadata.location_id.isin(location_id)]
        .set_index("location_id")["path_to_top_parent"]
        .str.split(",")
        .to_dict()
    )
    # Coerce list of parent ids to integers
    parent_ids = {loc_id: list(map(int, parents)) for loc_id, parents in parent_ids.items()}

    return parent_ids


def get_location_path_to_global() -> pd.DataFrame:
    """Wrapper for gbd call for test mocking purposes."""
    return gbd.get_location_path_to_global()


def get_demographic_dimensions(
    location_id: int | list[int],
    draws: bool = False,
    value: float = None,
    years: int | str | list[int] | None = None,
) -> pd.DataFrame:
    ages = get_age_group_ids()
    estimation_years = get_estimation_years()
    if years is None:  # default to most recent year
        years = [get_most_recent_year()]
    elif years == "all":
        years = range(min(estimation_years), max(estimation_years) + 1)
    else:
        years = years if isinstance(years, list) else [years]
        not_estimated_years = set(years).difference(estimation_years)
        if not_estimated_years:
            raise ValueError(
                f"years must be in {estimation_years}. You provided {not_estimated_years}."
            )

    sexes = [SEXES["Male"], SEXES["Female"]]
    location = [location_id] if isinstance(location_id, int) else location_id
    values = [location, sexes, ages, years]
    names = ["location_id", "sex_id", "age_group_id", "year_id"]

    data = pd.MultiIndex.from_product(values, names=names).to_frame(index=False)
    if draws:
        for i in range(NUM_DRAWS):
            data[f"draw_{i}"] = value
    return data


def get_tmrel_category(entity: RiskFactor) -> str:
    if entity.name in NON_MAX_TMREL:
        tmrel_cat = NON_MAX_TMREL[entity.name]
    else:
        tmrel_cat = sorted(list(entity.categories.to_dict()), key=lambda x: int(x[3:]))[-1]
    return tmrel_cat
