"""Module containing functions that standardize the format of GBD outputs."""
from typing import Mapping, Iterable
from itertools import chain

import pandas as pd
import numpy as np
from gbd_mapping import (causes, risk_factors, etiologies, coverage_gaps,
                         Cause, Risk, Sequela, CoverageGap, Etiology, ModelableEntity)
from gbd_mapping.id import scalar, UNKNOWN

from vivarium_inputs.mapping_extension import HealthcareEntity, HealthTechnology

try:
    from vivarium_gbd_access import gbd
except ModuleNotFoundError:
    class GbdDummy:
        def __getattr__(self, item):
            raise ModuleNotFoundError("Required package vivarium_gbd_access not found.")
    gbd = GbdDummy()

GBD_ROUND_ID_MAP = {3: 'GBD_2015', 4: 'GBD_2016'}


class DataError(Exception):
    """Base exception for errors in data loading."""
    pass


class InvalidQueryError(DataError):
    """Exception raised when the user makes an invalid request for data (e.g. exposures for a sequela)."""
    pass


class UnhandledDataError(DataError):
    """Exception raised when we receive data from the databases that we don't know how to handle."""
    pass


class DataMissingError(DataError):
    """Exception raised when data has unhandled missing entries."""
    pass


class DuplicateDataError(DataError):
    """Exception raised when data has duplication in the index."""
    pass


def select_draw_data(data, draw, column_name, src_column=None):
    if column_name:
        if src_column is not None:
            if isinstance(src_column, str):
                column_map = {src_column.format(draw=draw): column_name}
            else:
                column_map = {src.format(draw=draw): dest for src, dest in zip(src_column, column_name)}
        else:
            column_map = {'draw_{draw}'.format(draw=draw): column_name}

        # if 'measure' is in columns, then keep it, else do
        # not keep it (need measure for the relative risk estimations)
        if 'parameter' in data.columns:
            keep_columns = ['year_id', 'age_group_start', 'age_group_end',
                            'sex_id', 'parameter'] + list(column_map.keys())
        else:
            keep_columns = ['year_id', 'age_group_start', 'age_group_end', 'sex_id'] + list(column_map.keys())

        data = data[keep_columns]
        data = data.rename(columns=column_map)

        return normalize_for_simulation(data)
    return data


def normalize_for_simulation(df):
    """
    Parameters
    ----------
    df : DataFrame
        dataframe to change

    Returns
    -------
    Returns a df with column year_id changed to year, and year_start and year_end
    created as bin ends around year_id with year_start set to year_id;
    sex_id changed to sex, and sex values changed from 1 and 2 to Male and Female

    Notes
    -----
    Used by -- load_data_from_cache

    Assumptions -- None

    Questions -- None

    Unit test in place? -- Yes
    """
    if "sex_id" in df:
        if set(df["sex_id"]) == {3}:
            df_m = df.copy()
            df_f = df.copy()
            df_m['sex'] = 'Male'
            df_f['sex'] = 'Female'
            df = pd.concat([df_m, df_f], ignore_index=True)
        else:
            df["sex"] = df.sex_id.map({1: "Male", 2: "Female", 3: "Both"}).astype(
                pd.api.types.CategoricalDtype(categories=["Male", "Female", "Both"], ordered=False))

        df = df.drop("sex_id", axis=1)

    if "year_id" in df:
        # FIXME: use central comp interpolation tools
        if 2006 in df.year_id.unique() and 2007 not in df.year_id.unique():
            df = df.loc[(df.year_id != 2006)]

        df = df.rename(columns={"year_id": "year_start"})
        idx = df.index

        mapping = df[['year_start']].drop_duplicates().sort_values(by="year_start")
        mapping['year_end'] = mapping['year_start'].shift(-1).fillna(mapping.year_start.max()+1).astype(int)

        df = df.set_index("year_start", drop=False)
        mapping = mapping.set_index("year_start", drop=False)

        df[["year_start", "year_end"]] = mapping[["year_start", "year_end"]]

        df = df.set_index(idx)

    return df


def get_age_group_bins_from_age_group_id(df):
    """Creates "age_group_start" and "age_group_end" columns from the "age_group_id" column

    Parameters
    ----------
    df: df for which you want an age column that has an age_group_id column

    Returns
    -------
    df with "age_group_start" and "age_group_end" columns
    """
    if df.empty:
        df['age_group_start'] = 0
        df['age_group_end'] = 0
        return df

    df = df.copy()
    idx = df.index
    mapping = gbd.get_age_bins()
    mapping = mapping.set_index('age_group_id')

    df = df.set_index('age_group_id')
    df[['age_group_start', 'age_group_end']] = mapping[['age_group_years_start', 'age_group_years_end']]

    df = df.set_index(idx)

    return df


def get_id_for_measure(entity: ModelableEntity, measure: str) -> int:
    """Selects the appropriate gbd id type for each entity and measure pair.

    Parameters
    ----------
    entity :
        A data containers from the `gbd_mapping` package.
    measure :
        A GBD measures requested for the provided entity.

    Returns
    -------
    The appropriate GBD id for use with central comp tools for the
    provided entity and measure.

    Raises
    ------
    InvalidQueryError
        If the entities passed are inconsistent with the requested measures.
    """
    measure_types = {
        'death': (Cause, 'gbd_id'),
        'prevalence': ((Cause, Sequela), 'gbd_id'),
        'incidence': ((Cause, Sequela), 'gbd_id'),
        'exposure': ((Risk, CoverageGap), 'gbd_id'),
        'exposure_standard_deviation': ((Risk, CoverageGap), 'gbd_id'),
        'relative_risk': ((Risk, CoverageGap), 'gbd_id'),
        'population_attributable_fraction': ((Risk, Etiology, CoverageGap), 'gbd_id'),
        'cause_specific_mortality': ((Cause,), 'gbd_id'),
        'excess_mortality': ((Cause,), 'gbd_id'),
        'annual_visits': (HealthcareEntity, 'utilization'),
        'disability_weight': (Sequela, 'gbd_id'),
        'remission': (Cause, 'dismod_id'),
        'cost': ((HealthcareEntity, HealthTechnology), 'cost'),
    }

    valid_types, id_attr = measure_types.get(measure, (None, None))

    if not valid_types or not id_attr:
        raise InvalidQueryError(f"You've requested an invalid measure: {measure}")
    elif not isinstance(entity, valid_types) or entity[id_attr] is UNKNOWN:
        raise InvalidQueryError(f"Entity {entity.name} has no data for measure '{measure}'")
    else:
        measure_id = entity[id_attr]

    return measure_id


def get_additional_id_columns(data, entity):
    id_column_map = {
        'cause': 'cause_id',
        'sequela': 'sequela_id',
        'covariate': 'covariate_id',
        'risk_factor': 'rei_id',
        'etiology': 'rei_id',
        'coverage_gap': 'coverage_gap',
        'healthcare_entity': 'healthcare_entity',
        'health_technology': 'health_technology',
    }
    out = {id_column_map[entity.kind]}
    if entity.kind == 'coverage_gap' and 'relative_risk' in data['measure'].unique():
        out.add('rei_id')
    # TODO: Why?
    out |= set(data.columns) & set(id_column_map.values())
    return out


def validate_data(data: pd.DataFrame, key_columns: Iterable[str]=None):
    """Validates that no data is missing and that the provided key columns make a valid (unique) index.

    Parameters
    ----------
    data:
        The data table to be validated.
    key_columns:
        An iterable of the column names used to uniquely identify a row in the data table.

    Raises
    ------
    DataMissingError
        If the data contains any null (NaN or NaT) values.
    DuplicatedDataError
        If the provided key columns are insufficient to uniquely identify a record in the data table.
    """

    #  check draw_cols only since we may have nan in coverage_gap data from aux_data
    draw_cols = [f'draw_{i}' for i in range(1000)]
    if np.any(data[draw_cols].isnull()):
        raise DataMissingError()

    if key_columns and np.any(data.duplicated(key_columns)):
        raise DuplicateDataError()


def get_age_group_ids(restrictions, measure='yld'):
    assert measure in ['yld', 'yll']

    age_restriction_map = {scalar(0.0): [2, None],
                           scalar(0.01): [3, 2],
                           scalar(0.10): [4, 3],
                           scalar(1.0): [5, 4],
                           scalar(5.0): [6, 5],
                           scalar(10.0): [7, 6],
                           scalar(15.0): [8, 7],
                           scalar(20.0): [9, 8],
                           scalar(30.0): [11, 10],
                           scalar(40.0): [13, 12],
                           scalar(45.0): [14, 13],
                           scalar(50.0): [15, 14],
                           scalar(55.0): [16, 15],
                           scalar(65.0): [18, 17],
                           scalar(95.0): [235, 32], }

    age_start, age_end = restrictions[f'{measure}_age_start'], restrictions[f'{measure}_age_end']
    min_age_group = age_restriction_map[age_start][0]
    max_age_group = age_restriction_map[age_end][1]
    return list(range(min_age_group, max_age_group + 1))


def get_measure_id(measure):
    name_measure_map = {'death': 1,
                        'DALY': 2,
                        'YLD': 3,
                        'YLL': 4,
                        'prevalence': 5,
                        'incidence': 6,
                        'remission': 7,
                        'excess_mortality': 9,
                        'proportion': 18,
                        'continuous': 19, }
    return name_measure_map[measure]


def standardize_all_age_groups(data: pd.DataFrame):
    if set(data.age_group_id) != {22}:
        return data
    whole_age_groups = gbd.get_age_group_id()
    missing_age_groups = set(whole_age_groups).difference(set(data.age_group_id))
    df = []
    for i in missing_age_groups:
        missing = data.copy()
        missing['age_group_id'] = i
        df.append(missing)

    return pd.concat(df, ignore_index=True)


def standardize_data(data: pd.DataFrame, fill_value: int) -> pd.DataFrame:
    # age_groups that we expect to exist for each risk
    whole_age_groups = gbd.get_age_group_id()
    sex_id = data.sex_id.unique()
    year_id = data.year_id.unique()
    location_id = data.location_id.unique()

    index_cols = ['year_id', 'location_id', 'sex_id', 'age_group_id']
    draw_cols = [c for c in data.columns if 'draw_' in c]

    other_cols = {c: data[c].unique() for c in data.dropna(axis=1).columns if c not in index_cols and c not in draw_cols}
    index_cols += [*other_cols.keys()]
    data = data.set_index(index_cols)

    # expected indexes to be in the data
    expected = pd.MultiIndex.from_product([year_id, location_id, sex_id, whole_age_groups] + [*other_cols.values()],
                                          names=(['year_id', 'location_id', 'sex_id', 'age_group_id'] + [
                                              *other_cols.keys()]))

    new_data = data.copy()
    missing = expected.difference(data.index)

    # assign dtype=float to prevent the artifact error with mixed dtypes
    to_add = pd.DataFrame({column: fill_value for column in draw_cols}, index=missing, dtype=float)

    new_data = new_data.append(to_add).sort_index()

    return new_data.reset_index()


def filter_to_most_detailed(data):
    for column, entity_list in [('cause_id', causes),
                                ('rei_id', chain(etiologies, risk_factors)),
                                ('coverage_gap_id', coverage_gaps)]:
        most_detailed = {e.gbd_id for e in entity_list if e is not None}
        if column in data:
            data = data[data[column].isin(most_detailed)]
    return data
