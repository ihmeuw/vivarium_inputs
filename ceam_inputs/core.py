"""This module performs the core data transformations on GBD data and provides a basic API for data access."""
from collections import defaultdict

import numpy as np
import pandas as pd

from ceam_inputs import gbd
from ceam_inputs.gbd_mapping.templates import cid, sid, UNKNOWN

# Define GBD sex ids for usage with central comp tools.
MALE = [1]
FEMALE = [2]
COMBINED = [3]

name_measure_map = {'prevalence': 5, 'incidence': 6, 'remission': 7, 'excess_mortality': 9, 'deaths': 1}


class GbdDataError(Exception):
    """Base exception for errors in GBD data."""
    pass


class DataMissingError(GbdDataError):
    """Exception raised when data has unhandled missing entries."""
    pass


class DuplicateDataError(GbdDataError):
    """Exception raised when data has duplication in the index."""


def get_ids_for_measure(entities, measures):
    """Selects the appropriate gbd id type for each entity and measure pair."""
    out = defaultdict(set)
    if 'deaths' in measures:
        for entity in entities:
            # We only have death estimates for causes
            assert isinstance(entity.gbd_id, cid)
            out['deaths'].add(entity.gbd_id)
    if 'remission' in measures:
        for entity in entities:
            # Remission estimates only appear for cause level dismod models.  I think.
            assert isinstance(entity.gbd_id, cid) and entity.dismod_id is not UNKNOWN
            out['remission'].add(entity.dismod_id)
    if 'prevalence' in measures:
        for entity in entities:
            # Prevalence can be pulled at the cause or sequela level.
            assert isinstance(entity.gbd_id, cid) or isinstance(entity.gbd_id, sid)
            # Only one id type at a time can be queried
            assert isinstance(entity.gbd_id, type(entities[0].gbd_id))
            out['prevalence'].add(entity.gbd_id)
    if 'incidence' in measures:
        for entity in entities:
            # Prevalence can be pulled at the cause or sequela level.
            assert isinstance(entity.gbd_id, cid) or isinstance(entity.gbd_id, sid)
            # Only one id type at a time can be queried
            assert isinstance(entity.gbd_id, type(entities[0].gbd_id))
            out['incidence'].add(entity.gbd_id)
    if 'exposure' in measures:
        for entity in entities:
            # exposure is pulled only with rei_ids
            assert isinstance(entity.gbd_id, rei_id)
            out['exposure'].add(entity.gbd_id)
    return out


def validate_data(data, key_columns=None):
    if np.any(data.isnull()):
        raise DataMissingError()

    if data.duplicated(key_columns).sum():
        raise DuplicateDataError()


def get_gbd_draws(entities, measures, location_ids, gbd_round_id):
    measure_entity_map = get_ids_for_measure(entities, measures)
    data = []
    if 'deaths' in measure_entity_map:
        death_data = gbd.get_codcorrect_draws(cause_ids=list(measure_entity_map['deaths']),
                                              location_ids=location_ids,
                                              gbd_round_id=gbd_round_id)
        death_data = death_data[death_data['measure_id'] == name_measure_map['deaths']]
        death_data['measure'] = 'deaths'
        data.append(death_data)
    if 'remission' in measures:
        remission_data = gbd.get_modelable_entity_draws(me_ids=list(measure_entity_map['remission']),
                                                        location_ids=location_ids,
                                                        gbd_round_id=gbd_round_id)
        remission_data = remission_data[remission_data['measure_id'] == name_measure_map['remission']]
        id_map = {entity.dismod_id: entity.gbd_id for entity in entities}
        remission_data['cause_id'] = remission_data['modelable_entity_id'].replace(id_map)
        remission_data['measure'] = 'remission'
        data.append(remission_data)

    if 'prevalence' in measures or 'incidence' in measures:
        ids = measure_entity_map['prevalence'].union(measure_entity_map['incidence'])
        measure_data = gbd.get_como_draws(entity_ids=list(ids),
                                          location_ids=location_ids,
                                          gbd_round_id=gbd_round_id)
        if 'prevalence' not in measures:
            measure_data = measure_data[measure_data['measure_id'] == name_measure_map['incidence']]
            measure_data['measure'] = 'prevalence'
        elif 'incidence' not in measures:
            measure_data = measure_data[measure_data['measure_id'] == name_measure_map['prevalence']]
            measure_data['measure'] = 'incidence'
        else:
            measure_data['measure'] = 'temp'
            measure_data.loc[measure_data['measure_id'] == name_measure_map['prevalence'], 'measure'] = 'prevalence'
            measure_data.loc[measure_data['measure_id'] == name_measure_map['incidence'], 'measure'] = 'incidence'

        data.append(measure_data)

    data = pd.concat(data)

    if data['measure'].str.contains('paf') or data['measure'].str.contains('rr'):
        id_cols = ['cause_id', 'risk_id']
    elif 'cause_id' in data.columns:
        id_cols = ['cause_id']
    elif 'sequela_id' in data.columns:
        id_cols = ['sequela_id']
    elif 'risk_id' in data.columns:
        id_cols = ['risk_id']
    else:
        raise GbdDataError('Stuff is broken')

    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'measure'] + id_cols
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]

    data = data[key_columns + draw_columns]
    validate_data(data, key_columns)

    return data.reset_index(drop=True)


def get_populations(location_ids, gbd_round_id):
    populations = pd.concat([gbd.get_populations(location_id, gbd_round_id) for location_id in location_ids])
    populations = populations[populations['sex_id'] != COMBINED]
    keep_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id', 'population']
    return populations[keep_columns]


def get_prevalences(entities, location_ids, gbd_round_id):
    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'gbd_id']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]
    return get_gbd_draws(entities, ['prevalence'], location_ids, gbd_round_id)[key_columns + draw_columns]


def get_incidences(entities, location_ids, gbd_round_id):
    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'gbd_id']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]
    return get_gbd_draws(entities, ['incidence'], location_ids, gbd_round_id)[key_columns + draw_columns]


def get_remissions(causes, location_ids, gbd_round_id):
    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'gbd_id']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]
    return get_gbd_draws(causes, ['remission'], location_ids, gbd_round_id)[key_columns + draw_columns]

def get_exposures(risks, location_ids, gbd_round_id):
    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'gbd_id', 'parameter']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]
    return get_gbd_draws(risks, ['exposure'], location_ids, gbd_round_id)[key_columns + draw_columns]

def get_cause_specific_mortalities(causes, location_ids, gbd_round_id):
    deaths = get_gbd_draws(causes, ["deaths"], location_ids, gbd_round_id)
    del deaths['measure_id']

    populations = get_populations(location_ids, gbd_round_id)
    populations = populations[populations['year_id'] >= deaths.year_id.min()]

    merge_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id']
    key_columns = merge_columns + ['gbd_id']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]

    df = deaths.merge(populations, on=merge_columns).set_index(key_columns)
    csmr = df[draw_columns].divide(df['population'], axis=0).reset_index()

    csmr = csmr[key_columns + draw_columns]
    validate_data(csmr, key_columns)

    return csmr


def get_excess_mortalities(causes, location_ids, gbd_round_id):
    prevalences = get_prevalences(causes, location_ids, gbd_round_id)
    csmrs = get_cause_specific_mortalities(causes, location_ids, gbd_round_id)

    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', "gbd_id"]
    prevalences = prevalences.set_index(key_columns)
    csmrs = csmrs.set_index(key_columns)

    em = csmrs.divide(prevalences, axis="index")

    return em.reset_index().dropna()


def get_proportion(entity_pairs, location_ids, gbd_round_id):
    prevalences = get_prevalences(list({entity for pair in entity_pairs for entity in pair}), location_ids, gbd_round_id)

    # FIXME: stand in for actual calculation
    proportions = pd.concat([prevalences[child]/prevalences[parent] for parent, child in entity_pairs])

    return proportions


def get_disability_weights(entities, location_ids, gbd_round_id):
    _gbd_round_id_map = {3: 'GBD_2015', 4: 'GBD_2016'}
    df = []

    # part 1: get disability weights data & construct df
    for each in entities:
        if isinstance(each, tuple):
            entities = list(each)

    for entity in entities:
        disability_weights = gbd.get_data_from_auxiliary_file('Disability Weights',
                                                              gbd_round=_gbd_round_id_map[gbd_round_id])
        combined_disability_weights = gbd.get_data_from_auxiliary_file('Combined Disability Weights',
                                                                       gbd_round=_gbd_round_id_map[gbd_round_id])

        gbd_ids = entity.gbd_id
        df_1 = disability_weights.loc[disability_weights.healthstate_id == gbd_ids].copy()
        df_2 = combined_disability_weights.loc[combined_disability_weights.healthstate_id == gbd_ids].copy()

        if not df_1.empty:
            df.append(df_1)

        if not df_2.empty:
            df.append(df_2)

    df = pd.concat(df, axis=0)
    df = df.rename(columns={'healthstate_id': 'gbd_id'})
    df = df.rename(columns={'healthstate': 'measure'})
    df["age"] = np.nan
    df["location_id"] = np.nan
    df["sex"] = np.nan
    df["year_id"] = np.nan

    del df["hhseqid"]

    return df.reset_index(drop=True)


def get_mediation_factors(risks, location_ids, gbd_round_id):
    ids = [risk.gbd_id for risk in risks]
    _gbd_round_id_map = {3: 'GBD_2015', 4: 'GBD_2016'}
    data = gbd.get_data_from_auxiliary_file("Mediation Factors", gbd_round=_gbd_round_id_map[gbd_round_id])
    df_list = []
    for id in ids:
        rf_df = data[data["rei_id"] == id]
        if not rf_df.empty:
            df_list.append(rf_df)

    if len(df_list) >= 1:
        data = pd.concat(df_list)
        draw_columns = [f'draw_{i}' for i in range(0, 1000)]
        data[draw_columns] = 1 - (data[draw_columns])
        data = data.groupby(["cause_id", 'rei_id'])[draw_columns].prod()
        data.reset_index(inplace=True)
        data = data.rename(columns={'rei_id': 'risk_id'})
        return data

    else:
        print("No mediation data found.")


