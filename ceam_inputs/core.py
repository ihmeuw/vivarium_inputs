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
            out['prevalence'].add(entity.gbd_id)
    if 'incidence' in measures:
        for entity in entities:
            # Prevalence can be pulled at the cause or sequela level.
            assert isinstance(entity.gbd_id, cid) or isinstance(entity.gbd_id, sid)
            out['incidence'].add(entity.gbd_id)
    return out


def validate_data(data, key_columns=None):
    if np.any(data.isnull()):
        raise DataMissingError()

    if data.duplicated(key_columns).sum():
        raise DuplicateDataError()


def get_gbd_draws(entities, measures, location_ids, gbd_round_id):
    measure_entity_map = get_ids_for_measure(entities, measures)
    data = pd.DataFrame()
    if 'deaths' in measure_entity_map:
        death_data = gbd.get_codcorrect_draws(cause_ids=list(measure_entity_map['deaths']),
                                              location_ids=location_ids,
                                              gbd_round_id=gbd_round_id)
        # do processing to clean up data, add measure column if needed
        data = data.append(death_data)
        data = data.rename(columns={'cause_id': 'gbd_id'})
        data = data[data['sex_id'] != COMBINED]
        data = data[data['measure_id'] == name_measure_map['deaths']]
        data['measure'] = 'deaths'

    if 'remission' in measures:
        remission_data = gbd.get_modelable_entity_draws(me_ids=list(measure_entity_map['remission']),
                                                        location_ids=location_ids,
                                                        gbd_round_id=gbd_round_id)
        # do processing to clean up data, add measure column if needed
        data = data.append(remission_data)
        data = data.rename(columns={'modelable_entity_id': 'gbd_id'})
        data = data[data['measure_id'] == name_measure_map['remission']]
        data['measure'] = 'remission'

    if 'prevalence' in measures or 'incidence' in measures:
        ids = measure_entity_map['prevalence'].copy().union(measure_entity_map['incidence'])
        measure_data = gbd.get_como_draws(entity_ids=list(ids),
                                          location_ids=location_ids,
                                          gbd_round_id=gbd_round_id)
        # do processing to clean up data, add measure column if needed
        if 'prevalence' not in measures:  # if incidence
            measure_data = measure_data[measure_data.measure_id != name_measure_map['prevalence']]
            measure_data = measure_data[measure_data['measure_id'] == name_measure_map['incidence']]
            measure_data['measure']= 'incidence'

        if 'incidence' not in measures: # if prevalence
            measure_data = measure_data[measure_data.measure_id != name_measure_map['incidence']]
            measure_data = measure_data[measure_data['measure_id'] == name_measure_map['prevalence']]
            measure_data['measure'] = 'prevalence'

        data = data.append(measure_data)
        data = data.rename(columns={'cause_id': 'gbd_id'})
        data = data.rename(columns={'sequela_id': 'gbd_id'})

        del data['measure_id']
        if 'output_version_id' in data.columns:
            del data['output_version_id']

    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'gbd_id', 'measure']
    draw_columns = ['draw_{}'.format(i) for i in range(0, 1000)]

    data = data[key_columns + draw_columns].sort_values(by=key_columns)
    validate_data(data, key_columns)

    return data.reset_index(drop=True)


def get_populations(location_ids, gbd_round_id):
    populations = pd.concat([gbd.get_populations(location_id, gbd_round_id) for location_id in location_ids])
    populations = populations[populations['sex_id'] != COMBINED]
    keep_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id', 'population']
    return populations[keep_columns]


def get_prevalence(entities, location_ids, gbd_round_id):
    return get_gbd_draws(entities, ['prevalence'], location_ids, gbd_round_id)


def get_incidence(entities, location_ids, gbd_round_id):
    return get_gbd_draws(entities, ['incidence'], location_ids, gbd_round_id)


def get_remission(causes, location_ids, gbd_round_id):
    return get_gbd_draws(causes, ['remission'], location_ids, gbd_round_id)


def get_cause_specific_mortality(causes, location_ids, gbd_round_id):
    deaths = get_gbd_draws(causes, ["deaths"], location_ids, gbd_round_id)

    populations = get_populations(location_ids, gbd_round_id)
    populations = populations[populations['year_id'] >= deaths.year_id.min()]

    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id']
    populations = populations.sort_values(by=key_columns)
    deaths = deaths.set_index(key_columns)
    populations = populations.set_index(key_columns)
    del deaths["measure"]

    csmr = pd.DataFrame()

    # when given more than one cause:
    for each in deaths.gbd_id.unique():
        temp_df = deaths[deaths.gbd_id == each]
        del temp_df["gbd_id"]
        temp_df = temp_df.divide(populations.population, axis="index")
        temp_df["gbd_id"] = each
        csmr = pd.concat([csmr, temp_df])

    return csmr.reset_index()


def get_excess_mortality(causes, location_ids, gbd_round_id):
    prevalences = get_prevalence(causes, location_ids, gbd_round_id)
    csmrs = get_cause_specific_mortality(causes, location_ids, gbd_round_id)
    csmrs = csmrs[csmrs['year_id'] >= prevalences.year_id.min()]

    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', "gbd_id"]
    prevalences = prevalences.set_index(key_columns)
    csmrs = csmrs.set_index(key_columns)

    del prevalences["measure"]

    em = csmrs.divide(prevalences, axis="index")

    return em.reset_index()


def get_proportion(entity_pairs, location_ids, gbd_round_id):
    prevalences = get_prevalence(list({entity for pair in entity_pairs for entity in pair}), location_ids, gbd_round_id)

    # FIXME: stand in for actual calculation
    proportions = pd.concat([prevalences[child]/prevalences[parent] for parent, child in entity_pairs])

    return proportions
