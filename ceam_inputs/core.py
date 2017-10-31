"""This module performs the core data transformations on GBD data and provides a basic API for data access."""
from collections import defaultdict

import numpy as np
import pandas as pd

from ceam_inputs import gbd
from ceam_inputs.gbd_mapping.templates import cid, sid, UNKNOWN

name_measure_map = {'prevalence': 5, 'incidence': 6, 'remission': 7, 'excess_mortality': 9, 'deaths': 1}


class GbdDataError(Exception):
    """Base exception for errors in GBD data."""
    pass


class DataMissingError(GbdDataError):
    """Exception raised when data has unhandled missing entries."""
    pass


class DuplicateDataError(GbdDataError):
    """Exception raised when data has duplication in the index."""
    pass


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
        death_data = gbd.get_codcorrect_draws(gbd_ids=measure_entity_map['deaths'],
                                              location_ids=location_ids,
                                              gbd_round_id=gbd_round_id)
        # do processing to clean up data, add measure column if needed
        data = data.append(death_data)
    if 'remission' in measures:
        remission_data = gbd.get_modelable_entity_draws(gbd_ids=measure_entity_map['remission'],
                                                        location_ids=location_ids,
                                                        gbd_round_id=gbd_round_id)
        # do processing to clean up data, add measure column if needed
        data = data.append(remission_data)
    if 'prevalence' in measures or 'incidence' in measures:
        ids = measure_entity_map['prevalence'].copy().union(measure_entity_map['incidence'])
        measure_data = gbd.get_como_draws(gbd_ids=ids,
                                          location_ids=location_ids,
                                          gbd_round_id=gbd_round_id)
        # do processing to clean up data, add measure column if needed
        if 'prevalence' not in measures:
            measure_data = measure_data[measure_data.measure_id != name_measure_map['prevalence']]
        if 'incidence' not in measures:
            measure_data = measure_data[measure_data.measure_id != name_measure_map['incidence']]
        data = data.append(measure_data)

    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'gbd_id', 'gbd_name', 'measure']
    draw_columns = ['draw_{}'.format(i) for i in range(0, 1000)]

    validate_data(data, key_columns)

    return data[key_columns + draw_columns].sort_values(by=key_columns)


def get_populations(location_ids, gbd_round_id):
    populations = pd.concat([gbd.get_populations(location_id, gbd_round_id) for location_id in location_ids])
    keep_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id', 'population']
    return populations[keep_columns]


def get_prevalence(entities, location_ids, gbd_round_id):
    return get_gbd_draws(entities, ['prevalence'], location_ids, gbd_round_id)


def get_incidence(entities, location_ids, gbd_round_id):
    return get_gbd_draws(entities, ['incidence'], location_ids, gbd_round_id)


def get_remission(causes, location_ids, gbd_round_id):
    return get_gbd_draws(causes, ['remission'], location_ids, gbd_round_id)


def get_cause_specific_mortality(causes, location_ids, gbd_round_id):
    populations = get_populations(location_ids, gbd_round_id)
    deaths = get_gbd_draws(causes, ['deaths'], location_ids, gbd_round_id)

    # FIXME: stand in for actual calculation
    csmr = deaths/populations

    return csmr


def get_excess_mortality(causes, location_ids, gbd_round_id):
    prevalences = get_prevalence(causes, location_ids, gbd_round_id)
    csmrs = get_cause_specific_mortality(causes, location_ids, gbd_round_id)

    # FIXME: stand in for actual calculation
    em = csmrs/prevalences

    return em


def get_proportion(entity_pairs, location_ids, gbd_round_id):
    prevalences = get_prevalence(list({entity for pair in entity_pairs for entity in pair}), location_ids, gbd_round_id)

    # FIXME: stand in for actual calculation
    proportions = pd.concat([prevalences[child]/prevalences[parent] for parent, child in entity_pairs])

    return proportions
