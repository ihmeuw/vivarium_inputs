"""This module performs the core data transformations on GBD data and provides a basic API for data access."""
from collections import defaultdict
from typing import Iterable, Sequence, Union, DefaultDict, Set, List

import numpy as np
import pandas as pd

from ceam_inputs import gbd
from ceam_inputs.gbd import GBD_ROUND_ID
from ceam_inputs.gbd_mapping.templates import cid, sid, rid, UNKNOWN, Cause, Sequela, Etiology, Risk
Entity = Union[Cause, Sequela, Etiology, Risk]

# Define GBD sex ids for usage with central comp tools.
MALE = [1]
FEMALE = [2]
COMBINED = [3]

name_measure_map = {'prevalence': 5, 'incidence': 6, 'remission': 7, 'excess_mortality': 9, 'deaths': 1}
gbd_round_id_map = {3: 'GBD_2015', 4: 'GBD_2016'}



class GbdDataError(Exception):
    """Base exception for errors in GBD data."""
    pass


class InvalidQueryError(GbdDataError):
    """Exception raised when the user makes an invalid request for data (e.g. exposures for a sequela)."""
    pass


class DataMissingError(GbdDataError):
    """Exception raised when data has unhandled missing entries."""
    pass


class DuplicateDataError(GbdDataError):
    """Exception raised when data has duplication in the index."""
    pass


def get_ids_for_measure(entities: Sequence[Entity], measures: Iterable[str]) -> DefaultDict[str, Set]:
    """Selects the appropriate gbd id type for each entity and measure pair.

    Parameters
    ----------
    entities:
        A list of data containers from the `gbd_mapping` package. The entities must all be the same
        type (e.g. all `gbd_mapping.Cause` objects or all `gbd_mapping.Risk` objects, etc.
    measures:
        A list of the GBD measures requested for the provided entities.

    Returns
    -------
    A dictionary whose keys are the requested measures and whose values are sets of the appropriate
    GBD ids for use with central comp tools for the provided entities.

    Raises
    ------
    InvalidQueryError
        If the entities passed are inconsistent with the requested measures.
    """
    if not all([isinstance(e, type(entities[0])) for e in entities]):
        raise InvalidQueryError("All entities must be of the same type")

    out = defaultdict(set)
    if 'deaths' in measures:
        for entity in entities:
            if not isinstance(entity.gbd_id, cid):
                raise InvalidQueryError(f"Entity {entity.name} has no data for measure 'deaths'")

            out['deaths'].add(entity.gbd_id)
    if 'remission' in measures:
        for entity in entities:
            if not isinstance(entity.gbd_id, cid) or entity.dismod_id is UNKNOWN:
                raise InvalidQueryError(f"Entity {entity.name} has no data for measure 'remission'")

            out['remission'].add(entity.dismod_id)
    if 'prevalence' in measures:
        for entity in entities:
            if not isinstance(entity.gbd_id, (cid, sid)):
                raise InvalidQueryError(f"Entity {entity.name} has no data for measure 'prevalence'")

            out['prevalence'].add(entity.gbd_id)
    if 'incidence' in measures:
        for entity in entities:
            if not isinstance(entity.gbd_id, (cid, sid)):
                raise InvalidQueryError(f"Entity {entity.name} has no data for measure 'incidence'")

            out['incidence'].add(entity.gbd_id)
    if 'exposure' in measures:
        for entity in entities:
            if not isinstance(entity.gbd_id, rid):
                raise InvalidQueryError(f"Entity {entity.name} has no data for measure 'exposure'")

            out['exposure'].add(entity.gbd_id)
    if 'relative_risk' in measures:
        for entity in entities:
            if not isinstance(entity.gbd_id, rid):
                raise InvalidQueryError(f"Entity {entity.name} has no data for measure 'relative_risk'")

            out['relative_risk'].add(entity.gbd_id)
    if "population_attributable_fraction" in measures:
        for entity in entities:
            if not isinstance(entity.gbd_id, rid):
                raise InvalidQueryError(f"Entity {entity.name} has no data for measure 'population_attributable_fraction'")

            out['population_attributable_fraction'].add(entity.gbd_id)

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
    if np.any(data.isnull()):
        raise DataMissingError()

    if key_columns and data.duplicated(key_columns).sum():
        raise DuplicateDataError()


def get_gbd_draws(entities: Sequence[Entity], measures: Iterable[str], location_ids: Iterable[int]):
    measure_entity_map = get_ids_for_measure(entities, measures)

    data = []
    if 'deaths' in measure_entity_map:
        death_data = gbd.get_codcorrect_draws(cause_ids=list(measure_entity_map['deaths']),
                                              location_ids=location_ids)
        death_data = death_data[death_data['measure_id'] == name_measure_map['deaths']]
        death_data['measure'] = 'deaths'
        data.append(death_data)

    if 'remission' in measures:
        remission_data = gbd.get_modelable_entity_draws(me_ids=list(measure_entity_map['remission']),
                                                        location_ids=location_ids)
        remission_data = remission_data[remission_data['measure_id'] == name_measure_map['remission']]
        id_map = {entity.dismod_id: entity.gbd_id for entity in entities}
        remission_data['cause_id'] = remission_data['modelable_entity_id'].replace(id_map)
        remission_data['measure'] = 'remission'
        data.append(remission_data)

    if 'prevalence' in measures or 'incidence' in measures:
        ids = measure_entity_map['prevalence'].union(measure_entity_map['incidence'])
        measure_data = gbd.get_como_draws(entity_ids=list(ids),
                                          location_ids=location_ids)
        if 'prevalence' not in measures:
            measure_data = measure_data[measure_data['measure_id'] == name_measure_map['incidence']]
            measure_data['measure'] = 'incidence'
        elif 'incidence' not in measures:
            measure_data = measure_data[measure_data['measure_id'] == name_measure_map['prevalence']]
            measure_data['measure'] = 'prevalence'
        else:
            measure_data['measure'] = 'temp'
            measure_data.loc[measure_data['measure_id'] == name_measure_map['prevalence'], 'measure'] = 'prevalence'
            measure_data.loc[measure_data['measure_id'] == name_measure_map['incidence'], 'measure'] = 'incidence'

        data.append(measure_data)

    if 'relative_risk' in measures:
        ids = measure_entity_map['relative_risk']
        measure_data = gbd.get_relative_risks(risk_ids=list(ids), location_ids=location_ids)
        measure_data['measure'] = 'relative_risk'
        measure_data = measure_data.rename(columns={f'rr_{i}':f'draw_{i}' for i in range(1000)})

        data.append(measure_data)

    if 'population_attributable_fraction' in measures:
        ids = measure_entity_map['population_attributable_fraction']
        measure_data = gbd.get_pafs(risk_ids=list(ids), location_ids=location_ids)
        measure_data['measure'] = 'population_attributable_fraction'
        data.append(measure_data)

    if 'exposure' in measures:
        ids = measure_entity_map['exposure']
        measure_data = gbd.get_exposures(risk_ids=list(ids), location_ids=location_ids)
        measure_data['measure'] = 'exposure'
        data.append(measure_data)

    data = pd.concat(data)

    if np.any(data['measure'].str.contains('population_attributable_fraction')):
        id_cols = ['cause_id', 'risk_id', 'mortality', 'morbidity']
    elif np.any(data['measure'].str.contains('relative_risk')):
        id_cols = ['cause_id', 'risk_id', 'parameter', 'mortality', 'morbidity']
    elif np.any(data['measure'].str.contains('exposure')):
        id_cols = ['risk_id', 'parameter']
    elif 'cause_id' in data.columns:
        id_cols = ['cause_id']
    elif 'sequela_id' in data.columns:
        id_cols = ['sequela_id']
    else:
        raise GbdDataError('Stuff is broken')

    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'measure'] + id_cols
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]

    data = data[key_columns + draw_columns]
    validate_data(data, key_columns)

    return data.reset_index(drop=True)


####################################
# Measures for cause like entities #
####################################


def get_prevalences(entities, location_ids):
    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'gbd_id']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]
    return get_gbd_draws(entities, ['prevalence'], location_ids)[key_columns + draw_columns]


def get_incidences(entities, location_ids):
    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'gbd_id']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]
    return get_gbd_draws(entities, ['incidence'], location_ids)[key_columns + draw_columns]


def get_remissions(causes, location_ids):
    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'gbd_id']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]
    return get_gbd_draws(causes, ['remission'], location_ids)[key_columns + draw_columns]


def get_cause_specific_mortalities(causes, location_ids):
    deaths = get_gbd_draws(causes, ["deaths"], location_ids)
    del deaths['measure_id']

    populations = get_populations(location_ids)
    populations = populations[populations['year_id'] >= deaths.year_id.min()]

    merge_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id']
    key_columns = merge_columns + ['gbd_id']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]

    df = deaths.merge(populations, on=merge_columns).set_index(key_columns)
    csmr = df[draw_columns].divide(df['population'], axis=0).reset_index()

    csmr = csmr[key_columns + draw_columns]
    validate_data(csmr, key_columns)

    return csmr


def get_excess_mortalities(causes, location_ids):
    prevalences = get_prevalences(causes, location_ids)
    csmrs = get_cause_specific_mortalities(causes, location_ids)

    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', "gbd_id"]
    prevalences = prevalences.set_index(key_columns)
    csmrs = csmrs.set_index(key_columns)

    em = csmrs.divide(prevalences, axis="index")

    return em.reset_index().dropna()


def get_disability_weights(sequelae, _):
    gbd_round = gbd_round_id_map[GBD_ROUND_ID]
    disability_weights = gbd.get_data_from_auxiliary_file('Disability Weights', gbd_round=gbd_round)
    combined_disability_weights = gbd.get_data_from_auxiliary_file('Combined Disability Weights', gbd_round=gbd_round)

    data = []
    for s in sequelae:
        # Only sequelae have disability weights.
        assert isinstance(s.gbd_id, sid)
        if s.healthstate.gbd_id in disability_weights['healthstate_id']:
            df = disability_weights.loc[disability_weights.healthstate_id == s.healthstate.gbd_id].copy()
        elif s.healthstate.gbd_id in combined_disability_weights['healthstate_id']:
            df = disability_weights.loc[disability_weights.healthstate_id == s.healthstate.gbd_id].copy()
        else:
            raise DataMissingError(f"No disability weight available for the sequela {s.name}")
        df['sequela_id'] = s.gbd_id
        df['measure'] = 'disability_weight'
        data.append(df)

    data = pd.concat(data)
    return df.reset_index(drop=True)


####################################
# Measures for risk like entities  #
####################################


def get_relative_risks(risks, location_ids):
    key_columns = ['cause_id', 'year_id', 'sex_id', 'age_group_id', 'location_id', 'risk_id', 'cause_id', 'parameter', 'mortality', 'morbidity']
    draw_columns = [f'draw_{i}' for i in range(1000)]
    df = get_gbd_draws(risks, ['relative_risk'], location_ids)[key_columns + draw_columns]
    return df[key_columns + draw_columns]


def get_exposures(risks, location_ids):
    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'gbd_id', 'parameter']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]
    return get_gbd_draws(risks, ['exposure'], location_ids)[key_columns + draw_columns]


def get_population_attributable_fractions(risks, location_ids):
    key_columns = ['cause_id', 'year_id', 'sex_id', 'age_group_id', 'location_id', 'risk_id', 'cause_id', 'mortality', 'morbidity']
    draw_columns = [f'draw_{i}' for i in range(1000)]
    df = get_gbd_draws(risks, ['population_attributable_fraction'], location_ids)[key_columns + draw_columns]
    return df[key_columns + draw_columns]


def get_ensemble_weights(risks):
    data = []
    ids = [risk.gbd_id for risk in risks]
    for i in range(0, (len(ids))):
        risk_id = ids[i]
        temp = gbd.get_data_from_auxiliary_file('Ensemble Distribution Weights',
                                            gbd_round = gbd_round_id_map[GBD_ROUND_ID],
                                            rei_id = risk_id)
        temp['risk_id'] = risk_id
        data.append(temp)
    data = pd.concat(data)
    return data


def get_mediation_factors(risks, location_ids):
    risk_ids = [risk.gbd_id for risk in risks]
    _gbd_round_id_map = {3: 'GBD_2015', 4: 'GBD_2016'}
    data = gbd.get_data_from_auxiliary_file("Mediation Factors", gbd_round=_gbd_round_id_map[GBD_ROUND_ID])
    data = data.rename(columns={'rei_id': 'risk_id'})

    data = data.query('risk_id in @risk_ids').copy()

    if not data.empty:
        draw_columns = [f'draw_{i}' for i in range(0, 1000)]
        data[draw_columns] = 1 - (data[draw_columns])
        data = data.groupby(['cause_id', 'risk_id'])[draw_columns].prod()
        data = data.reset_index()
        return data

    else:
        return 0


#######################
# Other kinds of data #
#######################


def get_populations(location_ids):
    populations = pd.concat([gbd.get_populations(location_id) for location_id in location_ids])
    populations = populations[populations['sex_id'] != COMBINED]
    keep_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id', 'population']
    return populations[keep_columns]


def get_age_bins():
    return gbd.get_age_bins()


def get_life_tables(location_id):
    return gbd.get_life_table(location_id)

