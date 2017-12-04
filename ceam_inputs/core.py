"""This module performs the core data transformations on GBD data and provides a basic API for data access."""
from collections import defaultdict
from typing import Iterable, Sequence, Union, DefaultDict, Set, List

import numpy as np
import pandas as pd

from ceam_inputs import gbd, risk_factor_correlation
from ceam_inputs.gbd_mapping.templates import cid, sid, rid, UNKNOWN, Cause, Sequela, Etiology, Risk
from ceam_inputs.gbd_mapping.healthcare_entities import HealthcareEntity
GbdEntity = Union[Cause, Sequela, Etiology, Risk]

# Define GBD sex ids for usage with central comp tools.
MALE = [1]
FEMALE = [2]
COMBINED = [3]

name_measure_map = {'death': 1,
                    'DALY': 2,
                    'YLD': 3,
                    'YLL': 4,
                    'prevalence': 5,
                    'incidence': 6,
                    'remission': 7,
                    'excess_mortality': 9,
                    'proportion': 18,
                    'continuous': 19,}
gbd_round_id_map = {3: 'GBD_2015', 4: 'GBD_2016'}


class DataError(Exception):
    """Base exception for errors in data loading."""
    pass


class InvalidQueryError(DataError):
    """Exception raised when the user makes an invalid request for data (e.g. exposures for a sequela)."""
    pass


class DataMissingError(DataError):
    """Exception raised when data has unhandled missing entries."""
    pass


class DuplicateDataError(DataError):
    """Exception raised when data has duplication in the index."""
    pass


def get_ids_for_measure(entities: Sequence[GbdEntity], measures: Iterable[str]) -> DefaultDict[str, Set]:
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

    measure_types = {
            'death': (cid, 'gbd_id'),
            'remision': (cid, 'dismod_id'),
            'prevalence': ((cid, sid), 'gbd_id'),
            'incidence': ((cid, sid), 'gbd_id'),
            'exposure_mean': (rid, 'gbd_id'),
            'relative_risk': (rid, 'gbd_id'),
            'population_attributable_fraction': (rid, 'gbd_id'),
    }

    for measure in measures:
        if measure in measure_types:
            valid_types, id_attr = measure_types[measure]
            for entity in entities:
                if not isinstance(entity.gbd_id, valid_types):
                    raise InvalidQueryError(f"Entity {entity.name} has no data for measure '{measure}'")

                out[measure].add(entity[id_attr])
        elif "annual_visits" == measure:
            # This one's a bit special
            for entity in entities:
                if not isinstance(entity, HealthcareEntity):
                    raise InvalidQueryError(
                        f"Entity {entity.name} has no data for measure 'annual_visits'")
                out['annual_visits'].add(entity.utilization)
        else:
            raise DataError('Stuff is broken')

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

    if key_columns and np.any(data.duplicated(key_columns)):
        raise DuplicateDataError()


def _get_death(measure_name, measure_ids, entities, location_ids):
    death_data = gbd.get_codcorrect_draws(cause_ids=measure_ids,
                                          location_ids=location_ids)
    death_data = death_data[death_data['measure_id'] == name_measure_map[measure_name]]
    return death_data

def _get_remission(measure_name, measure_ids, entities, location_ids):
    remission_data = gbd.get_modelable_entity_draws(me_ids=measure_ids,
                                                    location_ids=location_ids)
    remission_data = remission_data[remission_data['measure_id'] == name_measure_map['remission']]
    id_map = {entity.dismod_id: entity.gbd_id for entity in entities}
    remission_data['cause_id'] = remission_data['modelable_entity_id'].replace(id_map)
    remission_data = remission_data[remission_data['sex_id'] != COMBINED]
    return remission_data

def _get_prevalence_or_prevalence(measure_name, measure_ids, entities, location_ids):
    measure_data = gbd.get_como_draws(entity_ids=measure_ids,
                                      location_ids=location_ids)

    estimation_years = gbd.get_estimation_years(gbd.GBD_ROUND_ID)
    measure_data = measure_data.query('year_id in @estimation_years')

    measure_data = measure_data[measure_data['measure_id'] == name_measure_map[measure_name]]

    return measure_data

def _get_relative_risk(measure_name, measure_ids, entities, location_ids):
    measure_data = gbd.get_relative_risks(risk_ids=measure_ids, location_ids=location_ids)
    err_msg = ("Not all relative risk data has both the 'mortality' and 'morbidity' flag "
               + "set. This may not indicate an error but it is a case we don't explicitly handle. "
               + "If you need this risk, come talk to one of the programmers.")
    assert np.all((measure_data.mortality == 1) & (measure_data.morbidity == 1)), err_msg

    del measure_data['mortality']
    del measure_data['morbidity']

    measure_data = measure_data.rename(columns={f'rr_{i}': f'draw_{i}' for i in range(1000)})

    return measure_data


def _get_population_attributable_fraction(measure_name, measure_ids, entities, location_ids):
    measure_data = gbd.get_pafs(risk_ids=measure_ids, location_ids=location_ids)

    key_columns = ['sex_id', 'risk_id', 'year_id', 'cause_id', 'age_group_id', 'location_id']
    measure_ids = {name_measure_map['death'], name_measure_map['DALY'],
                   name_measure_map['YLD'], name_measure_map['YLL']}
    err_msg = ("Not all PAF data has values for deaths, DALYs, YLDs and YLLs. "
               + "This may not indicate an error but it is a case we don't explicitly handle. "
               + "If you need this PAF, come talk to one of the programmers.")
    assert np.all(
        measure_data.groupby(key_columns).measure_id.unique().apply(lambda x: set(x) == measure_ids)), err_msg

    # TODO: figure out if we need to assert some property of the different PAF measures
    yld_id = name_measure_map['YLD']
    measure_data = measure_data.query('measure_id == @yld_id').copy()
    del measure_data['measure_id']
    return measure_data


def _get_exposure_mean(measure_name, measure_ids, entities, location_ids):
    measure_data = gbd.get_exposures(risk_ids=measure_ids, location_ids=location_ids)
    measure_data = measure_data[measure_data.measure_id == name_measure_map['proportion']]
    del measure_data['measure_id']
    return measure_data


def _get_annual_visits(measure_name, measure_ids, entities, location_ids):
    measure_data = gbd.get_modelable_entity_draws(me_ids=measure_ids, location_ids=location_ids)
    measure_data = measure_data[measure_data['measure_id'] == name_measure_map['continuous']]
    measure_data = measure_data[measure_data['sex_id'] != COMBINED]
    return measure_data


def get_gbd_draws(entities: Sequence[GbdEntity], measures: Iterable[str], location_ids: Iterable[int]) -> pd.DataFrame:
    """Gets draw level gbd data for each specified measure and entity.

    Parameters
    ----------
    entities:
        A list of data containers from the `gbd_mapping` package. The entities must all be the same
        type (e.g. all `gbd_mapping.Cause` objects or all `gbd_mapping.Risk` objects, etc.
    measures:
        A list of the GBD measures requested for the provided entities.
    location_ids:
        A list of location ids to pull data for.

    Returns
    -------
    A table of draw level data for indexed by an entity, measure, and location combination as well as demographic data
    (age_group_id, sex_id, year_id) where appropriate.
    """
    measure_entity_map = get_ids_for_measure(entities, measures)

    measure_handlers = {
            'death': _get_death,
            'remission': _get_remission,
            'prevalence': _get_prevalence_or_prevalence,
            'incidence': _get_prevalence_or_prevalence,
            'relative_risk': _get_relative_risk,
            'population_attributable_fraction': _get_population_attributable_fraction,
            'exposure_mean': _get_exposure_mean,
            'annual_visits': _get_annual_visits,
    }

    data = []
    for measure_name, measure_ids in measure_entity_map.items():
        measure_data = measure_handlers[measure_name](measure_name, list(measure_ids), entities, location_ids)
        measure_data['measure'] = measure_name
        data.append(measure_data)
    data = pd.concat(data)

    id_cols = set()
    if np.any(data['measure'].str.contains('population_attributable_fraction')):
        id_cols.update(['cause_id', 'risk_id'])
    if np.any(data['measure'].str.contains('relative_risk')):
        id_cols.update(['cause_id', 'risk_id', 'parameter'])
    if np.any(data['measure'].str.contains('exposure_mean')):
        id_cols.update(['risk_id', 'parameter'])
    if np.any(data['measure'].str.contains('annual_visits')):
        id_cols.add('modelable_entity_id')
    if 'cause_id' in data.columns:
        id_cols.add('cause_id')
    if 'sequela_id' in data.columns:
        id_cols.add('sequela_id')
    if not id_cols:
        raise DataError('Stuff is broken')

    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'measure'] + list(id_cols)
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]

    data = data[key_columns + draw_columns]
    validate_data(data, key_columns)

    return data.reset_index(drop=True)


####################################
# Measures for cause like entities #
####################################


def get_prevalences(entities: Union[Sequence[Cause], Sequence[Sequela]], location_ids: Sequence[int]) -> pd.DataFrame:
    """Gets prevalence data for the specified entities and locations.

    Parameters
    ----------
    entities:
        A list of data containers from the `gbd_mapping` package. The entities must all be the same
        type (all `gbd_mapping.Cause` objects or all `gbd_mapping.Sequela` objects).
    location_ids:
        A list of location ids to pull data for.

    Returns
    -------
    A table of prevalence data for indexed by the given entity ids and location ids
    as well as by demographic data (year_id, sex_id, and age_group_id).
    """
    id_type = 'cause_id' if isinstance(entities[0], Cause) else 'sequela_id'
    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', id_type]
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]
    return get_gbd_draws(entities, ['prevalence'], location_ids)[key_columns + draw_columns]


def get_incidences(entities: Union[Sequence[Cause], Sequence[Sequela]], location_ids: Sequence[int]) -> pd.DataFrame:
    """Gets incidence data for the specified entities and locations.

    Parameters
    ----------
    entities:
        A list of data containers from the `gbd_mapping` package. The entities must all be the same
        type (all `gbd_mapping.Cause` objects or all `gbd_mapping.Sequela` objects).
    location_ids:
        A list of location ids to pull data for.

    Returns
    -------
    A table of incidence data for indexed by the given entity ids and location ids
    as well as by demographic data (year_id, sex_id, and age_group_id).
    """
    id_type = 'cause_id' if isinstance(entities[0], Cause) else 'sequela_id'
    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', id_type]
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]
    return get_gbd_draws(entities, ['incidence'], location_ids)[key_columns + draw_columns]


def get_remissions(causes: Sequence[Cause], location_ids: Sequence[int]) -> pd.DataFrame:
    """Gets remission data for the specified causes and locations.

    Parameters
    ----------
    causes:
        A list of `Cause` data containers from the `gbd_mapping` package.
    location_ids:
        A list of location ids to pull data for.

    Returns
    -------
    A table of incidence data for indexed by the given cause ids and location ids
    as well as by demographic data (year_id, sex_id, and age_group_id).
    """
    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'cause_id']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]
    return get_gbd_draws(causes, ['remission'], location_ids)[key_columns + draw_columns]


def get_cause_specific_mortalities(causes: Sequence[Cause], location_ids: Sequence[int]) -> pd.DataFrame:
    """Gets cause specific mortality data for the specified causes and locations.

    Parameters
    ----------
    causes:
        A list of `Cause` data containers from the `gbd_mapping` package.
    location_ids:
        A list of location ids to pull data for.

    Returns
    -------
    A table of cause specific mortality data for indexed by the given cause ids and location ids
    as well as by demographic data (year_id, sex_id, and age_group_id).
    """
    deaths = get_gbd_draws(causes, ["death"], location_ids)

    populations = get_populations(location_ids)
    populations = populations[populations['year_id'] >= deaths.year_id.min()]

    merge_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id']
    key_columns = merge_columns + ['cause_id']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]

    df = deaths.merge(populations, on=merge_columns).set_index(key_columns)
    csmr = df[draw_columns].divide(df['population'], axis=0).reset_index()

    csmr = csmr[key_columns + draw_columns]
    validate_data(csmr, key_columns)

    return csmr


def get_excess_mortalities(causes: Sequence[Cause], location_ids: Sequence[int]) -> pd.DataFrame:
    """Gets excess mortality data for the specified causes and locations.

    Parameters
    ----------
    causes:
        A list of `Cause` data containers from the `gbd_mapping` package.
    location_ids:
        A list of location ids to pull data for.

    Returns
    -------
    A table of excess mortality data for indexed by the given cause ids and location ids
    as well as by demographic data (year_id, sex_id, and age_group_id).
    """
    prevalences = get_prevalences(causes, location_ids)
    csmrs = get_cause_specific_mortalities(causes, location_ids)

    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'cause_id']
    prevalences = prevalences.set_index(key_columns)
    csmrs = csmrs.set_index(key_columns)

    em = csmrs.divide(prevalences, axis='index')

    return em.reset_index().dropna()


def get_disability_weights(sequelae: Sequence[Sequela], _: Sequence[int]) -> pd.DataFrame:
    # TODO: Check out what the data looks like to verify Returns section.
    """Gets disability weight data for the specified sequelae and locations.

    Parameters
    ----------
    sequelae:
        A list of `Sequela` data containers from the `gbd_mapping` package.
    _:
        A list of location ids for API consistency.

    Returns
    -------
    A table of disability data for indexed by the given sequela ids and location ids
    as well as by demographic data (year_id, sex_id, and age_group_id).
    """
    gbd_round = gbd_round_id_map[gbd.GBD_ROUND_ID]
    disability_weights = gbd.get_data_from_auxiliary_file('Disability Weights', gbd_round=gbd_round)
    combined_disability_weights = gbd.get_data_from_auxiliary_file('Combined Disability Weights', gbd_round=gbd_round)

    data = []
    for s in sequelae:
        # Only sequelae have disability weights.
        assert isinstance(s.gbd_id, sid)
        if s.healthstate.gbd_id in disability_weights['healthstate_id'].values:
            df = disability_weights.loc[disability_weights.healthstate_id == s.healthstate.gbd_id].copy()
        elif s.healthstate.gbd_id in combined_disability_weights['healthstate_id'].values:
            df = disability_weights.loc[disability_weights.healthstate_id == s.healthstate.gbd_id].copy()
        else:
            raise DataMissingError(f"No disability weight available for the sequela {s.name}")
        df['sequela_id'] = s.gbd_id
        df['measure'] = 'disability_weight'
        data.append(df)

    data = pd.concat(data)
    data = data.rename(columns={f'draw{i}':f'draw_{i}' for i in range(1000)})
    return data.reset_index(drop=True)


####################################
# Measures for risk like entities  #
####################################


def get_relative_risks(entities, location_ids):
    id_col = 'risk_id' if isinstance(entities[0].gbd_id, rid) else 'treatment_technology'
    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', id_col, 'cause_id', 'parameter']
    draw_columns = [f'draw_{i}' for i in range(1000)]
    if isinstance(entities[0].gbd_id, rid):
        df = get_gbd_draws(entities, ['relative_risk'], location_ids)
    else:
        data = []
        for entity in entities:
            data.append(gbd.get_data_from_auxiliary_file(entity.relative_risk,
                                                         gbd_round=gbd_round_id_map[gbd.GBD_ROUND_ID]))
        df = pd.concat(data)
    return df[key_columns + draw_columns]


def get_exposure_means(risks, location_ids):
    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'risk_id', 'parameter']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]
    return get_gbd_draws(risks, ['exposure_mean'], location_ids)[key_columns + draw_columns]


def get_exposure_standard_deviations(risks, location_ids):
    # TODO: This stuff.
    pass


def get_population_attributable_fractions(entities, location_ids):
    id_col = 'risk_id' if isinstance(entities[0].gbd_id, rid) else 'treatment_technology'
    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location_id', id_col, 'cause_id']
    draw_columns = [f'draw_{i}' for i in range(1000)]
    if isinstance(entities[0].gbd_id, rid):
        df = get_gbd_draws(entities, ['population_attributable_fraction'], location_ids)
    else:
        data = []
        for entity in entities:
            temp = gbd.get_data_from_auxiliary_file(entity.population_attributable_fraction,
                                                    gbd_round=gbd_round_id_map[gbd.GBD_ROUND_ID])
            temp = temp[temp['location_id'].isin(location_ids)]
            data.append(temp)
        df = pd.concat(data)
    return df[key_columns + draw_columns]


def get_ensemble_weights(risks, location_ids):
    data = []
    ids = [risk.gbd_id for risk in risks]
    for i in range(0, (len(ids))):
        risk_id = ids[i]
        temp = gbd.get_data_from_auxiliary_file('Ensemble Distribution Weights',
                                                gbd_round=gbd_round_id_map[gbd.GBD_ROUND_ID],
                                                rei_id=risk_id)
        temp['risk_id'] = risk_id
        data.append(temp)
    data = pd.concat(data)
    return data


def get_mediation_factors(risks, location_ids):
    risk_ids = [risk.gbd_id for risk in risks]
    _gbd_round_id_map = {3: 'GBD_2015', 4: 'GBD_2016'}
    data = gbd.get_data_from_auxiliary_file("Mediation Factors", gbd_round=_gbd_round_id_map[gbd.GBD_ROUND_ID])
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


def get_risk_correlation_matrices(location_ids):
    data = []
    for location_id in location_ids:
        df = risk_factor_correlation.load_matrices(location_id=location_id,
                                                   gbd_round=gbd_round_id_map[gbd.GBD_ROUND_ID])
        df['location_id'] = location_id
        data.append(df)
    return pd.concat(data)


#######################
# Other kinds of data #
#######################


def get_populations(location_ids):
    populations = pd.concat([gbd.get_populations(location_id) for location_id in location_ids])
    keep_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id', 'population']
    return populations[keep_columns]


def get_age_bins():
    return gbd.get_age_bins()


def get_theoretical_minimum_risk_life_expectancy():
    return gbd.get_theoretical_minimum_risk_life_expectancy()


def get_subregions(location_ids):
    return gbd.get_subregions(location_ids)


def get_costs(entities, location_ids):
    data = []
    for entity in entities:
        df = gbd.get_data_from_auxiliary_file(entity.cost)
        if entity.name in ['inpatient_visits', 'outpatient_visits']:
            df = df[df['location_id'].isin(location_ids)]
        data.append(df)
    return pd.concat(data)

def get_healthcare_annual_visits(entities, location_ids):
    key_columns = ['age_group_id', 'location_id', 'modelable_entity_id', 'sex_id', 'year_id']
    draw_columns = [f'draw_{i}' for i in range(1000)]
    df = get_gbd_draws(entities, ['annual_visits'], location_ids)[key_columns + draw_columns]
    return df

def get_covariate_estimates(covariates, location_ids):
    return gbd.get_covariate_estimates([covariate.gbd_id for covariate in covariates], location_ids)


def get_protection(treatment_technologies, location_ids):
    data = []
    for tt in treatment_technologies:
        df = gbd.get_data_from_auxiliary_file(tt.protection)
        if not set(location_ids).issubset(set(df['location_id'].unique())):

            raise DataMissingError(f'Protection data for {tt.name} is not available for locations '
                                   f'{set(location_ids) - set(df["location_id"].unique())}')
        df = df[df['location_id'].isin(location_ids)]
        data.append(df)
    return pd.concat(data)

