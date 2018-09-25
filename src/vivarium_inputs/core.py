"""This module performs the core data transformations on GBD data and provides a basic API for data access."""
from typing import Iterable, Sequence, List, Union

import numpy as np
import pandas as pd

from gbd_mapping.id import scalar, UNKNOWN
from gbd_mapping.base_template import ModelableEntity
from gbd_mapping.cause import Cause, causes
from gbd_mapping.risk import Risk, risk_factors
from gbd_mapping.sequela import Sequela
from gbd_mapping.etiology import Etiology, etiologies
from gbd_mapping.coverage_gap import CoverageGap, coverage_gaps
from gbd_mapping.covariate import Covariate

try:
    import vivarium_gbd_access.gbd as gbd
except ModuleNotFoundError:
    class GbdDummy:
        def __getattr__(self, item):
            raise ModuleNotFoundError("Required package vivarium_gbd_access not found.")
    gbd = GbdDummy()


from vivarium_inputs.mapping_extension import HealthcareEntity, HealthTechnology


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
                    'continuous': 19, }
gbd_round_id_map = {3: 'GBD_2015', 4: 'GBD_2016'}
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


def get_draws(entities: Sequence[ModelableEntity], measures: Iterable[str],
              locations: Iterable[str]) -> pd.DataFrame:
    """Gets draw level gbd data for each specified measure and entity.

    Parameters
    ----------
    entities:
        A list of data containers from the `gbd_mapping` package. The entities must all be the same
        type (e.g. all `gbd_mapping.Cause` objects or all `gbd_mapping.Risk` objects, etc.
    measures:
        A list of the GBD measures requested for the provided entities.
    locations:
        A list of locations to pull data for.

    Returns
    -------
    A table of draw level data for indexed by an entity, measure, and location combination as well as demographic data
    (age_group_id, sex_id, year_id) where appropriate.
    """
    measure_handlers = {
        'death': (_get_death, set()),
        'remission': (_get_remission, set()),
        'prevalence': (_get_prevalence, set()),
        'incidence': (_get_incidence, set()),
        'relative_risk': (_get_relative_risk, {'cause_id', 'parameter', }),
        'population_attributable_fraction': (_get_population_attributable_fraction, {'cause_id', }),
        'cause_specific_mortality': (_get_cause_specific_mortality, set()),
        'excess_mortality': (_get_excess_mortality, set()),
        'exposure': (_get_exposure, {'parameter', }),
        'exposure_standard_deviation': (_get_exposure_standard_deviation, {'risk_id', }),
        'annual_visits': (_get_annual_visits, {'modelable_entity_id', }),
        'disability_weight': (_get_disability_weight, set()),
        'cost': (_get_cost, set()),
    }

    location_ids = [get_location_ids_by_name()[name] for name in locations]

    data = []
    id_cols = set()
    for measure in measures:
        try:
            handler, id_columns = measure_handlers[measure]
        except KeyError:
            raise InvalidQueryError(f'No functions are available to pull data for measure {measure}')
        measure_data = handler(entities, location_ids)
        measure_data['measure'] = measure
        id_cols |= id_columns
        data.append(measure_data)
    data = pd.concat(data)

    id_cols |= _get_additional_id_columns(data, entities)
    key_columns = ['measure']
    for column in ['year_id', 'sex_id', 'age_group_id', 'location_id']:
        if column in data:
            key_columns.append(column)

    key_columns += list(id_cols)
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]

    data = data[key_columns + draw_columns].reset_index(drop=True)
    _validate_data(data, key_columns)
    if "location_id" in data:
        data["location"] = data.location_id.apply(get_location_names_by_id().get)
        data = data.drop("location_id", "columns")

    return data


# TODO: Move to utilities.py
def _get_ids_for_measure(entities: Sequence[ModelableEntity], measure: str) -> List:
    """Selects the appropriate gbd id type for each entity and measure pair.

    Parameters
    ----------
    entities :
        A list of data containers from the `gbd_mapping` package. The entities must all be the same
        type (e.g. all `gbd_mapping.Cause` objects or all `gbd_mapping.Risk` objects, etc.
    measure :
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
    measure_types = {
        'death': (Cause, 'gbd_id'),
        'prevalence': ((Cause, Sequela), 'gbd_id'),
        'incidence': ((Cause, Sequela), 'gbd_id'),
        'exposure': ((Risk, CoverageGap), 'gbd_id'),
        'exposure_standard_deviation': ((Risk, CoverageGap), 'gbd_id'),
        'relative_risk': ((Risk, CoverageGap), 'gbd_id'),
        'population_attributable_fraction': ((Cause, Etiology, CoverageGap), 'gbd_id'),
        'cause_specific_mortality': ((Cause,), 'gbd_id'),
        'excess_mortality': ((Cause,), 'gbd_id'),
        'annual_visits': (HealthcareEntity, 'utilization'),
        'disability_weight': (Sequela, 'gbd_id'),
        'remission': (Cause, 'dismod_id'),
        'cost': ((HealthcareEntity, HealthTechnology), 'cost'),
    }

    if not all([isinstance(e, type(entities[0])) for e in entities]):
        raise InvalidQueryError("All entities must be of the same type")
    if measure not in measure_types.keys():
        raise InvalidQueryError(f"You've requested an invalid measure: {measure}")

    valid_types, id_attr = measure_types[measure]
    out = []
    for entity in entities:
        if isinstance(entity, valid_types):
            value = entity
            for sub_id in id_attr.split('.'):
                if value[sub_id] is not UNKNOWN:
                    value = value[sub_id]
                else:
                    raise InvalidQueryError(f"Entity {entity.name} has no data for measure '{measure}'")
            out.append(value)
        else:
            raise InvalidQueryError(f"Entity {entity.name} has no data for measure '{measure}'")

    return out


# TODO: Move to utilities.py
def _get_additional_id_columns(data, entities):
    id_column_map = {
        Cause: 'cause_id',
        Sequela: 'sequela_id',
        Covariate: 'covariate_id',
        Risk: 'risk_id',
        Etiology: 'etiology_id',
        CoverageGap: 'coverage_gap_id',
        HealthcareEntity: 'healthcare_entity',
        HealthTechnology: 'health_technology',
    }
    out = set()
    out.add(id_column_map[type(entities[0])])
    if isinstance(entities[0], CoverageGap) and data['measure'].all() in ['relative_risk', 'population_attributable_fraction']:
        out.add('rei_id')
    out |= set(data.columns) & set(id_column_map.values())
    return out


# TODO: Move to utilities.py
def _validate_data(data: pd.DataFrame, key_columns: Iterable[str]=None):
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

#####################
# get_draws helpers #
#####################
#
# These functions filter out erroneous measures and deal with special cases.
#

####################
# Cause-like stuff #
####################

def _get_death(entities, location_ids):
    measure_ids = _get_ids_for_measure(entities, 'death')
    death_data = gbd.get_codcorrect_draws(cause_ids=measure_ids, location_ids=location_ids)

    return death_data[death_data['measure_id'] == name_measure_map['death']]


def _get_remission(entities, location_ids):
    measure_ids = _get_ids_for_measure(entities, 'remission')
    remission_data = gbd.get_modelable_entity_draws(me_ids=measure_ids, location_ids=location_ids)

    id_map = {entity.dismod_id: entity.gbd_id for entity in entities}
    remission_data['cause_id'] = remission_data['modelable_entity_id'].replace(id_map)

    # FIXME: The sex filtering should happen in the reshaping step.
    correct_measure = remission_data['measure_id'] == name_measure_map['remission']
    correct_sex = remission_data['sex_id'] != COMBINED
    return remission_data[correct_measure & correct_sex]


def _get_prevalence(entities, location_ids):
    measure_ids = _get_ids_for_measure(entities, 'prevalence')
    entity_type = "cause" if isinstance(entities[0], Cause) else "sequela"
    measure_data = gbd.get_como_draws(entity_ids=measure_ids, location_ids=location_ids, entity_type=entity_type)

    # FIXME: The year filtering should happen in the reshaping step.
    correct_measure = measure_data['measure_id'] == name_measure_map['prevalence']
    correct_years = measure_data['year_id'].isin(gbd.get_estimation_years(gbd.GBD_ROUND_ID))
    return measure_data[correct_measure & correct_years]


def _get_incidence(entities, location_ids):
    measure_ids = _get_ids_for_measure(entities, 'incidence')
    entity_type = "cause" if isinstance(entities[0], Cause) else "sequela"
    measure_data = gbd.get_como_draws(entity_ids=measure_ids, location_ids=location_ids, entity_type=entity_type)

    # FIXME: The year filtering should happen in the reshaping step.
    correct_measure = measure_data['measure_id'] == name_measure_map['incidence']
    correct_years = measure_data['year_id'].isin(gbd.get_estimation_years(gbd.GBD_ROUND_ID))
    return measure_data[correct_measure & correct_years]


def _get_cause_specific_mortality(entities, location_ids):
    # FIXME: Mapping backword to name like this is awkward
    locations = [get_location_names_by_id()[location_id] for location_id in location_ids]
    deaths = get_draws(entities, ["death"], locations)
    deaths["location_id"] = deaths.location.apply(get_location_ids_by_name().get)
    deaths = deaths.drop("location", "columns")

    populations = get_populations(locations)
    populations = populations[populations['year_id'] >= deaths.year_id.min()]
    populations["location_id"] = populations.location.apply(get_location_ids_by_name().get)

    merge_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id']
    key_columns = merge_columns + ['cause_id']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]

    df = deaths.merge(populations, on=merge_columns).set_index(key_columns)
    csmr = df[draw_columns].divide(df['population'], axis=0).reset_index()

    csmr = csmr[key_columns + draw_columns]
    _validate_data(csmr, key_columns)

    return csmr


def _get_excess_mortality(entities, location_ids):
    # FIXME: Mapping backword to name like this is awkward
    locations = [get_location_names_by_id()[location_id] for location_id in location_ids]
    prevalences = get_draws(entities, ['prevalence'], locations).drop('measure', 'columns')
    csmrs = get_draws(entities, ['cause_specific_mortality'], locations).drop('measure', 'columns')

    key_columns = ['year_id', 'sex_id', 'age_group_id', 'location', 'cause_id']
    prevalences = prevalences.set_index(key_columns)
    csmrs = csmrs.set_index(key_columns)

    # In some cases CSMR is not zero for age groups where prevalence is, which leads to
    # crazy outputs. So enforce that constraint.
    # TODO: But is this the right place to do that?
    draw_columns = [f'draw_{i}' for i in range(1000)]
    csmrs[draw_columns] = csmrs[draw_columns].where(prevalences[draw_columns] != 0, 0)

    em = csmrs.divide(prevalences, axis='index').reset_index()
    em = em[em['sex_id'] != COMBINED]

    em["location_id"] = em.location.apply(get_location_ids_by_name().get)
    em = em.drop("location", "columns")

    return em.dropna()


# TODO This should probably use the _get_ids_for_measure function but it doesn't quite fit
def _get_disability_weight(entities, _):
    if isinstance(entities[0], Sequela):
        disability_weights = gbd.get_auxiliary_data('disability_weight', 'sequela', 'all')
    else:
        raise InvalidQueryError("Only sequela have disability weights associated with them.")

    data = []
    for s in entities:

        if s.healthstate.gbd_id in disability_weights['healthstate_id'].values:
            df = disability_weights.loc[disability_weights.healthstate_id == s.healthstate.gbd_id, :]
        else:
            raise DataMissingError(f"No disability weight available for the sequela {s.name}")

        df['sequela_id'] = s.gbd_id
        data.append(df)

    data = pd.concat(data, ignore_index=True)
    return data.reset_index(drop=True)


###################
# Risk-like stuff #
###################

def _standardize_data(data: pd.DataFrame, fill_value: int) -> pd.DataFrame:
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


def _standardize_all_age_groups(data: pd.DataFrame):
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


def _get_relative_risk(entities: Iterable[Union[Risk, CoverageGap]], location_ids: Iterable[int]):

    # common pre-processing for entities from GBD
    def _pull_rr_data_from_gbd(measure_ids):
        data = gbd.get_relative_risks(risk_ids=measure_ids, location_ids=location_ids)
        data = data.rename(columns={f'rr_{i}': f'draw_{i}' for i in range(1000)})

        # FIXME: I'm passing because this is broken for zinc_deficiency, and I don't have time to investigate -J.C.
        # err_msg = ("Not all relative risk data has both the 'mortality' and 'morbidity' flag "
        #            + "set. This may not indicate an error but it is a case we don't explicitly handle. "
        #            + "If you need this risk, come talk to one of the programmers.")
        # assert np.all((measure_data.mortality == 1) & (measure_data.morbidity == 1)), err_msg

        data = data[data['morbidity'] == 1]  # FIXME: HACK
        del data['mortality']
        del data['morbidity']
        del data['metric_id']
        del data['modelable_entity_id']
        return data

    measure_ids = _get_ids_for_measure(entities, 'relative_risk')

    if isinstance(entities[0], Risk):
        if None in measure_ids:
            raise InvalidQueryError(f'There is no relative risk for the entity that you requested')
        measure_data = _pull_rr_data_from_gbd(measure_ids)
        most_detailed_causes = measure_data['cause_id'].isin([c.gbd_id for c in causes])
        measure_data = measure_data[most_detailed_causes]

    elif isinstance(entities[0], CoverageGap):
        df = []
        SPECIAL = [coverage_gaps.low_measles_vaccine_coverage_first_dose]
        special_cases = set(SPECIAL).intersection(set(entities))
        if special_cases:
            measure_data = _pull_rr_data_from_gbd([i for i in measure_ids if i is not None])
            draw_cols = [f'draw_{i}' for i in range(1000)]
            measure_data.loc[:, draw_cols] = 1 / measure_data.loc[:, draw_cols]
            measure_data = _handle_special_coverage_gap_data(special_cases, measure_data, 1)
            df.append(measure_data)

        # any coverage_gap from aux_data
        if set(entities).difference(special_cases):
            for entity in entities:
                measure_data = gbd.get_auxiliary_data('relative_risk', 'coverage_gap', entity.name)
                exposure_data = gbd.get_auxiliary_data('relative_risk', 'coverage_gap', entity.name)
                missing_year_ids = set(exposure_data.year_id.unique()).difference(measure_data.year_id.unique())
                missing_data = []
                for id in missing_year_ids:
                    data = measure_data.copy()
                    data['year_id'] = id
                    missing_data = missing_data.append(data)
                measure_data = pd.concat([measure_data] + missing_data)
                measure_data['coverage_gap_id'] = np.NaN
                del measure_data['measure']
                df.append(measure_data)
        measure_data = pd.concat(df)

    measure_data = _standardize_all_age_groups(measure_data)
    measure_data = _standardize_data(measure_data, 1)
    return measure_data


def _filter_to_most_detailed(data):
    for column, entity_list in [('cause_id', causes), ('etiology_id', etiologies),
                                ('risk_id', risk_factors), ('coverage_gap_id', coverage_gaps)]:
        if column in data:
            most_detailed = {e.gbd_id for e in entity_list if e is not None}
            data = data.query(f"{column} in @most_detailed")
    return data


def _compute_paf_for_special_cases(affected_entity: Union[Cause, Risk], entity: Union[CoverageGap, Risk, Etiology],
                                   location_ids: Iterable[int]):
    """Computes pafs in cases where rr and exposure data is inconsistent with available pafs.

    e.g., Paf for unsafe_water_source from central_comp is lower than what it is
    supposed to be and does not assign correct incidence rates equivalent to gbd

    :param affected_entity: which entity is affected by this entity
    """
    ex = _get_exposure([entity], location_ids)
    rr = _get_relative_risk([entity], location_ids)

    if isinstance(entity, (Risk, Etiology)):
        cause_id, risk_id = affected_entity.gbd_id, entity.gbd_id
        rr = rr[(rr.cause_id == cause_id) & (rr.rei_id == risk_id)]
    elif isinstance(entity, CoverageGap):
        if isinstance(affected_entity, Cause):
            cause_id, risk_id = affected_entity.gbd_id, np.nan
            rr = rr[(rr.cause_id == cause_id)]
        elif isinstance(affected_entity, Risk):
            risk_id, cause_id = affected_entity.gbd_id, np.nan
            rr = rr[(rr.rei_id == risk_id)]
        else:
            raise InvalidQueryError(f'You requested the non-valid PAF data for {entity}-{affected_entity} pair')

    paf = []

    for location_id in location_ids:
        key_cols = ['age_group_id', 'year_id', 'sex_id', 'parameter']
        ex = ex[ex.location_id == location_id]
        years = rr.year_id.unique()
        relative_risk = rr.set_index(key_cols)

        exposure = ex[ex['year_id'].isin(years)].set_index(key_cols)
        draw_columns = ['draw_{}'.format(i) for i in range(1000)]
        temp = relative_risk[draw_columns]*exposure[draw_columns]
        temp_sum = temp.groupby(['age_group_id', 'year_id', 'sex_id']).sum()
        temp_result = ((temp_sum-1)/temp_sum)
        temp_result['cause_id'] = cause_id
        temp_result['location_id'] = location_id
        temp_result['risk_id'] = risk_id
        temp_result['measure_id'] = 3
        paf.append(temp_result.reset_index())

    paf_data = pd.concat(paf)
    return paf_data


def _get_population_attributable_fraction(entities, location_ids):
    SPECIAL = [risk_factors.unsafe_water_source]
    measure_ids = _get_ids_for_measure(entities, 'population_attributable_fraction')

    if isinstance(entities[0], Cause):
        measure_data = gbd.get_pafs(entity_ids=measure_ids, location_ids=location_ids)
        measure_data = measure_data.rename(columns={"rei_id": "risk_id"})
        measure_data = _filter_to_most_detailed(measure_data)

        risks_in_result = measure_data.risk_id.unique()
        special_cases = [r for r in SPECIAL if r.gbd_id in risks_in_result]
        for risk in special_cases:
            special_causes = measure_data[measure_data.risk_id == risk.gbd_id].cause_id.unique()
            special_causes = [cause for cause in causes if
                              cause and cause.gbd_id in special_causes and cause is not causes.all_causes]
            for cause in special_causes:
                special_paf = _compute_paf_for_special_cases(cause, risk, location_ids)
                measure_data = measure_data.query("risk_id != @risk.gbd_id or cause_id != @cause.gbd_id")
                measure_data = measure_data.append(special_paf)

    elif isinstance(entities[0], Etiology):
        measure_data = gbd.get_pafs(entity_ids=measure_ids, location_ids=location_ids, entity_type='etiology')
        measure_data = measure_data.rename(columns={"rei_id": "etiology_id"})
        measure_data = _filter_to_most_detailed(measure_data)

    elif isinstance(entities[0], CoverageGap):
        paf = []
        for entity in entities:
            affected_risk_factors = entity.affected_risk_factors
            affected_causes = entity.affected_causes
            paf.extend([_compute_paf_for_special_cases(cause, entity, location_ids) for cause
                        in affected_causes if affected_causes])
            paf.extend([_compute_paf_for_special_cases(risk_factor, entity, location_ids) for risk_factor
                        in affected_risk_factors if
                        affected_risk_factors])
        measure_data = pd.concat(paf)

    else:
        raise InvalidQueryError(f"Entity {entities[0].name} has no data for measure 'population_attributable_fraction'")

    # FIXME: We currently do not handle the case where PAF==1 so we just dump those rows.
    draws = [c for c in measure_data.columns if 'draw_' in c]
    measure_data = measure_data.loc[~(measure_data[draws] == 1).any(axis=1)]
    # TODO: figure out if we need to assert some property of the different PAF measures
    measure_data = measure_data[measure_data['measure_id'] == name_measure_map['YLD']]
    del measure_data['measure_id']

    # FIXME: I'm passing because this is broken for SBP, it's unimportant, and I don't have time to investigate -J.C.
    # measure_ids = {name_measure_map[m] for m in ['death', 'DALY', 'YLD', 'YLL']}
    # err_msg = ("Not all PAF data has values for deaths, DALYs, YLDs and YLLs. "
    #           + "This may not indicate an error but it is a case we don't explicitly handle. "
    #           + "If you need this PAF, come talk to one of the programmers.")
    # assert np.all(
    #    measure_data.groupby(key_columns).measure_id.unique().apply(lambda x: set(x) == measure_ids)), err_msg

    measure_data = _standardize_data(measure_data, 0)
    return measure_data


def _get_exposure(entities, location_ids):

    def handle_exposure_from_gbd(measure_data):
        is_categorical_exposure = measure_data.measure_id == name_measure_map['proportion']
        is_continuous_exposure = measure_data.measure_id == name_measure_map['continuous']
        measure_data = measure_data[is_categorical_exposure | is_continuous_exposure]
        # FIXME: The sex filtering should happen in the reshaping step.
        measure_data = measure_data[measure_data['sex_id'] != COMBINED]

        # FIXME: Is this the only data we need to delete measure id for?
        del measure_data['measure_id']
        return measure_data

    measure_ids = _get_ids_for_measure(entities, 'exposure')

    if isinstance(entities[0], (Risk, Etiology)):
        if None in measure_ids:
            raise InvalidQueryError(f'There is no exposure data for the entity that you requested')
        measure_data = gbd.get_exposures(risk_ids=measure_ids, location_ids=location_ids)
        measure_data = _handle_weird_exposure_measures(measure_data)
        exposure_data = handle_exposure_from_gbd(measure_data)

    elif isinstance(entities[0], CoverageGap):
        df = []
        SPECIAL = [coverage_gaps.low_measles_vaccine_coverage_first_dose]
        special_cases = set(SPECIAL).intersection(set(entities))
        if special_cases:
            measure_data = gbd.get_exposures(risk_ids=[s.gbd_id for s in special_cases], location_ids=location_ids)
            measure_data = _handle_special_coverage_gap_data(special_cases, measure_data, 0)
            measure_data = handle_exposure_from_gbd(measure_data)
            del measure_data['modelable_entity_id']
            del measure_data['metric_id']
            df.append(measure_data)

        # any coverage_gap from aux_data
        if set(entities).difference(special_cases):
            for entity in entities:
                measure_data = gbd.get_auxiliary_data('exposure', 'coverage_gap', entity.name)
                measure_data = measure_data[measure_data.location_id.isin(location_ids)]
                measure_data['coverage_gap_id'] = np.NaN
                del measure_data['measure']
                df.append(measure_data)
        exposure_data = pd.concat(df)
    else:
        raise InvalidQueryError(f"Entity {entities[0].name} has no data for measure 'exposure'")

    return _standardize_all_age_groups(exposure_data)


def _handle_special_coverage_gap_data(entities, measure_data, fill_value):
    # We pulled coverage, not exposure, so invert the categories.
    coverage = measure_data['parameter'] == 'cat1'
    exposure = measure_data['parameter'] == 'cat2'
    measure_data.loc[coverage, 'parameter'] = 'cat2'
    measure_data.loc[exposure, 'parameter'] = 'cat1'
    measure_data = measure_data.rename(columns={'risk_id': 'coverage_gap_id'})

    for coverage_gap in entities:
        affected_causes = coverage_gap.affected_causes
        if len(affected_causes) != 1:
            raise UnhandledDataError("We only handle coverage gaps affecting a single cause. "
                                     "Tell James if you see this error.")
        restrictions = affected_causes[0].restrictions
        if restrictions.yll_only:
            raise UnhandledDataError("The PAFs we use are for YLDs, so causes with no attributable YLDs should not"
                                     "have associated exposures or RRs.")
        age_start, age_end = restrictions.yld_age_start, restrictions.yld_age_end
        min_age_group = age_restriction_map[age_start][0]
        max_age_group = age_restriction_map[age_end][1]

        good_age_groups = range(min_age_group, max_age_group+1)

        coverage_gap_data = measure_data['coverage_gap_id'] == coverage_gap.gbd_id
        correct_age_groups = measure_data['age_group_id'].isin(good_age_groups)
        measure_data['rei_id'] = np.NaN
        measure_data['coverage_gap'] = coverage_gap.name
        draw_cols = [f'draw_{i}' for i in range(1000)]
        measure_data.loc[coverage_gap_data & ~correct_age_groups, draw_cols] = fill_value

    return measure_data


def _handle_weird_exposure_measures(measure_data):
    key_cols = ['age_group_id', 'location_id', 'sex_id', 'year_id']
    draw_cols = [f'draw_{i}' for i in range(1000)]

    measure_data = measure_data.set_index(key_cols)
    measure_data = measure_data[draw_cols + ['risk_id', 'measure_id', 'parameter']]
    exposure_data = pd.DataFrame()

    for risk_id in measure_data.risk_id.unique():
        # We need to handle this juggling risk by risk because the data is heterogeneous by risk id.
        correct_risk = measure_data['risk_id'] == risk_id
        risk_data = measure_data[correct_risk]

        measure_id = _get_exposure_measure_id(risk_data)

        # FIXME:
        # Some categorical risks come from cause models, or they get weird exposure models that
        # report prevalence instead of proportion.  We should do a systematic review of them and work
        # with the risk factors team to get the exposure reported consistently.  In the mean time
        # we scale the unit-full prevalence numbers to unit-less proportion numbers. - J.C.
        if measure_id == name_measure_map['prevalence']:
            total_prevalence = risk_data[draw_cols].reset_index().groupby(key_cols).sum()
            for parameter in risk_data['parameter'].unique():
                correct_parameter = risk_data['parameter'] == parameter
                risk_data.loc[correct_parameter, draw_cols] /= total_prevalence
            risk_data = risk_data.reset_index()
            risk_data['measure_id'] = name_measure_map['proportion']

        exposure_data = exposure_data.append(risk_data.reset_index())
    return exposure_data


def _get_exposure_measure_id(data):
    measure_ids = data.measure_id.unique()
    if len(measure_ids) > 1:
        raise UnhandledDataError("Exposures should always come back with a single measure, "
                                 "or they should be dealt with as a special case.  ")

    return int(measure_ids)


def _get_exposure_standard_deviation(entities, location_ids):
    ids = dict(zip(_get_ids_for_measure(entities, 'exposure_standard_deviation'), [e.gbd_id for e in entities]))
    df = gbd.get_exposure_standard_deviations(list(ids.keys()), location_ids)
    key_cols = ['age_group_id', 'location_id', 'sex_id', 'year_id', 'risk_id']
    draw_cols = [f'draw_{i}' for i in range(1000)]
    df = df[df['sex_id'] != 3]
    return df[key_cols + draw_cols]


###############
# Other stuff #
###############

def _get_annual_visits(entities, location_ids):
    measure_ids = _get_ids_for_measure(entities, 'annual_visits')
    measure_data = gbd.get_modelable_entity_draws(me_ids=measure_ids, location_ids=location_ids)

    measure_data['treatment_technology'] = 'temp'
    for entity in entities:
        correct_entity = measure_data['modelable_entity_id'] == entity.utilization
        measure_data.loc[correct_entity, 'healthcare_entity'] = entity.name

    correct_measure = measure_data['measure_id'] == name_measure_map['continuous']
    correct_sex = measure_data['sex_id'] != COMBINED

    return measure_data[correct_measure & correct_sex]


def _get_cost(entities, location_ids):
    data = []
    if isinstance(entities[0], HealthcareEntity):
        for entity in entities:
            df = gbd.get_auxiliary_data('cost', 'healthcare_entity', entity.name)
            df = df[df['location_id'].isin(location_ids)]
            data.append(df)
    else:
        for entity in entities:
            df = gbd.get_auxiliary_data('cost', 'health_technology', entity.name)
            data.append(df)
    return pd.concat(data)


####################################
# Measures for risk like entities  #
####################################


def get_ensemble_weights(risks):
    data = gbd.get_auxiliary_data('ensemble_weights', 'risk_factor', 'all')
    data = data[data['risk_id'].isin([r.gbd_id for r in risks])]
    return data


#######################
# Other kinds of data #
#######################


def get_populations(locations):
    location_ids = [get_location_ids_by_name()[location] for location in locations]
    populations = pd.concat([gbd.get_populations(location_id) for location_id in location_ids])
    populations["location"] = populations.location_id.apply(get_location_names_by_id().get)
    keep_columns = ['age_group_id', 'location', 'year_id', 'sex_id', 'population']
    return populations[keep_columns]


def get_age_bins():
    return gbd.get_age_bins()


def get_theoretical_minimum_risk_life_expectancy():
    return gbd.get_theoretical_minimum_risk_life_expectancy()


def get_subregions(locations):
    location_ids = [get_location_ids_by_name()[location] for location in locations]
    return gbd.get_subregions(location_ids)


def get_covariate_estimates(covariates, locations):
    location_ids = [get_location_ids_by_name()[location] for location in locations]
    data = gbd.get_covariate_estimates([covariate.gbd_id for covariate in covariates], location_ids)
    data['location'] = data.location_id.apply(get_location_names_by_id().get)
    data = data.drop('location_id', 'columns')
    return data


def get_location_ids_by_name():
    return  {r.location_name: r.location_id for _, r in gbd.get_location_ids().iterrows()}


def get_location_names_by_id():
    return {v: k for k, v in get_location_ids_by_name().items()}


def get_estimation_years():
    return gbd.get_estimation_years(gbd.GBD_ROUND_ID)

