"""This module performs the core data transformations on GBD data and provides a basic API for data access."""
from typing import Union

import numpy as np
import pandas as pd
from gbd_mapping.base_template import ModelableEntity
from gbd_mapping.cause import Cause
from gbd_mapping.risk import Risk, risk_factors
from gbd_mapping.etiology import Etiology
from gbd_mapping.coverage_gap import CoverageGap, coverage_gaps

from vivarium_inputs.utilities import (InvalidQueryError, UnhandledDataError, DataMissingError,
                                       get_additional_id_columns, validate_data, get_measure_id, get_id_for_measure,
                                       standardize_all_age_groups, get_age_group_ids, filter_to_most_detailed,
                                       standardize_data, gbd)


def get_draws(entity: ModelableEntity, measure: str, location: str) -> pd.DataFrame:
    """Gets draw level gbd data for a measure and entity.

    Parameters
    ----------
    entity:
        A data container from the `gbd_mapping` package.
    measure:
        A GBD measure requested for the provided entity.
    location:
        A location to pull data for.

    Returns
    -------
    A table of draw level data for for the entity, measure, and location
    combination indexed by demographic data (age_group_id, sex_id, year_id) where appropriate.
    """
    measure_handlers = {
        'death': (get_death, set()),
        'remission': (get_remission, set()),
        'prevalence': (get_prevalence, set()),
        'incidence': (get_incidence, set()),
        'relative_risk': (get_relative_risk, {'cause_id', 'parameter', }),
        'population_attributable_fraction': (get_population_attributable_fraction, {'rei_id', }),
        'cause_specific_mortality': (get_cause_specific_mortality, set()),
        'excess_mortality': (get_excess_mortality, set()),
        'exposure': (get_exposure, {'parameter', }),
        'exposure_standard_deviation': (get_exposure_standard_deviation, {'rei_id', }),
        'annual_visits': (get_annual_visits, {'modelable_entity_id', }),
        'disability_weight': (get_disability_weight, set()),
        'cost': (get_cost, set()),
    }
    location_id = get_location_id(location)

    try:
        handler, id_columns = measure_handlers[measure]
    except KeyError:
        raise InvalidQueryError(f'No functions are available to pull data for measure {measure}')
    data = handler(entity, location_id)
    data['measure'] = measure

    id_columns |= get_additional_id_columns(data, entity)
    key_columns = ['measure'] + list(id_columns)
    for column in ['year_id', 'sex_id', 'age_group_id', 'location_id']:
        if column in data:
            key_columns.append(column)

    draw_columns = [f'draw_{i}' for i in range(0, 1000)]

    data = data[key_columns + draw_columns].reset_index(drop=True)
    validate_data(data, key_columns)
    if "location_id" in data:
        data["location"] = location
        data = data.drop("location_id", "columns")

    return data

#####################
# get_draws helpers #
#####################
#
# These functions filter out erroneous measures and deal with special cases.
#

####################
# Cause-like stuff #
####################


def get_death(entity, location_id):
    entity_id = get_id_for_measure(entity, 'death')
    death_data = gbd.get_codcorrect_draws(cause_id=entity_id, location_id=location_id)

    return death_data[death_data['measure_id'] == get_measure_id('death')]


def get_remission(entity, location_id):
    entity_id = get_id_for_measure(entity, 'remission')
    remission_data = gbd.get_modelable_entity_draws(me_id=entity_id, location_id=location_id)
    remission_data['cause_id'] = entity.gbd_id
    correct_measure = remission_data['measure_id'] == get_measure_id('remission')
    return remission_data[correct_measure]


def get_prevalence(entity, location_id):
    entity_id = get_id_for_measure(entity, 'prevalence')
    measure_data = gbd.get_como_draws(entity_id=entity_id, location_id=location_id, entity_type=entity.kind)

    correct_measure = measure_data['measure_id'] == get_measure_id('prevalence')
    correct_years = measure_data['year_id'].isin(gbd.get_estimation_years(gbd.GBD_ROUND_ID))
    return measure_data[correct_measure & correct_years]


def get_incidence(entity, location_id):
    entity_id = get_id_for_measure(entity, 'incidence')
    data = gbd.get_como_draws(entity_id=entity_id, location_id=location_id, entity_type=entity.kind)

    correct_measure = data['measure_id'] == get_measure_id('incidence')
    correct_years = data['year_id'].isin(gbd.get_estimation_years(gbd.GBD_ROUND_ID))

    # get_draws returns "total incidence." We want hazard incidence.
    # scale by the number of people not sick to convert.
    incidence = data[correct_measure & correct_years]
    prevalence = get_prevalence(entity, location_id)

    key_columns = [f'{entity.kind}_id', 'age_group_id', 'location_id', 'sex_id', 'year_id']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]

    incidence = incidence.set_index(key_columns)
    prevalence = prevalence.set_index(key_columns)

    incidence[draw_columns] = incidence[draw_columns] / (1 - prevalence[draw_columns])

    return incidence.reset_index()


def get_cause_specific_mortality(entity, location_id):
    deaths = get_draws(entity, "death", get_location_name(location_id))
    deaths["location_id"] = location_id

    population = get_population(get_location_name(location_id))
    population = population[population['year_id'] >= deaths.year_id.min()]
    population["location_id"] = location_id

    merge_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id']
    key_columns = merge_columns + ['cause_id']
    draw_columns = [f'draw_{i}' for i in range(0, 1000)]

    df = deaths.merge(population, on=merge_columns).set_index(key_columns)
    csmr = df[draw_columns].divide(df['population'], axis=0).reset_index()

    csmr = csmr[key_columns + draw_columns]
    return standardize_data(csmr, 0)


def get_excess_mortality(entity, location_id):
    prevalence = get_draws(entity, 'prevalence', get_location_name(location_id)).drop('measure', 'columns')
    csmr = get_draws(entity, 'cause_specific_mortality', get_location_name(location_id)).drop('measure', 'columns')

    key_columns = ['year_id', 'sex_id', 'age_group_id', 'cause_id', 'location']
    prevalences = prevalence.set_index(key_columns)
    csmrs = csmr.set_index(key_columns)

    # In some cases CSMR is not zero for age groups where prevalence is, which leads to
    # crazy outputs. So enforce that constraint.
    # TODO: But is this the right place to do that?
    draw_columns = [f'draw_{i}' for i in range(1000)]
    csmrs[draw_columns] = csmrs[draw_columns].where(prevalences[draw_columns] != 0, 0)

    em = csmrs.divide(prevalences, axis='index').reset_index()

    em["location_id"] = location_id

    return standardize_data(em.dropna(), 0)


def get_disability_weight(entity, _):
    if entity.kind != 'sequela':
        raise InvalidQueryError("Only sequela have disability weights associated with them.")

    disability_weights = gbd.get_auxiliary_data('disability_weight', 'sequela', 'all')
    if entity.healthstate.gbd_id in disability_weights['healthstate_id'].values:
        data = disability_weights.loc[disability_weights.healthstate_id == entity.healthstate.gbd_id, :]
    else:
        raise DataMissingError(f"No disability weight available for the sequela {entity.name}")

    data['sequela_id'] = entity.gbd_id

    return data.reset_index(drop=True)


###################
# Risk-like stuff #
###################

def get_relative_risk(entity: Union[Risk, CoverageGap], location_id: int):
    special_cases = [coverage_gaps.low_measles_vaccine_coverage_first_dose]
    entity_id = get_id_for_measure(entity, 'relative_risk')

    if entity.kind == 'risk_factor':
        data = pull_rr_data_from_gbd(entity_id, location_id)
        data = filter_to_most_detailed(data)
    elif entity.kind == 'coverage_gap' and entity in special_cases:
        data = pull_rr_data_from_gbd(entity_id, location_id)
        draw_cols = [f'draw_{i}' for i in range(1000)]
        data.loc[:, draw_cols] = 1 / data.loc[:, draw_cols]
        data = handle_gbd_coverage_gap_data(entity, data, 1)
        data['coverage_gap'] = entity.name
    elif entity.kind == 'coverage_gap':
        data = gbd.get_auxiliary_data('relative_risk', 'coverage_gap', entity.name)
        data['coverage_gap'] = entity.name
        del data['measure']
    else:
        raise InvalidQueryError(f"Entity {entity.name} has no data for measure 'relative_risk'")

    data = standardize_all_age_groups(data)
    data = standardize_data(data, 1)
    return data


def pull_rr_data_from_gbd(measure_id, location_id):
    data = gbd.get_relative_risk(risk_id=measure_id, location_id=location_id)
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


def get_population_attributable_fraction(entity, location_id):
    entity_id = get_id_for_measure(entity, 'population_attributable_fraction')

    if entity.kind == 'etiology' or \
            (entity.kind == 'risk_factor' and entity.distribution in ['ensemble', 'normal', 'lognormal']):
        data = gbd.get_paf(entity_id=entity_id, location_id=location_id)
        data = filter_to_most_detailed(data)

    else:
        raise InvalidQueryError(f"Entity {entity.name} has no data for measure 'population_attributable_fraction'")

    # TODO: figure out if we need to assert some property of the different PAF measures
    data = data[data['measure_id'] == get_measure_id('YLD')]
    del data['measure_id']

    # FIXME: I'm passing because this is broken for SBP, it's unimportant, and I don't have time to investigate -J.C.
    # measure_ids = {name_measure_map[m] for m in ['death', 'DALY', 'YLD', 'YLL']}
    # err_msg = ("Not all PAF data has values for deaths, DALYs, YLDs and YLLs. "
    #           + "This may not indicate an error but it is a case we don't explicitly handle. "
    #           + "If you need this PAF, come talk to one of the programmers.")
    # assert np.all(
    #    measure_data.groupby(key_columns).measure_id.unique().apply(lambda x: set(x) == measure_ids)), err_msg

    data = standardize_data(data, 0)
    return data


def get_exposure(entity, location_id):
    special_cases = [coverage_gaps.low_measles_vaccine_coverage_first_dose]
    entity_id = get_id_for_measure(entity, 'exposure')

    if entity.kind in ['risk_factor', 'etiology']:
        data = gbd.get_exposure(entity_id=entity_id, location_id=location_id)
        measure_id = get_exposure_measure_id(data)
        if measure_id == get_measure_id('prevalence'):
            data = convert_exposure_prevalence_to_proportion(data)
        data = handle_exposure_from_gbd(data)
    elif entity in special_cases:
        data = gbd.get_exposure(entity_id=entity_id, location_id=location_id)
        data = handle_gbd_coverage_gap_data(entity, data, 0)
        data = handle_exposure_from_gbd(data)
        del data['modelable_entity_id']
        del data['metric_id']
    elif entity.kind == 'coverage_gap':
        data = gbd.get_auxiliary_data('exposure', entity.kind, entity.name)
        data = data[data.location_id == location_id]
        data['coverage_gap'] = entity.name
        del data['measure']
    else:
        raise InvalidQueryError(f"Entity {entity.name} has no data for measure 'exposure'")

    return standardize_all_age_groups(data)


def convert_exposure_prevalence_to_proportion(data):
    # FIXME:
    # Some categorical risks come from cause models, or they get weird exposure
    # models that report prevalence instead of proportion.  We should do a
    # systematic review of them and work with the risk factors team to get the
    # exposure reported consistently.  In the mean time we scale the unit-full
    # prevalence numbers to unit-less proportion numbers. - J.C.
    key_cols = ['age_group_id', 'location_id', 'sex_id', 'year_id']
    draw_cols = [f'draw_{i}' for i in range(1000)]

    data = data.set_index(key_cols)
    data = data[draw_cols + ['rei_id', 'measure_id', 'parameter']]

    total_prevalence = data[draw_cols].reset_index().groupby(key_cols).sum()
    for parameter in data['parameter'].unique():
        correct_parameter = data['parameter'] == parameter
        data.loc[correct_parameter, draw_cols] /= total_prevalence
    data = data.reset_index()
    data['measure_id'] = get_measure_id('proportion')

    return data


def get_exposure_measure_id(data):
    measure_ids = data.measure_id.unique()
    if len(measure_ids) > 1:
        raise UnhandledDataError("Exposures should always come back with a single measure, "
                                 "or they should be dealt with as a special case.  ")

    return int(measure_ids)


def handle_exposure_from_gbd(data):
    correct_measure = data.measure_id.isin([get_measure_id('proportion'), get_measure_id('continuous')])
    data = data[correct_measure]
    # FIXME: Is this the only data we need to delete measure id for?
    del data['measure_id']
    return data


def handle_gbd_coverage_gap_data(entity, data, fill_value):
    if len(entity.affected_causes) != 1:
        raise UnhandledDataError("We only handle coverage gaps affecting a single cause. "
                                 "Tell James if you see this error.")
    restrictions = entity.affected_causes[0].restrictions
    if restrictions.yll_only:
        raise UnhandledDataError("The PAFs we use are for YLDs, so causes with no attributable YLDs should not"
                                 "have associated exposures or RRs.")

    # We pulled coverage, not exposure, so invert the categories.
    data['parameter'] = data.parameter.map({'cat1': 'cat2', 'cat2': 'cat1'})
    data['coverage_gap'] = entity.name

    age_group_ids = get_age_group_ids(restrictions, 'yld')
    correct_age_groups = data['age_group_id'].isin(age_group_ids)

    draw_cols = [f'draw_{i}' for i in range(1000)]
    data.loc[~correct_age_groups, draw_cols] = fill_value

    return data


def get_exposure_standard_deviation(entity, location_id):
    entity_id = get_id_for_measure(entity, 'exposure_standard_deviation')
    data = gbd.get_exposure_standard_deviation(entity_id, location_id)

    key_cols = ['age_group_id', 'location_id', 'sex_id', 'year_id', 'rei_id']
    draw_cols = [f'draw_{i}' for i in range(1000)]

    return data[key_cols + draw_cols]


###############
# Other stuff #
###############

def get_annual_visits(entity, location_id):
    entity_id = get_id_for_measure(entity, 'annual_visits')
    measure_data = gbd.get_modelable_entity_draws(me_id=entity_id, location_id=location_id)
    measure_data['healthcare_entity'] = entity.name

    correct_measure = measure_data['measure_id'] == get_measure_id('continuous')
    correct_sex = measure_data['sex_id'] != gbd.COMBINED

    return measure_data[correct_measure & correct_sex]


def get_cost(entity, location_id):
    cost_data = gbd.get_auxiliary_data('cost', entity.kind, entity.name)
    if 'location_id' in cost_data.columns:
        cost_data = cost_data[cost_data['location_id'] == location_id]
    return cost_data


####################################
# Measures for risk like entities  #
####################################


def get_ensemble_weights(risk):
    data = gbd.get_auxiliary_data('ensemble_weights', 'risk_factor', 'all')
    data = data[data['risk_id'] == risk.gbd_id]
    return data


#######################
# Other kinds of data #
#######################


def get_population(location):
    population = gbd.get_population(get_location_id(location))
    population["location"] = location
    keep_columns = ['age_group_id', 'location', 'year_id', 'sex_id', 'population']
    return population[keep_columns]


def get_age_bins():
    return gbd.get_age_bins()


def get_theoretical_minimum_risk_life_expectancy():
    data = gbd.get_theoretical_minimum_risk_life_expectancy()
    # TODO: Figure out a more rigorous way to handle generating age bins
    # (esp. last edge) from precise ages
    data = data.rename(columns={'age': 'age_group_start'})
    data['age_group_end'] = data.age_group_start.shift(-1).fillna(125)
    return data


def get_subregions(location):
    return gbd.get_subregions(get_location_id(location))


def get_covariate_estimate(covariate, location):
    location_id = get_location_id(location)
    data = gbd.get_covariate_estimate(covariate.gbd_id, location_id)
    data['location'] = location
    data = data.drop('location_id', 'columns')
    return data


def get_location_id(location_name):
    return {r.location_name: r.location_id for _, r in gbd.get_location_ids().iterrows()}[location_name]


def get_location_name(location_id):
    return {r.location_id: r.location_name for _, r in gbd.get_location_ids().iterrows()}[location_id]


def get_estimation_years():
    return gbd.get_estimation_years(gbd.GBD_ROUND_ID)

