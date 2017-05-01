import os

from ceam import config
_config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'gbd_config.yaml')
config.load(_config_path, layer='base', source=_config_path)

import joblib
from joblib import Memory

import pandas as pd
from datetime import datetime

from ceam_inputs import gbd_ms_functions as functions
from ceam_inputs import distributions
from ceam_inputs.util import get_cache_directory, gbd_year_range
from ceam_inputs.gbd_mapping import meid

memory = Memory(cachedir=get_cache_directory(), verbose=1)



def get_excess_mortality(modelable_entity_id):
    """Get excess mortality associated with a modelable entity.

    Parameters
    ----------
    modelable_entity_id : int
                          The entity to retrieve

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'rate' columns
    """
    year_start, year_end = gbd_year_range()

    df = functions.load_data_from_cache(
            functions.get_modelable_entity_draws,
            'rate',
            location_id=config.simulation_parameters.location_id,
            year_start=year_start,
            year_end=year_end,
            measure=9,
            me_id=modelable_entity_id
        )
    df.metadata = {'modelable_entity_id': modelable_entity_id}
    return df

def get_incidence(modelable_entity_id):
    """Get incidence rates for a modelable entity.

    Parameters
    ----------
    modelable_entity_id : int
                          The entity to retrieve

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'rate' columns
    """
    year_start, year_end = gbd_year_range()

    df = functions.load_data_from_cache(
            functions.get_modelable_entity_draws,
            'rate',
            location_id=config.simulation_parameters.location_id,
            year_start=year_start,
            year_end=year_end,
            measure=6,
            me_id=modelable_entity_id
        )
    df.metadata = {'modelable_entity_id': modelable_entity_id}
    return df


def get_cause_specific_mortality(modelable_entity_id):
    """Get excess mortality associated with a modelable entity.

    Parameters
    ----------
    modelable_entity_id : int
                          The entity to retrieve

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'rate' columns
    """
    year_start, year_end = gbd_year_range()

    df = functions.load_data_from_cache(
            functions.get_modelable_entity_draws, 'rate',
            location_id=config.simulation_parameters.location_id,
            year_start=year_start,
            year_end=year_end,
            measure=15,
            me_id=modelable_entity_id
        )
    df.metadata = {'modelable_entity_id': modelable_entity_id}
    return df


def get_remission(modelable_entity_id):
    """Get remission rates for a modelable entity.

    Parameters
    ----------
    modelable_entity_id : int
                          The entity to retrieve

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'rate' columns
    """
    year_start, year_end = gbd_year_range()

    df = functions.load_data_from_cache(
            functions.get_modelable_entity_draws, 'remission',
            location_id=config.simulation_parameters.location_id,
            year_start=year_start,
            year_end=year_end,
            measure=7,
            me_id=modelable_entity_id
        )
    df.metadata = {'modelable_entity_id': modelable_entity_id}
    return df


def get_duration_in_days(modelable_entity_id):
    """Get duration of disease for a modelable entity in days.

    Parameters
    ----------
    modelable_entity_id : int
                          The entity to retrieve

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'duration' columns
    """

    remission = get_remission(modelable_entity_id)

    duration = remission.copy()

    duration['duration'] = (1 / duration['remission']) *365

    duration.metadata = {'modelable_entity_id': modelable_entity_id}

    return duration[['year', 'age', 'duration', 'sex']]


def get_continuous(modelable_entity_id):
    """Get the continuous measure from a modelable entity. This measure is used
    for things like the distribution of BMI in a population.

    Parameters
    ----------
    modelable_entity_id : int
                          The entity to retrieve

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'value' columns
    """
    year_start, year_end = gbd_year_range()

    df = functions.load_data_from_cache(
            functions.get_modelable_entity_draws,
            'value',
            location_id=config.simulation_parameters.location_id,
            year_start=year_start,
            year_end=year_end,
            measure=19,
            me_id=modelable_entity_id
        )
    df.metadata = {'modelable_entity_id': modelable_entity_id}
    return df


def get_proportion(modelable_entity_id):
    """Get proportion data for a modelable entity. This is used for entities that represent
    outcome splits like severities of heart failure after an infarction.

    Parameters
    ----------
    modelable_entity_id : int
                          The entity to retrieve

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'proportion' columns
    """
    year_start, year_end = gbd_year_range()

    df = functions.load_data_from_cache(
            functions.get_modelable_entity_draws,
            'proportion',
            location_id=config.simulation_parameters.location_id,
            year_start=year_start,
            year_end=year_end,
            measure=18,
            me_id=modelable_entity_id
        )
    df.metadata = {'modelable_entity_id': modelable_entity_id}
    return df

@memory.cache
def get_age_bins():
    from db_tools import ezfuncs # This import is here to make the dependency on db_tools optional if the data is available from cache
    return ezfuncs.query('''select age_group_id, age_group_years_start, age_group_years_end, age_group_name from age_group''', conn_def='shared')

def get_prevalence(modelable_entity_id):
    """Get prevalence data for a modelable entity.

    Parameters
    ----------
    modelable_entity_id : int
                          The entity to retrieve

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'prevalence' columns
    """
    year_start, year_end = gbd_year_range()

    df = functions.load_data_from_cache(
            functions.get_modelable_entity_draws,
            'prevalence',
            location_id=config.simulation_parameters.location_id,
            year_start=year_start,
            year_end=year_end,
            measure=5,
            me_id=modelable_entity_id
        )
    df.metadata = {'modelable_entity_id': modelable_entity_id}
    return df


def get_disease_states(population, states):
    location_id = config.simulation_parameters.location_id
    year_start = config.simulation_parameters.year_start

    population = population.reset_index()
    population['simulant_id'] = population['index']
    condition_column = functions.load_data_from_cache(functions.assign_cause_at_beginning_of_simulation, col_name=None, simulants_df=population[['simulant_id', 'age', 'sex']], year_start=year_start, states=states)

    return condition_column


def get_cause_deleted_mortality_rate(list_of_csmrs):
    # This sort is a because we don't want the cache to invalidate when
    # the csmrs come in in different orders but they aren't hashable by
    # standard python so we can't put them in a set.
    list_of_csmrs = sorted(list_of_csmrs, key=lambda x: joblib.hash(x))
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    gbd_round_id = config.simulation_parameters.gbd_round_id
    draw_number = config.run_configuration.draw_number
    return functions.get_cause_deleted_mortality_rate(location_id=location_id, year_start=year_start, year_end=year_end, list_of_csmrs=list_of_csmrs, gbd_round_id=gbd_round_id, draw_number=draw_number)


def get_relative_risks(risk_id, cause_id, rr_type='morbidity'):
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    gbd_round_id = config.simulation_parameters.gbd_round_id
    draw_number = config.run_configuration.draw_number
    funct_output = functions.load_data_from_cache(functions.get_relative_risks, col_name='rr', src_column='rr_{draw}', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=risk_id, cause_id=cause_id, gbd_round_id=gbd_round_id, draw_number=draw_number, rr_type=rr_type)

    # need to reshape the funct output since there can be multiple categories
    output = funct_output.pivot_table(index=['age', 'year', 'sex'], columns=[funct_output.parameter.values], values=['rr'])
    output.columns = output.columns.droplevel()
    output.reset_index(inplace=True)

    output.metadata = {'risk_id': risk_id, 'cause_id': cause_id}
    return output


def get_pafs(risk_id, cause_id):
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    gbd_round_id = config.simulation_parameters.gbd_round_id
    draw_number = config.run_configuration.draw_number
    df = functions.load_data_from_cache(functions.get_pafs, col_name='PAF', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=risk_id, cause_id=cause_id, gbd_round_id=gbd_round_id, draw_number=draw_number)
    df.metadata = {'risk_id': risk_id, 'cause_id': cause_id}
    return df


def get_exposures(risk_id):
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    gbd_round_id = config.simulation_parameters.gbd_round_id

    funct_output = functions.load_data_from_cache(functions.get_exposures, col_name='exposure', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=risk_id, gbd_round_id=gbd_round_id)

    # need to reshape the funct output since there can be multiple categories
    output = funct_output.pivot_table(index=['age', 'year', 'sex'], columns=[funct_output.parameter.values], values=['exposure'])
    output.columns = output.columns.droplevel()
    output.reset_index(inplace=True)

    output.metadata = {'risk_id': risk_id}
    return output


def generate_ceam_population(number_of_simulants, initial_age=None, time=None):
    location_id = config.simulation_parameters.location_id
    pop_age_start = config.simulation_parameters.pop_age_start
    pop_age_end = config.simulation_parameters.pop_age_end
    if time is None:
        year_start, year_end = gbd_year_range()
        time = datetime(year_start, 1, 1)
    population = functions.load_data_from_cache(functions.generate_ceam_population, col_name=None,
                                                location_id=location_id, time=time,
                                                number_of_simulants=number_of_simulants, initial_age=initial_age,
                                                pop_age_start=pop_age_start, pop_age_end=pop_age_end)
    population['sex'] = population['sex_id'].map(
        {1: 'Male', 2: 'Female'}).astype('category', categories=['Male', 'Female'])
    population['alive'] = True
    return population


def get_age_specific_fertility_rates():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    return functions.load_data_from_cache(functions.get_age_specific_fertility_rates, col_name=['mean_value', 'lower_value', 'upper_value'], src_column=['mean_value', 'lower_value', 'upper_value'], location_id=location_id, year_start=year_start, year_end=year_end)


def get_etiology_probability(etiology_name):
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw_number = config.run_configuration.draw_number
    return functions.load_data_from_cache(functions.get_etiology_probability, etiology_name=etiology_name, location_id=location_id, year_start=year_start, year_end=year_end, draw_number=draw_number)


def get_etiology_specific_prevalence(eti_risk_id, cause_id, me_id):
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    gbd_round_id = config.simulation_parameters.gbd_round_id
    draw_number = config.run_configuration.draw_number
    return functions.load_data_from_cache(functions.get_etiology_specific_prevalence, location_id=location_id,
                                          year_start=year_start, year_end=year_end, eti_risk_id=eti_risk_id,
                                          cause_id=cause_id, me_id=me_id, gbd_round_id=gbd_round_id, draw_number=draw_number, col_name='prevalence')



def get_etiology_specific_incidence(eti_risk_id, cause_id, me_id):
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    gbd_round_id = config.simulation_parameters.gbd_round_id
    draw_number = config.run_configuration.draw_number
    return functions.load_data_from_cache(functions.get_etiology_specific_incidence, location_id=location_id,
                                          year_start=year_start, year_end=year_end, eti_risk_id=eti_risk_id,
                                          cause_id=cause_id, me_id=me_id, gbd_round_id=gbd_round_id, draw_number=draw_number, col_name='eti_inc')




def get_bmi_distributions():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw = config.run_configuration.draw_number

    return distributions.get_bmi_distributions(location_id, year_start, year_end, draw)

def get_fpg_distributions():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw = config.run_configuration.draw_number

    return distributions.get_fpg_distributions(location_id, year_start, year_end, draw)

def make_gbd_disease_state(cause, dwell_time=0, side_effect_function=None):
    from ceam_public_health.components.disease import ExcessMortalityState
    if 'mortality' in cause:
        if isinstance(cause.mortality, meid):
            csmr = get_cause_specific_mortality(cause.mortality)
        else:
            csmr = cause.mortality
    else:
        csmr = pd.DataFrame()

    if 'disability_weight' in cause:
        draw = config.run_configuration.draw_number
        if isinstance(cause.disability_weight, meid):
            disability_weight = functions.get_disability_weight(draw, cause.disability_weight)
        else:
            disability_weight = cause.disability_weight
    else:
        disability_weight = 0.0

    if 'excess_mortality' in cause:
        if isinstance(cause.excess_mortality, meid):
            excess_mortality = get_excess_mortality(cause.excess_mortality)
        else:
            excess_mortality = cause.excess_mortality
    else:
        excess_mortality = 0.0

    if 'prevalence' in cause:
        if isinstance(cause.prevalence, meid):
            prevalence = get_prevalence(cause.prevalence)
        else:
            prevalence = cause.prevalence
    else:
        prevalence = 0.0

    return ExcessMortalityState(
            cause.name,
            dwell_time=dwell_time,
            disability_weight=disability_weight,
            excess_mortality_data=excess_mortality,
            prevalence_data=prevalence,
            csmr_data=csmr,
            side_effect_function=side_effect_function
        )


def get_diarrhea_severity_split_excess_mortality(excess_mortality_dataframe, severity_split):
    return functions.get_diarrhea_severity_split_excess_mortality(excess_mortality_dataframe, severity_split)

def get_covariate_estimates(covariate_short_name):
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()

    return functions.get_covariate_estimates(location_id, year_start, year_end, covariate_short_name) 

def get_ors_exposure():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw_number = config.run_configuration.draw_number

    return functions.load_data_from_cache(functions.get_ors_exposure, location_id=location_id, year_start=year_start, year_end=year_end, draw_number=draw_number, col_name=None)
