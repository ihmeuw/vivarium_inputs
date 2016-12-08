# ~/ceam/ceam/gbd_data/__init__.py

from ceam import config
from ceam_inputs import gbd_ms_functions as functions

# TODO: None of these functions can be run from an ipython notebook. ipython notebook is not currently able to submit Stata scripts. See if there is a way to make ipython notebooks submit Stata scripts

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
    return functions.load_data_from_cache(functions.get_modelable_entity_draws, 'rate', location_id=config.getint('simulation_parameters', 'location_id'), year_start=config.getint('simulation_parameters', 'year_start'), year_end=config.getint('simulation_parameters', 'year_end'), measure=9, me_id=modelable_entity_id)

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
    return functions.load_data_from_cache(functions.get_modelable_entity_draws, 'rate', location_id=config.getint('simulation_parameters', 'location_id'), year_start=config.getint('simulation_parameters', 'year_start'), year_end=config.getint('simulation_parameters', 'year_end'), measure=6, me_id=modelable_entity_id)


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
    return functions.load_data_from_cache(functions.get_modelable_entity_draws, 'rate', location_id=config.getint('simulation_parameters', 'location_id'), year_start=config.getint('simulation_parameters', 'year_start'), year_end=config.getint('simulation_parameters', 'year_end'), measure=7, me_id=modelable_entity_id)



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
    return functions.load_data_from_cache(functions.get_modelable_entity_draws, 'value', location_id=config.getint('simulation_parameters', 'location_id'), year_start=config.getint('simulation_parameters', 'year_start'), year_end=config.getint('simulation_parameters', 'year_end'), measure=19, me_id=modelable_entity_id)

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
    return functions.load_data_from_cache(functions.get_modelable_entity_draws, 'proportion', location_id=config.getint('simulation_parameters', 'location_id'), year_start=config.getint('simulation_parameters', 'year_start'), year_end=config.getint('simulation_parameters', 'year_end'), measure=18, me_id=modelable_entity_id)

def get_disease_states(population, states):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')

    population = population.reset_index()
    population['simulant_id'] = population['index']
    condition_column = functions.load_data_from_cache(functions.assign_cause_at_beginning_of_simulation, col_name=None, simulants_df=population[['simulant_id', 'age', 'sex']], location_id=location_id, year_start=year_start, states=states)
    condition_column = condition_column.set_index('simulant_id')

    return condition_column


def get_cause_deleted_mortality_rate(meids=[]):
    """Get the all cause mortality rate with the excess mortality rate of explicitly modeled
    modelable entity ids deleted out.

    Parameters
    ----------
    meids : [int]
            A list of modelable entity ids to delete from the all cause rate

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'rate' columns
    """

    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    return functions.load_data_from_cache(functions.get_cause_deleted_mortality_rate, \
            'cause_deleted_mortality_rate', \
            location_id,
            year_start,
            year_end,
            meids,
            src_column='cause_deleted_mortality_rate_{draw}')

def get_relative_risks(risk_id, cause_id):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    return functions.load_data_from_cache(functions.get_relative_risks, col_name='rr', src_column='rr_{draw}', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=risk_id, cause_id=cause_id)

def get_pafs(risk_id, cause_id):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    return functions.load_data_from_cache(functions.get_pafs, col_name='PAF', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=risk_id, cause_id=cause_id)

def get_exposures(risk_id):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    return functions.load_data_from_cache(functions.get_exposures, col_name='exposure', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=risk_id)

def generate_ceam_population(number_of_simulants, initial_age=None, year_start=None):
    location_id = config.getint('simulation_parameters', 'location_id')
    if year_start is None:
        year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    population = functions.load_data_from_cache(functions.generate_ceam_population, col_name=None, location_id=location_id, year_start=year_start, number_of_simulants=number_of_simulants, initial_age=initial_age)
    population['sex'] = population['sex_id'].map({1:'Male', 2:'Female'}).astype('category')
    population['alive'] = True
    return population

def get_age_specific_fertility_rates():
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    return functions.load_data_from_cache(functions.get_age_specific_fertility_rates, col_name=['mean_value', 'lower_value', 'upper_value'], src_column=['mean_value', 'lower_value', 'upper_value'], location_id=location_id, year_start=year_start, year_end=year_end)


def get_etiology_probability(etiology_name):
    return functions.load_data_from_cache(functions.get_etiology_probability, etiology_name=etiology_name)


def get_etiology_specific_prevalence(eti_risk_id, cause_id):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    return functions.load_data_from_cache(functions.get_etiology_specific_prevalence, location_id=location_id,
                                          year_start=year_start, year_end=year_end, eti_risk_id=eti_risk_id,
                                          cause_id=cause_id, col_name='draw_{}'.format(config.getint('run_configuration','draw_number')))


def get_etiology_specific_incidence(eti_risk_id, cause_id):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    return functions.load_data_from_cache(functions.get_etiology_specific_incidence, location_id=location_id,
                                          year_start=year_start, year_end=year_end, eti_risk_id=eti_risk_id,
                                          cause_id=cause_id, col_name='eti_inc')

# End.
