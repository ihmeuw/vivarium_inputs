# ~/ceam/ceam/gbd_data/__init__.py

from ceam import config
from ceam_inputs import gbd_ms_functions as functions
from ceam_inputs import distributions

# TODO: None of these functions can be run from an ipython notebook. ipython notebook is not currently able to submit Stata scripts. See if there is a way to make ipython notebooks submit Stata scripts
from ceam_public_health.util.risk import RiskEffect

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
    return functions.load_data_from_cache(functions.get_modelable_entity_draws, 'rate', location_id=config.getint('simulation_parameters', 'location_id'), year_start=config.getint('simulation_parameters', 'year_start'), year_end=config.getint('simulation_parameters', 'year_end'), measure=15, me_id=modelable_entity_id)


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
    return functions.load_data_from_cache(functions.get_modelable_entity_draws, 'remission', location_id=config.getint('simulation_parameters', 'location_id'), year_start=config.getint('simulation_parameters', 'year_start'), year_end=config.getint('simulation_parameters', 'year_end'), measure=7, me_id=modelable_entity_id)


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
    return functions.load_data_from_cache(functions.get_modelable_entity_draws, 'prevalence', location_id=config.getint('simulation_parameters', 'location_id'), year_start=config.getint('simulation_parameters', 'year_start'), year_end=config.getint('simulation_parameters', 'year_end'), measure=5, me_id=modelable_entity_id)


def get_disease_states(population, states):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')

    population = population.reset_index()
    population['simulant_id'] = population['index']
    condition_column = functions.load_data_from_cache(functions.assign_cause_at_beginning_of_simulation, col_name=None, simulants_df=population[['simulant_id', 'age', 'sex']], year_start=year_start, states=states)

    return condition_column


def get_all_cause_mortality_rate():
    """Get the all cause mortality rate.


    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'rate' columns
    """

    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    return functions.load_data_from_cache(functions.get_all_cause_mortality_rate, \
            'rate', \
            location_id,
            year_start,
            year_end,
            src_column='all_cause_mortality_rate_{draw}')

def get_cause_deleted_mortality_rate(list_of_csmrs):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    return functions.get_cause_deleted_mortality_rate(location_id=location_id, year_start=year_start, year_end=year_end, list_of_csmrs=list_of_csmrs)


def get_relative_risks(risk_id, cause_id, rr_type='morbidity'):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    funct_output = functions.load_data_from_cache(functions.get_relative_risks, col_name='rr', src_column='rr_{draw}', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=risk_id, cause_id=cause_id, rr_type=rr_type)

    # need to reshape the funct output since there can be multiple categories
    output = funct_output.pivot_table(index=['age', 'year', 'sex'], columns=[funct_output.parameter.values], values=['rr'])
    output.columns = output.columns.droplevel()
    output.reset_index(inplace=True)

    return output


def get_pafs(risk_id, cause_id):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    return functions.load_data_from_cache(functions.get_pafs, col_name='PAF', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=risk_id, cause_id=cause_id)


def get_exposures(risk_id):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    funct_output = functions.load_data_from_cache(functions.get_exposures, col_name='exposure', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=risk_id)
    
    # need to reshape the funct output since there can be multiple categories
    output = funct_output.pivot_table(index=['age', 'year', 'sex'], columns=[funct_output.parameter.values], values=['exposure'])
    output.columns = output.columns.droplevel()
    output.reset_index(inplace=True)

    return output


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


def get_etiology_specific_prevalence(eti_risk_id, cause_id, me_id):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    draw_number = config.getint('run_configuration', 'draw_number')
    return functions.load_data_from_cache(functions.get_etiology_specific_prevalence, location_id=location_id,
                                          year_start=year_start, year_end=year_end, eti_risk_id=eti_risk_id,
                                          cause_id=cause_id, me_id=me_id, col_name='prevalence')



def get_etiology_specific_incidence(eti_risk_id, cause_id, me_id):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    draw_number = config.getint('run_configuration', 'draw_number')
    return functions.load_data_from_cache(functions.get_etiology_specific_incidence, location_id=location_id,
                                          year_start=year_start, year_end=year_end, eti_risk_id=eti_risk_id,
                                          cause_id=cause_id, me_id=me_id, col_name='eti_inc')




def get_bmi_distributions():
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    draw = config.getint('run_configuration', 'draw_number')

    return distributions.get_bmi_distributions(location_id, year_start, year_end, draw)

def get_fpg_distributions():
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')
    draw = config.getint('run_configuration', 'draw_number')

    return distributions.get_fpg_distributions(location_id, year_start, year_end, draw)


def make_gbd_risk_effects(risk_id, causes, effect_function):
    return [RiskEffect(
        get_relative_risks(risk_id=risk_id, cause_id=cause_id),
        get_pafs(risk_id=risk_id, cause_id=cause_id),
        cause_name,
        effect_function)
        for cause_id, cause_name in causes]


def get_diarrhea_severity_split_excess_mortality(excess_mortality_dataframe, severity_split):
    return functions.get_diarrhea_severity_split_excess_mortality(excess_mortality_dataframe, severity_split)

def get_covariate_estimates(covariate_short_name):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')

    return functions.get_covariate_estimates(location_id, year_start, year_end, covariate_short_name) 

# End.
