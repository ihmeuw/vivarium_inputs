import os
from datetime import datetime

import pandas as pd
import joblib
from joblib import Memory

from ceam_inputs import distributions, gbd_ms_functions as functions
from ceam_inputs.util import get_cache_directory, gbd_year_range
from ceam_inputs.gbd_mapping import meid

from ceam import config

from ceam_public_health.util.risk import RiskEffect

_config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'gbd_config.yaml')
config.load(_config_path, layer='base', source=_config_path)

memory = Memory(cachedir=get_cache_directory(), verbose=1)


def _get_modelable_entity_draws(column_name, measure, modelable_entity_id):
    year_start, year_end = gbd_year_range()

    df = functions.load_data_from_cache(
        functions.get_modelable_entity_draws,
        col_name=column_name,
        location_id=config.simulation_parameters.location_id,
        year_start=year_start,
        year_end=year_end,
        measure=measure,
        me_id=modelable_entity_id
    )
    df.metadata = {'modelable_entity_id': modelable_entity_id}

    return df


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
    return _get_modelable_entity_draws(column_name='rate', measure=9, modelable_entity_id=modelable_entity_id)


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
    return _get_modelable_entity_draws(column_name='rate', measure=6, modelable_entity_id=modelable_entity_id)


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
    return _get_modelable_entity_draws(column_name='rate', measure=15, modelable_entity_id=modelable_entity_id)


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
    return _get_modelable_entity_draws(column_name='remission', measure=7, modelable_entity_id=modelable_entity_id)


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
    return _get_modelable_entity_draws(column_name='value', measure=19, modelable_entity_id=modelable_entity_id)


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
    return _get_modelable_entity_draws(column_name='proportion', measure=19, modelable_entity_id=modelable_entity_id)


@memory.cache
def get_age_bins():
    # This import is here to make the dependency on db_tools optional if the data is available from cache
    from db_tools import ezfuncs
    return ezfuncs.query('''
        SELECT age_group_id, age_group_years_start, age_group_years_end, age_group_name 
        FROM age_group''', conn_def='shared')


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
    return _get_modelable_entity_draws(column_name='prevalence', measure=5, modelable_entity_id=modelable_entity_id)
    year_start, year_end = gbd_year_range()


def get_disease_states(population, states):
    location_id = config.simulation_parameters.location_id
    year_start = config.simulation_parameters.year_start

    population = population.reset_index()
    population['simulant_id'] = population['index']
    condition_column = functions.load_data_from_cache(functions.assign_cause_at_beginning_of_simulation,
                                                      col_name=None,
                                                      simulants_df=population[['simulant_id', 'age', 'sex']],
                                                      location_id=location_id,
                                                      year_start=year_start,
                                                      states=states)

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


def get_pafs(risk_id, cause_id, paf_type='morbidity'):
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    gbd_round_id = config.simulation_parameters.gbd_round_id
    draw_number = config.run_configuration.draw_number

    df = functions.load_data_from_cache(functions.get_pafs, col_name='PAF', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=risk_id, cause_id=cause_id, gbd_round_id=gbd_round_id, draw_number=draw_number, paf_type=paf_type)

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


def make_gbd_risk_effects(risk_id, causes, effect_function):#, risk_name):
    return [RiskEffect(
        get_relative_risks(risk_id=risk_id, cause_id=cause_id),
        get_pafs(risk_id=risk_id, cause_id=cause_id),
        cause_name, #risk_name,
        effect_function)
        for cause_id, cause_name in causes]

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

def get_annual_live_births(location_id, year, sex_id=3):
    data = functions.load_data_from_cache(functions.get_covariate_estimates,
                                          covariate_name_short='live_births_by_sex',
                                          location_id=location_id,
                                          year_id=year,
                                          sex_id=sex_id,
                                          col_name=None)
    return data['mean_value']



def get_ors_exposure():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw_number = config.run_configuration.draw_number

    return functions.load_data_from_cache(functions.get_ors_exposure,
                                          location_id=location_id,
                                          year_start=year_start,
                                          year_end=year_end,
                                          draw_number=draw_number, col_name=None)


def get_sbp_distribution():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()

    return functions.load_data_from_cache(functions.get_sbp_mean_sd,
                                          col_name=['log_mean', 'log_sd'],
                                          src_column=['log_mean_{draw}', 'log_sd_{draw}'],
                                          location_id=location_id,
                                          year_start=year_start,
                                          year_end=year_end)


def get_post_mi_heart_failure_proportion_draws():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw = config.run_configuration.draw_number

    return functions.load_data_from_cache(functions.get_post_mi_heart_failure_proportion_draws,
                                          col_name='proportion',
                                          src_column='draw_{draw}',
                                          location_id=location_id,
                                          year_start=year_start,
                                          year_end=year_end,
                                          draw_number=draw)


def get_angina_proportions():
    return functions.load_data_from_cache(get_angina_proportions,
                                          col_name='proportion',
                                          src_column='angina_prop')


def get_asympt_ihd_proportions():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw = config.run_configuration.draw_number
    return functions.load_data_from_cache(functions.get_asympt_ihd_proportions,
                                          col_name='proportion',
                                          src_column='asympt_prop_{draw}',
                                          location_id=location_id,
                                          year_start=year_start,
                                          year_end=year_end,
                                          draw_number=draw)

def get_utilization_proportion():
    year_start, year_end = gbd_year_range()
    location_id = config.simulation_parameters.location_id
    # me_id 9458 is 'out patient visits'
    # measure 18 is 'Proportion'
    # TODO: Currently this is monthly, not anually
    return functions.load_data_from_cache(functions.get_modelable_entity_draws,
                                          col_name='utilization_proportion',
                                          year_start=year_start,
                                          year_end=year_end,
                                          location_id=location_id,
                                          measure=18,
                                          me_id=9458)


def get_ckd_prevalence(stage, location_id, year_start, year_end):
    stage_map = {'three': 2018, 'four': 2019, 'five': 2022}
    return functions.get_modelable_entity_draws(location_id, year_start,
                                                year_end, 5, stage_map[stage])


def get_bmi_category_prevalence(category, location_id, year_start, year_end):
    category_map = {'obese': 9364, 'overweight': 9363}
    return functions.get_modelable_entity_draws(location_id, year_start,
                                                year_end, 18, category_map[category])


def get_smoking_exposure(location_id, year_start, year_end):
    return functions.get_modelable_entity_draws(location_id, year_start,
                                                year_end, 18, 8941)


def assign_subregions(population, location_id, year_start):
    return functions.assign_subregions(population, location_id, year_start)


def assign_cause_at_beginning_of_simulation(population, location_id, year, states={}):
    return functions.assign_cause_at_beginning_of_simulation(population, location_id,
                                                             year, states=states)


def get_severity_splits(parent_meid, child_meid):
    draw_number = config.run_configuration.draw_number

    return functions.get_severity_splits(parent_meid=parent_meid, child_meid=child_meid, draw_number=draw_number)
    
def get_severe_diarrhea_excess_mortality():
    draw_number = config.run_configuration.draw_number
    severe_diarrhea_proportion = get_severity_splits(1181, 2610) 

    return functions.get_severe_diarrhea_excess_mortality(excess_mortality_dataframe=get_excess_mortality(1181), severe_diarrhea_proportion=severe_diarrhea_proportion)


def make_age_group_1_to_4_rates_constant(df):
    """
    Takes a dataframe where incidence or excess mortality rates are
        being set at age group midpoints and reassigns the values
        that are set at the age group 1 - 4 midpoint (3) and assigns
        those values to the age group end and age group start. That
        way our interpolation spline will yield constant values in
        between the age group start and age group end for the 1 to
        4 age group

    Parameters
    ----------
    df: pd.DataFrame()
        df with excess mortality or incidence rates for each age, 
        sex, year, and location
    """
    age_bins = get_age_bins()
    new_rows = pd.DataFrame()
    
    assert 3 in df.age.values, "the input dataframe needs to" + \
                               " simulants that are at the" + \
                               " age group midpoint"
    
    assert [1, 2, 4, 5] not in df.age.values, "the input df" + \
        "should only have simulants that are at the age" + \
        "group midpoint for the 1 to 4 age group"
    

    # get estimates for the age 1-4 age group (select at the age
    #     group midpoint)
    for index, row in df.loc[df.age == 3].iterrows():
        year = (row['year'])
        age_group_max = age_bins.set_index('age_group_name').get_value('1 to 4', 'age_group_years_end')  # the age group max for the 1-4 age group
        age = age_group_max
        if 'rate' in df.columns:
            value_col = 'rate'
            value = (row['rate'])
        elif 'eti_inc' in df.columns:
            value_col = 'eti_inc'
            value = (row['eti_inc'])
        sex = (row['sex'])
        # create a new line in the daataframe
        line = pd.DataFrame({"year": year,
                            "age": 5, value_col: value, "sex": sex},
                            index=[index+1])
        new_rows = new_rows.append(line)
        
    df = pd.concat([df, new_rows]).sort_values(
        by=['year', 'sex', 'age']).reset_index(drop=True)
    age_group_min = age_bins.set_index('age_group_name').get_value('1 to 4', 'age_group_years_start')  # the age group min for the 1-4 age group
    df.loc[df.age == 3, 'age'] = age_group_min
    
    return df


def get_disability_weight(dis_weight_modelable_entity_id=None, healthstate_id=None):
    draw_number = config.run_configuration.draw_number

    return functions.get_disability_weight(draw_number, dis_weight_modelable_entity_id, healthstate_id)


def get_rota_vaccine_coverage():
    """Get rota vaccine coverage.

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'coverage' columns
    """
    year_start, year_end = gbd_year_range()
    # NOTE: There are no rota_vaccine_coverage estimates for GBD 2015, so we're pulling GBD 2016 estimates
    gbd_round_id = config.simulation_parameters.gbd_round_id
    if gbd_round_id == 3:
        gbd_round_id = 4

    df = functions.load_data_from_cache(
            functions.get_rota_vaccine_coverage,
            'coverage',
            location_id=config.simulation_parameters.location_id,
            year_start=year_start,
            year_end=year_end,
            gbd_round_id=gbd_round_id
        )
    return df

