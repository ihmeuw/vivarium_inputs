import os

import pandas as pd
import joblib

from vivarium import config
_config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'gbd_config.yaml')
config.load(_config_path, layer='base', source=_config_path)

# Make these toplevel imports until external references can be removed.
from ceam_inputs.gbd_mapping import causes, risk_factors, meid, hid, cid
from ceam_inputs import gbd, risk_factor_correlation, gbd_ms_functions as functions
from ceam_inputs.util import gbd_year_range


def _get_gbd_draws(column_name, measure, gbd_id):
    year_start, year_end = gbd_year_range()
    draws = functions.get_gbd_draws(location_id=config.simulation_parameters.location_id,
                                                 year_start=year_start,
                                                 year_end=year_end,
                                                 measure=measure,
                                                 gbd_id=gbd_id,
                                                 gbd_round_id=config.simulation_parameters.gbd_round_id)

    df = functions.select_draw_data(draws, config.run_configuration.draw_number, column_name=column_name)
    df.metadata = {'gbd_id': gbd_id}

    return df


def get_excess_mortality(gbd_id):
    """Get excess mortality associated with a modelable entity.

    Parameters
    ----------
    gbd_id : int
                          The entity to retrieve

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'rate' columns
    """
    if isinstance(gbd_id, cid):
        csmr = get_cause_specific_mortality(gbd_id).set_index(['age', 'sex', 'year'])
        prevalence = get_prevalence(gbd_id).set_index(['age', 'sex', 'year'])
        prevalence.columns = ['rate']
        df = (csmr/prevalence).dropna()
        df[prevalence == 0] = 0
        return df.reset_index()
    else:
        return _get_gbd_draws(column_name='rate', measure=9, gbd_id=gbd_id)


def get_incidence(gbd_id):
    """Get incidence rates for a modelable entity.

    Parameters
    ----------
    gbd_id : int
                          The entity to retrieve

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'rate' columns
    """
    return _get_gbd_draws(column_name='rate', measure=6, gbd_id=gbd_id)


def get_cause_specific_mortality(cause_id):
    """Get excess mortality associated with a modelable entity.

    Parameters
    ----------
    cause_id : int
        The entity to retrieve

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'rate' columns
    """
    year_start, year_end = gbd_year_range()

    return functions.get_cause_specific_mortality(cause_id=cause_id,
                                                  location_id=config.simulation_parameters.location_id,
                                                  year_start=year_start,
                                                  year_end=year_end,
                                                  gbd_round_id=config.simulation_parameters.gbd_round_id,
                                                  draw_number=config.run_configuration.draw_number)


def get_remission(gbd_id):
    """Get remission rates for a modelable entity.

    Parameters
    ----------
    gbd_id : int
                          The entity to retrieve

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'rate' columns
    """
    return _get_gbd_draws(column_name='remission', measure=7, gbd_id=gbd_id)



def get_proportion(gbd_id):
    """Get proportion data for a modelable entity. This is used for entities that represent
    outcome splits like severities of heart failure after an infarction.

    Parameters
    ----------
    gbd_id : int
                          The entity to retrieve

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'proportion' columns
    """
    return _get_gbd_draws(column_name='proportion', measure=18, gbd_id=gbd_id)


def get_age_bins():
    return gbd.get_age_bins()


def get_prevalence(gbd_id):
    """Get prevalence data for a modelable entity.

    Parameters
    ----------
    gbd_id : int
                          The entity to retrieve

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'prevalence' columns
    """
    return _get_gbd_draws(column_name='prevalence', measure=5, gbd_id=gbd_id)


def get_relative_risks(risk_id, cause_id, rr_type='morbidity'):
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    gbd_round_id = config.simulation_parameters.gbd_round_id
    draw_number = config.run_configuration.draw_number
    draws = functions.get_relative_risks(location_id=location_id,
                                         year_start=year_start,
                                         year_end=year_end,
                                         risk_id=risk_id,
                                         cause_id=cause_id,
                                         gbd_round_id=gbd_round_id,
                                         draw_number=draw_number,
                                         rr_type=rr_type)
    funct_output = functions.select_draw_data(draws, draw_number, column_name='rr', src_column='rr_{draw}')

    # need to reshape the funct output since there can be multiple categories
    output = funct_output.pivot_table(index=['age', 'year', 'sex'],
                                      columns=[funct_output.parameter.values],
                                      values=['rr'])
    output.columns = output.columns.droplevel()
    output.reset_index(inplace=True)

    output.metadata = {'risk_id': risk_id, 'cause_id': cause_id}
    return output


def get_pafs(risk_id, cause_id, paf_type='morbidity'):
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    gbd_round_id = config.simulation_parameters.gbd_round_id
    draw_number = config.run_configuration.draw_number
    draws = functions.get_pafs(location_id=location_id,
                               year_start=year_start,
                               year_end=year_end,
                               risk_id=risk_id,
                               cause_id=cause_id,
                               gbd_round_id=gbd_round_id,
                               draw_number=draw_number,
                               paf_type=paf_type)
    df = functions.select_draw_data(draws, draw_number, column_name='PAF')
    df.metadata = {'risk_id': risk_id, 'cause_id': cause_id}
    return df


def get_exposures(risk_id):
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    gbd_round_id = config.simulation_parameters.gbd_round_id
    draw_number = config.run_configuration.draw_number
    draws = functions.get_exposures(location_id=location_id,
                                    year_start=year_start,
                                    year_end=year_end,
                                    risk_id=risk_id,
                                    gbd_round_id=gbd_round_id)
    funct_output = functions.select_draw_data(draws, draw_number, column_name='exposure')

    # need to reshape the funct output since there can be multiple categories
    output = funct_output.pivot_table(index=['age', 'year', 'sex'],
                                      columns=[funct_output.parameter.values],
                                      values=['exposure'])
    output.columns = output.columns.droplevel()
    output.reset_index(inplace=True)

    output.metadata = {'risk_id': risk_id}
    return output


def get_populations(location_id, year=-1, sex='All'):
    gbd_round_id = config.simulation_parameters.gbd_round_id
    return functions.get_populations(location_id=location_id,
                                     year=year,
                                     sex=sex,
                                     gbd_round_id=gbd_round_id)


def get_age_specific_fertility_rates():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    return functions.get_age_specific_fertility_rates(location_id=location_id,
                                                      year_start=year_start,
                                                      year_end=year_end)


def get_bmi_distribution_parameters():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw = config.run_configuration.draw_number

    return functions.get_bmi_distribution_parameters(location_id, year_start, year_end, draw)


def get_fpg_distribution_parameters():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw = config.run_configuration.draw_number

    return functions.get_fpg_distribution_parameters(location_id, year_start, year_end, draw)


def get_annual_live_births(location_id, year, sex_id=3):
    data = functions.get_covariate_estimates(covariate_name_short='live_births_by_sex',
                                             location_id=location_id,
                                             year_id=year,
                                             sex_id=sex_id)
    return data['mean_value']


def get_sbp_distribution():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw_number = config.run_configuration.draw_number
    draws = functions.get_sbp_mean_sd(location_id=location_id,
                                      year_start=year_start,
                                      year_end=year_end)
    return functions.select_draw_data(draws, draw_number,
                                      column_name=['log_mean', 'log_sd'],
                                      src_column=['log_mean_{draw}', 'log_sd_{draw}'])


def get_post_mi_heart_failure_proportion_draws():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw = config.run_configuration.draw_number
    draws = functions.get_post_mi_heart_failure_proportion_draws(location_id=location_id,
                                                                 year_start=year_start,
                                                                 year_end=year_end,
                                                                 draw_number=draw,
                                                                 gbd_round_id=config.simulation_parameters.gbd_round_id)
    return functions.select_draw_data(draws, draw,
                                      column_name='proportion',
                                      src_column='draw_{draw}')


def get_angina_proportions():
    draw_number = config.run_
    draws = functions.get_angina_proportions()

    return functions.select_draw_data(draws, draw_number,
                                      column_name='proportion',
                                      src_column='angina_prop')


def get_asympt_ihd_proportions():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw_number = config.run_configuration.draw_number
    draws = functions.get_asympt_ihd_proportions(location_id=location_id,
                                                 year_start=year_start,
                                                 year_end=year_end,
                                                 draw_number=draw_number)
    return functions.select_draw_data(draws, draw_number,
                                      column_name='proportion',
                                      src_column='asympt_prop_{draw}')


def get_subregions(location_id):
    return gbd.get_subregions(location_id)


def get_severity_splits(parent_meid, child_meid):
    draw_number = config.run_configuration.draw_number

    return functions.get_severity_splits(parent_meid=parent_meid, child_meid=child_meid, draw_number=draw_number)


def get_disability_weight(dis_weight_gbd_id=None, healthstate_id=None):
    return functions.get_disability_weight(config.run_configuration.draw_number,
                                           dis_weight_gbd_id, healthstate_id)


def get_rota_vaccine_coverage():
    year_start, year_end = gbd_year_range()
    # NOTE: There are no rota_vaccine_coverage estimates for GBD 2015, so we're pulling GBD 2016 estimates
    gbd_round_id = config.simulation_parameters.gbd_round_id
    if gbd_round_id == 3:
        gbd_round_id = 4
    draws = functions.get_rota_vaccine_coverage(location_id=config.simulation_parameters.location_id,
                                                year_start=year_start,
                                                year_end=year_end,
                                                gbd_round_id=gbd_round_id)
    return functions.select_draw_data(draws, config.run_configuration.draw_number, column_name='coverage')


def get_ors_pafs():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw_number = config.run_configuration.draw_number
    draws = functions.get_ors_pafs(location_id=location_id,
                                   year_start=year_start,
                                   year_end=year_end)
    return functions.select_draw_data(draws, draw_number,
                                      column_name='paf',
                                      src_column='paf_{draw}')


def get_ors_relative_risks():
    return functions.get_ors_relative_risks(config.run_configuration.draw_number)


def get_ors_exposures():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw_number = config.run_configuration.draw_number
    draws = functions.get_ors_exposures(location_id=location_id,
                                        year_start=year_start,
                                        year_end=year_end,
                                        draw_number=draw_number)
    funct_output = functions.select_draw_data(draws, draw_number, column_name='exp')

    output = funct_output.pivot_table(index=['age', 'year', 'sex'],
                                      columns=[funct_output.parameter.values], values=['exp'])
    output.columns = output.columns.droplevel()
    output.reset_index(inplace=True)
    return output


def get_diarrhea_visit_costs():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw_number = config.run_configuration.draw_number
    return functions.get_diarrhea_visit_costs(location_id, year_start, year_end, draw_number)


def get_life_table():
    return gbd.get_data_from_auxiliary_file('Life Table')


def get_outpatient_visit_costs():
    return gbd.get_data_from_auxiliary_file('Outpatient Visit Costs')


def get_inpatient_visit_costs():
    return gbd.get_data_from_auxiliary_file('Inpatient Visit Costs')


def get_hypertension_drug_costs():
    data = gbd.get_data_from_auxiliary_file('Hypertension Drug Costs')
    return data.set_index(['name'])


def load_risk_correlation_matrices():
    location_id = config.simulation_parameters.location_id
    return risk_factor_correlation.load_matrices(location_id)


def get_mediation_factors(risk_id, cause_id):
    draw_number = config.run_configuration.draw_number

    return functions.get_mediation_factors(risk_id, cause_id, draw_number)


def get_dtp3_coverage():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw_number = config.run_configuration.draw_number
    draws = functions.get_dtp3_coverage(location_id, year_start, year_end, draw_number)
    return functions.select_draw_data(draws, config.run_configuration.draw_number, column_name='coverage')


def get_rota_vaccine_protection():
    location_id = config.simulation_parameters.location_id
    draw_number = config.run_configuration.draw_number

    return functions.get_rota_vaccine_protection(location_id, draw_number)

def get_rota_vaccine_rrs():
    location_id = config.simulation_parameters.location_id
    draw_number = config.run_configuration.draw_number

    return functions.get_rota_vaccine_rrs(location_id, draw_number)

def get_diarrhea_costs():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw_number = config.run_configuration.draw_number
    return functions.get_diarrhea_costs(location_id, year_start, year_end, draw_number)


def get_ors_costs():
    location_id = config.simulation_parameters.location_id
    year_start, year_end = gbd_year_range()
    draw_number = config.run_configuration.draw_number
    return functions.get_ors_costs(location_id, year_start, year_end, draw_number)
