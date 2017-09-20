import os

import pandas as pd

from vivarium.framework.engine import build_simulation_configuration
config = build_simulation_configuration({})
_inputs_config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'gbd_config.yaml')
config.update(_inputs_config_path, layer='base', source=_inputs_config_path)

# Make these top level imports until external references can be removed.
from ceam_inputs.gbd_mapping import (causes, risk_factors, sequelae, etiologies,
                                     healthcare_entities, treatment_technologies,
                                     meid, hid, cid, rid, scalar, UNKNOWN,
                                     UnknownEntityError, Cause)
from ceam_inputs import gbd, gbd_ms_functions as functions


def _get_gbd_draws(modelable_entity, measure, column_name):
    """
    Parameters
    ----------
    modelable_entity : ceam_inputs.gbd_mapping.CauseLike
        A mapping of the GBD ids for a modelable entity onto its various measures.
    measure : str
        The name of the requested measure.
    column_name: str
        The name of the column with the measure data in the output data table.

    Returns
    -------
    pandas.DataFrame :
        Table with columns 'age', 'sex', 'year' and `column_name`
    """
    gbd_id = modelable_entity[measure]
    if gbd_id is UNKNOWN:
        if modelable_entity.gbd_id is not None:
            import warnings
            warnings.warn('The mapping between {} and {} '.format(modelable_entity.name, measure)
                          + 'has not been verified.  We are using {} '.format(repr(modelable_entity.gbd_id))
                          + 'but you should verify that this is the correct id and update the gbd mapping.')
            gbd_id = modelable_entity.gbd_id
        else:
            raise UnknownEntityError('No mapping exists for cause {} and measure {}'.format(modelable_entity.name, measure))
    elif isinstance(gbd_id, scalar):  # We have a scalar value rather than an actual id.
        return gbd_id

    return functions.get_gbd_draws(location_id=config.input_data.location_id,
                                   measure=measure,
                                   gbd_id=gbd_id,
                                   gbd_round_id=config.input_data.gbd_round_id,
                                   publication_ids=config.input_data.gbd_publication_ids,
                                   draw_number=config.run_configuration.input_draw_number,
                                   column_name=column_name)


def get_excess_mortality(cause):
    """Get excess mortality associated with a modelable entity.

    Parameters
    ----------
    cause :  ceam_inputs.gbd_mapping.CauseLike
        A mapping of the GBD ids for a modelable entity onto its various measures.

    Returns
    -------
    pandas.DataFrame or float
        Table with 'age', 'sex', 'year' and 'rate' columns
    """
    if isinstance(cause.excess_mortality, cid):
        csmr = get_cause_specific_mortality(cause).set_index(['age', 'sex', 'year'])
        prevalence = get_prevalence(cause).set_index(['age', 'sex', 'year'])
        prevalence.columns = ['rate']
        df = (csmr/prevalence).dropna()
        df[prevalence == 0] = 0
        return df.reset_index()
    else:
        return _get_gbd_draws(modelable_entity=cause, measure='excess_mortality', column_name='rate')


def get_incidence(cause):
    """Get incidence rates for a modelable entity.

    Parameters
    ----------
    cause :  ceam_inputs.gbd_mapping.CauseLike
        A mapping of the GBD ids for a modelable entity onto its various measures.

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'rate' columns
    """
    return _get_gbd_draws(modelable_entity=cause, measure='incidence', column_name='rate')


def get_cause_specific_mortality(cause):
    """Get excess mortality associated with a modelable entity.

    Parameters
    ----------
    cause : ceam_inputs.gbd_mapping.CauseLike
        A mapping of the GBD ids for a modelable entity onto its various measures.

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'rate' columns
    """
    return functions.get_cause_specific_mortality(cause_id=cause.gbd_id,
                                                  location_id=config.input_data.location_id,
                                                  gbd_round_id=config.input_data.gbd_round_id,
                                                  draw_number=config.run_configuration.input_draw_number,
                                                  publication_ids=config.input_data.gbd_publication_ids)


def get_remission(cause):
    """Get remission rates for a modelable entity.

    Parameters
    ----------
    cause :  ceam_inputs.gbd_mapping.CauseLike
        A mapping of the GBD ids for a modelable entity onto its various measures.

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'rate' columns
    """
    return _get_gbd_draws(modelable_entity=cause, measure='remission', column_name='remission')


def get_duration(cause):
    """Get duration times for a modelable entity.

    Parameters
    ----------
    cause :  ceam_inputs.gbd_mapping.CauseLike
        A mapping of the GBD ids for a modelable entity onto its various measures.

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'duration' columns
    """
    if cause.duration is UNKNOWN:
        raise UnknownEntityError('No mapping exists for cause {} and measure {}'.format(cause.name, 'duration'))
    else:
        return pd.Timedelta(days=cause.duration)


def get_proportion(modelable_entity):
    """Get proportion data for a modelable entity. This is used for entities that represent
    outcome splits like severities of heart failure after an infarction.

    Parameters
    ----------
    modelable_entity : ceam_inputs.gbd_mapping.CauseLike
        A mapping of the GBD ids for a modelable entity onto its various measures.

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'proportion' columns
    """
    return _get_gbd_draws(modelable_entity=modelable_entity,  measure='proportion', column_name='proportion')


def get_age_bins():
    """Retrieves the age bin structure the GBD uses for demographic classification."""
    return gbd.get_age_bins(gbd_round_id=config.input_data.gbd_round_id)


def get_prevalence(cause):
    """Get prevalence data for a modelable entity.

    Parameters
    ----------
    cause : ceam_inputs.gbd_mapping.CauseLike
        A mapping of the GBD ids for a modelable entity onto its various measures.

    Returns
    -------
    pandas.DataFrame
        Table with 'age', 'sex', 'year' and 'prevalence' columns
    """
    return _get_gbd_draws(modelable_entity=cause,  measure='prevalence', column_name='prevalence')


def get_relative_risks(risk, cause, rr_type='morbidity'):
    """Get the relative risk for a cause risk pair

    Parameters
    ----------
    risk: ceam_inputs.gbd_mapping.Risk
    cause: ceam_inputs.gbd_mapping.Cause
    rr_type: {'morbidity', 'mortality'}

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_relative_risks(location_id=config.input_data.location_id,
                                        risk_id=risk.gbd_id,
                                        cause_id=cause.gbd_id,
                                        gbd_round_id=config.input_data.gbd_round_id,
                                        rr_type=rr_type,
                                        draw_number=config.run_configuration.input_draw_number)


def get_pafs(risk, cause, paf_type='morbidity'):
    """Get the population attributable fraction for a cause risk pair.

    Parameters
    ----------
    risk: ceam_inputs.gbd_mapping.Risk
    cause: ceam_inputs.gbd_mapping.Cause
    paf_type: {'morbidity', 'mortality'}

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_pafs(location_id=config.input_data.location_id,
                              risk_id=risk.gbd_id,
                              cause_id=cause.gbd_id,
                              gbd_round_id=config.input_data.gbd_round_id,
                              paf_type=paf_type,
                              draw_number=config.run_configuration.input_draw_number)


def get_exposure_means(risk):
    """Gets the exposure distribution mean for a given risk.

    Parameters
    ----------
    risk: ceam_inputs.gbd_mapping.Risk

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_exposures(location_id=config.input_data.location_id,
                                   risk_id=risk.gbd_id,
                                   gbd_round_id=config.input_data.gbd_round_id,
                                   draw_number=config.run_configuration.input_draw_number)


def get_exposure_standard_errors(_):
    """Gets the exposure distribution standard error for a given risk.

    Parameters
    ----------
    risk: ceam_inputs.gbd_mapping.Risk

    Returns
    -------
    pandas.DataFrame
    """
    # TODO : I still need to generate the standard deviations for the continuous risks.  So stub here for now. J.C.
    pass


def get_populations(location_id, year=-1, sex='All'):
    """Gets the demographic structure for a population.

    Parameters
    ----------
    location_id: int
    year: int
    sex: {'All', 'Male', 'Female', 'Both'}

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_populations(location_id=location_id,
                                     year=year,
                                     sex=sex,
                                     gbd_round_id=config.input_data.gbd_round_id)


def get_age_specific_fertility_rates():
    """Gets fertility rates broken down by age.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_age_specific_fertility_rates(location_id=config.input_data.location_id,
                                                      gbd_round_id=config.input_data.gbd_round_id)


def get_bmi_distribution_parameters():
    """Gets the parameters for the body mass index exposure distribution (a 4-parameter beta distribution).

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_bmi_distribution_parameters(location_id=config.input_data.location_id,
                                                     gbd_round_id=config.input_data.gbd_round_id,
                                                     draw_number=config.run_configuration.input_draw_number)


def get_fpg_distribution_parameters():
    """Gets the parameters for the fasting plasma glucose exposure distribution.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_fpg_distribution_parameters(location_id=config.input_data.location_id,
                                                     gbd_round_id=config.input_data.gbd_round_id,
                                                     draw_number=config.run_configuration.input_draw_number,
                                                     use_subregions=config.input_data.use_subregions)


def get_annual_live_births(location_id, year, sex_id=3):
    """Gets the live births in a given location and year.

    Parameters
    ----------
    location_id: int
    year: int
    sex_id: {1, 2, 3}

    Returns
    -------
    float
        Average live births.
    """
    return functions.get_covariate_estimates(covariate_name_short='live_births_by_sex',
                                             location_id=location_id,
                                             year_id=year,
                                             sex_id=sex_id)['mean_value']


def get_sbp_distribution():
    """Gets the parameters for the high systolic blood pressure exposure distribution.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_sbp_mean_sd(location_id=config.input_data.location_id,
                                     gbd_round_id=config.input_data.gbd_round_id,
                                     draw_number=config.run_configuration.input_draw_number)


def get_post_mi_heart_failure_proportion_draws():
    """Gets the proportion of heart failure following myocardial infarction.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_post_mi_heart_failure_proportion_draws(location_id=config.input_data.location_id,
                                                                gbd_round_id=config.input_data.gbd_round_id,
                                                                publication_ids=config.input_data.gbd_publication_ids,
                                                                draw_number=config.run_configuration.input_draw_number)


def get_angina_proportions():
    """Gets the proportion of angina following myocardial infarction.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_angina_proportions(gbd_round_id=config.input_data.gbd_round_id,
                                            draw_number=config.run_configuration.input_draw_number)


def get_asympt_ihd_proportions():
    """Gets the proportion of asymptomatic ischemic heart disease following myocardial infarction.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_asympt_ihd_proportions(location_id=config.input_data.location_id,
                                                gbd_round_id=config.input_data.gbd_round_id,
                                                publication_ids=config.input_data.gbd_publication_ids,
                                                draw_number=config.run_configuration.input_draw_number)


def get_subregions(location_id):
    """Gets a list of subregions associated with the given location.

    Parameters
    ----------
    location_id: int

    Returns
    -------
    [int]
        Subregions of the given location.
    """
    return gbd.get_subregions(location_id)


def get_severity_splits(parent, child):
    """Gets the proportion of the parent cause cases represented by the child cause.

    Parameters
    ----------
    parent: ceam_inputs.gbd_mapping.Cause
    child: ceam_inputs.gbd_mapping.CauseLike

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_severity_splits(parent_meid=parent.incidence,
                                         child_meid=child.proportion,
                                         gbd_round_id=config.input_data.gbd_round_id,
                                         draw_number=config.run_configuration.input_draw_number)


def get_disability_weight(cause):
    """Gets the disability weight associated with the given cause-like entity.

    Parameters
    ----------
    cause: ceam_inputs.gbd_mapping.Sequela or ceam_inputs.gbd_mapping.SeveritySplit

    Returns
    -------
    float or pandas.DataFrame
    """
    if cause.disability_weight is UNKNOWN:
        if isinstance(cause, Cause):
            import warnings
            warnings.warn('Cause-level disability weights are not implemented. '
                          'Disability is specified at the sequela level. You can specify a '
                          'scalar value to use in ceam_inputs.gbd_mapping.causes.  Returning 0.')
            return 0
        raise UnknownEntityError('No mapping exists between cause {} and measure disability weight'.format(cause.name))
    elif isinstance(cause.disability_weight, scalar):
        return cause.disability_weight
    else:
        return functions.get_disability_weight(cause,
                                               gbd_round_id=config.input_data.gbd_round_id,
                                               draw_number=config.run_configuration.input_draw_number)


def get_rota_vaccine_coverage():
    """Gets the background amount of rota vaccine coverage.

    Returns
    -------
    pandas.DataFrame"""
    return functions.get_rota_vaccine_coverage(location_id=config.input_data.location_id,
                                               gbd_round_id=config.input_data.gbd_round_id,
                                               draw_number=config.run_configuration.input_draw_number)


def get_ors_pafs():
    """Gets the population attributable fraction of diarrhea deaths due to lack of oral rehydration salts solution.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_ors_pafs(location_id=config.input_data.location_id,
                                  gbd_round_id=config.input_data.gbd_round_id,
                                  draw_number=config.run_configuration.input_draw_number)


def get_ors_relative_risks():
    """Gets the relative risk of lack of oral rehydration salts solution on diarrhea excess mortality.

    Returns
    -------
    float
    """
    return functions.get_ors_relative_risks(gbd_round_id=config.input_data.gbd_round_id,
                                            draw_number=config.run_configuration.input_draw_number)


def get_ors_exposures():
    """Get the exposure to lack of oral rehydration salts solution (1 - ORS coverage).

    Returns
    pandas.DataFrame
    """
    return functions.get_ors_exposures(location_id=config.input_data.location_id,
                                       gbd_round_id=config.input_data.gbd_round_id,
                                       draw_number=config.run_configuration.input_draw_number)


def get_life_table():
    """Gets the life expectancy table.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_life_table(gbd_round_id=config.input_data.gbd_round_id)


def get_outpatient_visit_costs():
    """Gets the cost table for outpatient visits.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_outpatient_visit_costs(gbd_round_id=config.input_data.gbd_round_id)


def get_inpatient_visit_costs():
    """Gets the cost table for outpatient visits.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_inpatient_visit_costs(gbd_round_id=config.input_data.gbd_round_id)


def get_hypertension_drug_costs():
    """Gets a table of the daily costs for several common hypertension drugs.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_hypertension_drug_costs(gbd_round_id=config.input_data.gbd_round_id)


def load_risk_correlation_matrices():
    """Gets the matrix of correlation coefficients for risk factors.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.load_risk_correlation_matrices(location_id=config.input_data.location_id,
                                                    gbd_round_id=config.input_data.gbd_round_id)


def get_mediation_factors(risk, cause):
    """Gets the total mediation factor for the given risk cause pair.

    Returns
    -------
    float or pandas.DataFrame
    """
    return functions.get_mediation_factors(risk_id=risk.gbd_id,
                                           cause_id=cause.gbd_id,
                                           gbd_round_id=config.input_data.gbd_round_id,
                                           draw_number=config.run_configuration.input_draw_number)


def get_dtp3_coverage():
    """Gets the Diphtheria-tetanus-pertussis immunization coverage.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_dtp3_coverage(location_id=config.input_data.location_id,
                                       gbd_round_id=config.input_data.gbd_round_id,
                                       draw_number=config.run_configuration.input_draw_number)


def get_rota_vaccine_protection():
    """Gets rota vaccine protection estimates.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_rota_vaccine_protection(location_id=config.input_data.location_id,
                                                 gbd_round_id=config.input_data.gbd_round_id,
                                                 draw_number=config.run_configuration.input_draw_number)


def get_rota_vaccine_rrs():
    """Gets the relative risk of lack of rota vaccine on rotaviral entiritis incidence.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_rota_vaccine_rrs(location_id=config.input_data.location_id,
                                          gbd_round_id=config.input_data.gbd_round_id,
                                          draw_number=config.run_configuration.input_draw_number)


# FIXME: Why are there two of these?
def get_diarrhea_visit_costs():
    """Gets the cost of a healthcare visit due to diarrhea.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_diarrhea_visit_costs(location_id=config.input_data.location_id,
                                              gbd_round_id=config.input_data.gbd_round_id,
                                              draw_number=config.run_configuration.input_draw_number)


def get_diarrhea_costs():
    """Gets the cost of a healthcare visit due to diarrhea.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_diarrhea_costs(location_id=config.input_data.location_id,
                                        gbd_round_id=config.input_data.gbd_round_id,
                                        draw_number=config.run_configuration.input_draw_number)


def get_ors_costs():
    """Gets the daily cost associated with oral rehydration salt solution treatment.

    Returns
    -------
    pandas.DataFrame
    """
    return functions.get_ors_costs(location_id=config.input_data.location_id,
                                   gbd_round_id=config.input_data.gbd_round_id,
                                   draw_number=config.run_configuration.input_draw_number)
