from vivarium.config_tree import ConfigTree  # Just for typing info.

# Make these top level imports until external references can be removed.
from ceam_inputs.gbd_mapping import *

from ceam_inputs import core, gbd_ms_auxiliary_functions as aux, gbd_ms_functions as functions
from ceam_inputs.util import get_input_config


def _clean_and_filter_data(data, draw_number, column_name):
    key_cols = [c for c in data.columns if 'draw' not in c]
    data = data[key_cols + [f'draw_{draw_number}']]
    data = aux.get_age_group_midpoint_from_age_group_id(data)
    return functions.select_draw_data(data, draw_number, column_name)


####################################
# Measures for cause like entities #
####################################

def get_prevalence(entity, override_config=None):
    config = get_input_config(override_config)
    data = core.get_prevalences([entity.gbd_id], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.input_data.draw_number, 'prevalence')


def get_incidence(entity, override_config: ConfigTree=None):
    config = get_input_config(override_config)
    data = core.get_incidences([entity.gbd_id], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.input_data.draw_number, 'rate')


def get_remission(cause, override_config=None):
    config = get_input_config(override_config)
    data = core.get_remissions([cause], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.input_data.draw_number, 'rate')


def get_cause_specific_mortality(cause, override_config=None):
    config = get_input_config(override_config)
    data = core.get_cause_specific_mortalities([cause], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.input_data.draw_number, 'rate')


def get_excess_mortality(cause, override_config: ConfigTree=None):
    config = get_input_config(override_config)
    data = core.get_excess_mortalities([cause], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.input_data.draw_number, 'rate')


# FIXME: This function almost certainly will not work.
def get_disability_weight(sequela, override_config=None):
    config = get_input_config(override_config)
    data = core.get_disability_weights([sequela], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.input_data.draw_number, 'disability_weight')


####################################
# Measures for risk like entities  #
####################################


def get_relative_risks(risk, cause, override_config=None):
    config = get_input_config(override_config)
    data = core.get_relative_risks([risk], [config.input_data.location_id])
    data = data[data['cause_id'] == cause.gbd_id]
    return _clean_and_filter_data(data, config.input_data.draw_number, 'relative_risk')


def get_exposure_mean(risk, override_config=None):
    config = get_input_config(override_config)
    data = core.get_exposure_means([risk], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.input_data.draw_number, 'exposure_mean')


def get_exposure_standard_deviation(risk, override_config=None):
    config = get_input_config(override_config)
    data = core.get_exposure_standard_deviations([risk], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.input_data.draw_number, 'exposure_standard_deviation')


def get_population_attributable_fraction(risk, cause, override_config=None):
    config = get_input_config(override_config)
    data = core.get_population_attributable_fractions([risk], [config.input_data.location_id])
    data = data[data['cause_id'] == cause.gbd_id]
    return _clean_and_filter_data(data, config.input_data.draw_number, 'population_attributable_fraction')


def get_ensemble_weights(risk, override_config=None):
    config = get_input_config(override_config)
    return core.get_ensemble_weights([risk], [config.input_data.location_id])


def get_mediation_factors(risk, cause, override_config=None):
    config = get_input_config(override_config)
    data = core.get_mediation_factors([risk], [config.input_data.location_id])
    return data[data['cause_id'] == cause.gbd_id]


#######################
# Other kinds of data #
#######################


def get_populations(override_config=None):
    config = get_input_config(override_config)
    return core.get_populations([config.input_data.location_id])


def get_age_bins():
    return core.get_age_bins()


def get_life_table(override_config=None):
    config = get_input_config(override_config)
    return core.get_life_tables([config.input_data.location_id])


def get_subregions(override_config=None):
    config = get_input_config(override_config)
    return core.get_subregions([config.input_data.location_id])


def get_outpatient_visit_costs(override_config=None):
    config = get_input_config(override_config)
    return core.get_costs([healthcare_entities.outpatient_visits], [config.input_data.location_id])


def get_inpatient_visit_costs(override_config=None):
    config = get_input_config(override_config)
    return core.get_costs([healthcare_entities.inpatient_visits], [config.input_data.location_id])


def get_hypertension_drug_costs(override_config=None):
    config = get_input_config(override_config)
    return core.get_costs([treatment_technologies.hypertension_drugs], [config.input_data.location_id])


def get_age_specific_fertility_rates(override_config=None):
    config = get_input_config(override_config)
    return core.get_covariate_estimates([covariates.age_specific_fertility_rate], [config.input_data.location_id])


def get_live_births_by_sex(override_config=None):
    config = get_input_config(override_config)
    return core.get_covariate_estimates([covariates.live_births_by_sex], [config.input_data.location_id])


def get_dtp3_coverage(override_config=None):
    config = get_input_config(override_config)
    return core.get_covariate_estimates([covariates.dtp3_coverage_proportion], [config.input_data.location_id])


# TODO: These have no backing implementation in core.py yet.

def load_risk_correlation_matrices(override_config=None):
    config = get_input_config(override_config)
    return functions.load_risk_correlation_matrices(location_id=config.input_data.location_id,
                                                    gbd_round_id=config.input_data.gbd_round_id)


def get_ors_pafs(override_config=None):
    config = get_input_config(override_config)
    return functions.get_ors_pafs(location_id=config.input_data.location_id,
                                  gbd_round_id=config.input_data.gbd_round_id,
                                  draw_number=config.run_configuration.input_draw_number)


def get_ors_relative_risks(override_config=None):
    config = get_input_config(override_config)
    return functions.get_ors_relative_risks(gbd_round_id=config.input_data.gbd_round_id,
                                            draw_number=config.run_configuration.input_draw_number)

def get_rota_vaccine_protection(override_config=None):
    config = get_input_config(override_config)
    return functions.get_rota_vaccine_protection(location_id=config.input_data.location_id,
                                                 gbd_round_id=config.input_data.gbd_round_id,
                                                 draw_number=config.run_configuration.input_draw_number)
