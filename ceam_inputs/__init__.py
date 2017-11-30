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
    return _clean_and_filter_data(data, config.run_configuration.input_draw_number, 'prevalence')



def get_incidence(entity, override_config: ConfigTree=None):
    config = get_input_config(override_config)
    data = core.get_incidences([entity.gbd_id], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.run_configuration.input_draw_number, 'rate')


def get_remission(cause, override_config=None):
    config = get_input_config(override_config)
    data = core.get_remissions([cause], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.run_configuration.input_draw_number, 'rate')


def get_cause_specific_mortality(cause, override_config=None):
    config = get_input_config(override_config)
    data = core.get_cause_specific_mortalities([cause], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.run_configuration.input_draw_number, 'rate')


def get_excess_mortality(cause, override_config: ConfigTree=None):
    config = get_input_config(override_config)
    data = core.get_excess_mortalities([cause], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.run_configuration.input_draw_number, 'rate')


def get_disability_weight(sequela, override_config=None):
    config = get_input_config(override_config)
    data = core.get_disability_weights([sequela], [config.input_data.location_id])
    return float(data[f'draw_{config.run_configuration.input_draw_number}'])


####################################
# Measures for risk like entities  #
####################################


def get_relative_risks(entity, cause, override_config=None):
    config = get_input_config(override_config)
    data = core.get_relative_risks([entity], [config.input_data.location_id])
    data = data[data['cause_id'] == cause.gbd_id]
    return _clean_and_filter_data(data, config.run_configuration.input_draw_number, 'relative_risk')


def get_exposure_mean(risk, override_config=None):
    config = get_input_config(override_config)
    data = core.get_exposure_means([risk], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.run_configuration.input_draw_number, 'exposure_mean')


def get_exposure_standard_deviation(risk, override_config=None):
    config = get_input_config(override_config)
    data = core.get_exposure_standard_deviations([risk], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.run_configuration.input_draw_number, 'exposure_standard_deviation')



def get_population_attributable_fraction(entity, cause, override_config=None):
    config = get_input_config(override_config)
    data = core.get_population_attributable_fractions([entity], [config.input_data.location_id])
    data = data[data['cause_id'] == cause.gbd_id]
    return _clean_and_filter_data(data, config.run_configuration.input_draw_number, 'population_attributable_fraction')


def get_ensemble_weights(risk, override_config=None):
    config = get_input_config(override_config)
    return core.get_ensemble_weights([risk], [config.input_data.location_id])


def get_mediation_factors(risk, cause, override_config=None):
    config = get_input_config(override_config)
    data = core.get_mediation_factors([risk], [config.input_data.location_id])
    return data[data['cause_id'] == cause.gbd_id]


def get_risk_correlation_matrices(override_config=None):
    config = get_input_config(override_config)
    return core.get_risk_correlation_matrices([config.input_data.location_id])

#######################
# Other kinds of data #
#######################


def get_populations(override_config=None):
    config = get_input_config(override_config)
    data = core.get_populations([config.input_data.location_id])
    data = aux.get_age_group_midpoint_from_age_group_id(data)
    data = aux.normalize_for_simulation(data)
    return data


def get_age_bins():
    return core.get_age_bins()


def get_theoretical_minimum_risk_life_expectancy():
    return core.get_theoretical_minimum_risk_life_expectancy()


def get_subregions(override_config=None):
    config = get_input_config(override_config)
    return core.get_subregions([config.input_data.location_id])


def get_outpatient_visit_costs(override_config=None):
    config = get_input_config(override_config)
    data = core.get_costs([healthcare_entities.outpatient_visits], [config.input_data.location_id])
    data = data[['year_id', f'draw_{config.run_configuration.input_draw_number}']]
    return data.rename(columns={'year_id':'year', f'draw_{config.run_configuration.input_draw_number}':'op_cost'})


def get_inpatient_visit_costs(override_config=None):
    config = get_input_config(override_config)
    data = core.get_costs([healthcare_entities.inpatient_visits], [config.input_data.location_id])
    data = data[['year_id', f'draw_{config.run_configuration.input_draw_number}']]
    return data.rename(columns={'year_id':'year', f'draw_{config.run_configuration.input_draw_number}':'ip_cost'})


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


def get_protection(treatment_technology, override_config=None):
    config = get_input_config(override_config)
    data = core.get_protection([treatment_technology], [config.input_data.location_id])
    data = data[['location_id', 'measure', 'treatment_technology', f'draw_{config.run_configuration.input_draw_number}']]
    return data.rename(columns={f'draw_{config.run_configuration.input_draw_number}': 'protection'})


def get_healthcare_annual_visits(healthcare_entity, override_config=None):
    config = get_input_config(override_config)
    data = core.get_healthcare_annual_visits([healthcare_entity], [config.input_data.location_id])
    return _clean_and_filter_data(data, config.run_configuration.input_draw_number, 'annual_visits')
