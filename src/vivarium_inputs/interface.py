from vivarium.config_tree import ConfigTree  # Just for typing info.
from gbd_mapping import covariates

try:
    from vivarium_gbd_access.utilities import get_input_config
except ModuleNotFoundError:
    from vivarium.framework.configuration import build_simulation_configuration

    def get_input_config(override_config):
        config = build_simulation_configuration()
        config.update(override_config)
        config.update({
            'input_data': {
                'location': 'Kenya',
                'input_draw_number': 0,
            }
        })

        return config


from vivarium_inputs import core
from vivarium_inputs.utilities import (select_draw_data, get_age_group_midpoint_from_age_group_id,
                                       normalize_for_simulation)
from vivarium_inputs.mapping_extension import healthcare_entities

__all__ = [
    "get_prevalence", "get_incidence", "get_remission", "get_cause_specific_mortality", "get_excess_mortality",
    "get_disability_weight", "get_relative_risk", "get_exposure", "get_exposure_standard_deviation",
    "get_population_attributable_fraction", "get_ensemble_weights", "get_populations", "get_age_bins",
    "get_theoretical_minimum_risk_life_expectancy", "get_subregions", "get_outpatient_visit_costs",
    "get_inpatient_visit_costs", "get_inpatient_visit_costs",
    "get_age_specific_fertility_rates", "get_live_births_by_sex", "get_dtp3_coverage",
    "get_protection", "get_healthcare_annual_visits",
]


def _clean_and_filter_data(data, draw_number, column_name):
    key_cols = [c for c in data.columns if 'draw' not in c]
    data = data[key_cols + [f'draw_{draw_number}']]
    data = get_age_group_midpoint_from_age_group_id(data)
    return select_draw_data(data, draw_number, column_name)


####################################
# Measures for cause like entities #
####################################

def get_prevalence(entity, override_config=None):
    config = get_input_config(override_config)
    data = core.get_draws([entity], ['prevalence'], [config.input_data.location]).drop('measure', 'columns')
    return _clean_and_filter_data(data, config.input_data.input_draw_number, 'prevalence')


def get_incidence(entity, override_config: ConfigTree=None):
    config = get_input_config(override_config)
    data = core.get_draws([entity], ['incidence'], [config.input_data.location]).drop('measure', 'columns')
    return _clean_and_filter_data(data, config.input_data.input_draw_number, 'rate')


def get_remission(cause, override_config=None):
    config = get_input_config(override_config)
    data = core.get_draws([cause], ['remission'], [config.input_data.location]).drop('measure', 'columns')
    return _clean_and_filter_data(data, config.input_data.input_draw_number, 'rate')


def get_cause_specific_mortality(cause, override_config=None):
    config = get_input_config(override_config)
    data = core.get_draws([cause], ['cause_specific_mortality'],
                          [config.input_data.location]).drop('measure', 'columns')
    return _clean_and_filter_data(data, config.input_data.input_draw_number, 'rate')


def get_excess_mortality(cause, override_config: ConfigTree=None):
    config = get_input_config(override_config)
    data = core.get_draws([cause], ['excess_mortality'], [config.input_data.location]).drop('measure', 'columns')
    return _clean_and_filter_data(data, config.input_data.input_draw_number, 'rate')


def get_disability_weight(sequela, override_config=None):
    config = get_input_config(override_config)
    data = core.get_draws([sequela], ['disability_weight'], [config.input_data.location]).drop('measure', 'columns')
    return float(data[f'draw_{config.input_data.input_draw_number}'])


####################################
# Measures for risk like entities  #
####################################


def get_relative_risk(entity, cause, override_config=None):
    config = get_input_config(override_config)
    data = core.get_draws([entity], ['relative_risk'], [config.input_data.location]).drop('measure', 'columns')
    data = data[data['cause_id'] == cause.gbd_id]
    return _clean_and_filter_data(data, config.input_data.input_draw_number, 'relative_risk')


def get_exposure(risk, override_config=None):
    config = get_input_config(override_config)
    data = core.get_draws([risk], ['exposure'], [config.input_data.location]).drop('measure', 'columns')
    data = _clean_and_filter_data(data, config.input_data.input_draw_number, 'mean')
    # FIXME: This is here because FPG puts zeros in its unmodelled age groups unlike most other gbd risks
    data = data[data['mean'] != 0]
    return data


def get_exposure_standard_deviation(risk, override_config=None):
    config = get_input_config(override_config)
    data = core.get_draws([risk], ['exposure_standard_deviation'],
                          [config.input_data.location]).drop('measure', 'columns')
    data = _clean_and_filter_data(data, config.input_data.input_draw_number, 'standard_deviation')
    # FIXME: This is here because FPG puts zeros in its unmodelled age groups unlike most other gbd risks
    data = data[data['standard_deviation'] != 0]
    return data


def get_population_attributable_fraction(entity, risk, override_config=None):
    config = get_input_config(override_config)
    data = core.get_draws([entity], ['population_attributable_fraction'],
                          [config.input_data.location]).drop('measure', 'columns')
    data = data[data['risk_id'] == risk.gbd_id]
    return _clean_and_filter_data(data, config.input_data.input_draw_number, 'population_attributable_fraction')


def get_ensemble_weights(risk, override_config=None):
    return core.get_ensemble_weights([risk])


#######################
# Other kinds of data #
#######################


def get_populations(override_config=None, location=None):
    config = get_input_config(override_config)
    if location:
        data = core.get_populations([location])
    else:
        data = core.get_populations([config.input_data.location])
    data = get_age_group_midpoint_from_age_group_id(data)
    data = normalize_for_simulation(data)
    return data


def get_age_bins():
    return core.get_age_bins()


def get_theoretical_minimum_risk_life_expectancy():
    return core.get_theoretical_minimum_risk_life_expectancy()


def get_subregions(override_config=None):
    config = get_input_config(override_config)
    return core.get_subregions([config.input_data.location])


def get_outpatient_visit_costs(override_config=None):
    config = get_input_config(override_config)
    data = core.get_draws([healthcare_entities.outpatient_visits],
                          ['cost'], [config.input_data.location]).drop('measure', 'columns')
    data = data[['year_id', f'draw_{config.input_data.input_draw_number}']]
    return data.rename(columns={'year_id':'year', f'draw_{config.input_data.input_draw_number}': 'cost'})


def get_inpatient_visit_costs(override_config=None):
    config = get_input_config(override_config)
    data = core.get_draws([healthcare_entities.inpatient_visits],
                          ['cost'], [config.input_data.location]).drop('measure', 'columns')
    data = data[['year_id', f'draw_{config.input_data.input_draw_number}']]
    return data.rename(columns={'year_id':'year', f'draw_{config.input_data.input_draw_number}': 'cost'})


def get_age_specific_fertility_rates(override_config=None):
    config = get_input_config(override_config)
    data = core.get_covariate_estimates([covariates.age_specific_fertility_rate], [config.input_data.location])
    data = get_age_group_midpoint_from_age_group_id(data)
    data = normalize_for_simulation(data)
    return data.loc[data.sex == 'Female', ['age', 'year', 'mean_value']].rename(columns={'mean_value': 'rate'})


def get_live_births_by_sex(override_config=None):
    config = get_input_config(override_config)
    data = core.get_covariate_estimates([covariates.live_births_by_sex], [config.input_data.location])
    data = data[['sex_id', 'year_id', 'mean_value', 'lower_value', 'upper_value']]
    return normalize_for_simulation(data)


def get_dtp3_coverage(override_config=None):
    config = get_input_config(override_config)
    data = core.get_covariate_estimates([covariates.dtp3_coverage_proportion], [config.input_data.location])
    data = normalize_for_simulation(data)
    return data[['mean_value', 'lower_value', 'upper_value', 'year']]


def get_protection(treatment_technology, override_config=None):
    config = get_input_config(override_config)
    data = core.get_draws([treatment_technology],
                          ['protection'], [config.input_data.location]).drop('measure', 'columns')
    data = data[['location', 'treatment_technology', f'draw_{config.input_data.input_draw_number}']]
    return data.rename(columns={f'draw_{config.input_data.input_draw_number}': 'protection'})


def get_healthcare_annual_visits(healthcare_entity, override_config=None):
    config = get_input_config(override_config)
    data = core.get_draws([healthcare_entity],
                          ['annual_visits'], [config.input_data.location]).drop('measure', 'columns')
    return _clean_and_filter_data(data, config.input_data.input_draw_number, 'annual_visits')
