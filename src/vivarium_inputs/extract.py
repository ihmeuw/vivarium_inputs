import pandas as pd

from get_draws.api import EmptyDataFrameException, InputsException
from gbd_artifacts.exceptions import NoBestVersionError

from .globals import gbd, METRICS, MEASURES, DataAbnormalError, DataNotExistError
import vivarium_inputs.validation.raw as validation


def extract_data(entity, measure: str, location_id: int) -> pd.DataFrame:
    extractors = {
        # Cause-like measures
        'incidence': extract_incidence,
        'prevalence': extract_prevalence,
        'birth_prevalence': extract_birth_prevalence,
        'disability_weight': extract_disability_weight,
        'remission': extract_remission,
        'deaths': extract_deaths,
        # Risk-like measures
        'exposure': extract_exposure,
        'exposure_standard_deviation': extract_exposure_standard_deviation,
        'exposure_distribution_weights': extract_exposure_distribution_weights,
        'relative_risk': extract_relative_risk,
        'population_attributable_fraction': extract_population_attributable_fraction,
        'mediation_factors': extract_mediation_factors,
        # Covariate measures
        'estimate': extract_estimate,
        # Health system measures
        'cost': extract_cost,
        'utilization': extract_utilization,
        # Population measures
        'structure': extract_structure,
        'theoretical_minimum_risk_life_expectancy': extract_theoretical_minimum_risk_life_expectancy,
    }

    validation.check_metadata(entity, measure)

    try:
        data = extractors[measure](entity, location_id)
    except (ValueError, AssertionError, EmptyDataFrameException, NoBestVersionError, InputsException) as e:
        if isinstance(e, ValueError) and f'Metadata associated with rei_id = {entity.gbd_id}' not in e.args:
            raise e
        elif isinstance(e, AssertionError) and f'Invalid covariate_id {entity.gbd_id}' not in e.args:
            raise e
        elif isinstance(e, InputsException) and measure != 'birth_prevalence':
            raise e
        else:
            raise DataNotExistError(f'{measure.capitalize()} data for {entity.name} does not exist.')

    validation.validate_raw_data(data, entity, measure, location_id)
    return data


def extract_prevalence(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_incidence_prevalence(entity_id=entity.gbd_id, location_id=location_id, entity_type=entity.kind)
    data = data[data.measure_id == MEASURES['Prevalence']]
    return data


def extract_incidence(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_incidence_prevalence(entity_id=entity.gbd_id, location_id=location_id, entity_type=entity.kind)
    data = data[data.measure_id == MEASURES['Incidence']]
    return data


def extract_birth_prevalence(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_birth_prevalence(entity_id=entity.gbd_id, location_id=location_id, entity_type=entity.kind)
    data = data[data.measure_id == MEASURES['Incidence']]
    return data


def extract_remission(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_modelable_entity_draws(entity.dismod_id, location_id)
    data = data[data.measure_id == MEASURES['Remission']]
    return data


def extract_disability_weight(entity, location_id: int) -> pd.DataFrame:
    disability_weights = gbd.get_auxiliary_data('disability_weight', entity.kind, 'all', location_id)
    data = disability_weights.loc[disability_weights.healthstate_id == entity.healthstate.gbd_id, :]
    return data


def extract_deaths(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_codcorrect_draws(entity.gbd_id, location_id)
    data = data[data.measure_id == MEASURES['Deaths']]
    return data


def extract_exposure(entity, location_id: int) -> pd.DataFrame:
    if entity.kind == 'risk_factor':
        data = gbd.get_exposure(entity.gbd_id, location_id)
        allowable_measures = [MEASURES['Proportion'], MEASURES['Continuous'], MEASURES['Prevalence']]
        proper_measure_id = set(data.measure_id).intersection(allowable_measures)
        if len(proper_measure_id) != 1:
            raise DataAbnormalError(f'Exposure data have {len(proper_measure_id)} measure id(s). Data should have'
                                    f'exactly one id out of {allowable_measures} but came back with {proper_measure_id}.')
        else:
            data = data[data.measure_id == proper_measure_id.pop()]

    else:  # alternative_risk_factor or coverage_gap
        data = gbd.get_auxiliary_data('exposure', entity.kind, entity.name, location_id)
    return data


def extract_exposure_standard_deviation(entity, location_id: int) -> pd.DataFrame:
    if entity.kind == 'risk_factor':
        data = gbd.get_exposure_standard_deviation(entity.gbd_id, location_id)
    else:  # alternative_risk_factor
        data = gbd.get_auxiliary_data('exposure_standard_deviation', entity.kind, entity.name, location_id)
    return data


def extract_exposure_distribution_weights(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_auxiliary_data('exposure_distribution_weights', entity.kind, entity.name, location_id)
    return data


def extract_relative_risk(entity, location_id: int) -> pd.DataFrame:
    if entity.kind == 'risk_factor':
        data = gbd.get_relative_risk(entity.gbd_id, location_id)
    else:  # coverage_gap
        data = gbd.get_auxiliary_data('relative_risk', entity.kind, entity.name, location_id)
    return data


def extract_population_attributable_fraction(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_paf(entity.gbd_id, location_id)
    data = data[data.measure_id == MEASURES['YLDs']]
    data = data[data.metric_id == METRICS['Percent']]
    return data


def extract_mediation_factors(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_auxiliary_data('mediation_factor', entity.kind, entity.name, location_id)
    return data


def extract_estimate(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_covariate_estimate(entity.gbd_id, location_id)
    return data


def extract_cost(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_auxiliary_data('cost', entity.kind, entity.name, location_id)
    return data


def extract_utilization(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_modelable_entity_draws(entity.gbd_id, location_id)
    return data


def extract_structure(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_population(location_id)
    return data


def extract_theoretical_minimum_risk_life_expectancy(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_theoretical_minimum_risk_life_expectancy()
    return data
