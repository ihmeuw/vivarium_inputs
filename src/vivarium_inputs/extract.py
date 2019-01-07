import pandas as pd

from .globals import gbd, METRICS, MEASURES
import vivarium_inputs.validation.raw as validation


def extract_data(entity, measure: str, location_id: str):
    extractors = {
        # Cause-like measures
        'incidence': extract_incidence,
        'prevalence': extract_prevalence,
        'birth_prevalence': extract_birth_prevalence,
        'disability_weight': extract_disability_weight,
        'remission': extract_remission,
        'cause_specific_mortality': extract_death,
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
    data = extractors[measure](entity, location_id)
    validation.validate_raw_data(data, entity, measure, location_id)
    return data


def extract_prevalence(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_como_draws(entity_id=entity.gbd_id, location_id=location_id, entity_type=entity.kind)
    data = data[data.measure_id == MEASURES['Prevalence']]
    return data


def extract_incidence(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_como_draws(entity_id=entity.gbd_id, location_id=location_id, entity_type=entity.kind)
    data = data[data.measure_id == MEASURES['Incidence']]
    return data


def extract_birth_prevalence(entity, location_id: int) -> pd.DataFrame:
    raise NotImplementedError()


def extract_remission(entity, location_id: int) -> pd.DataFrame:
    raise NotImplementedError()


def extract_disability_weight(entity, location_id: int) -> pd.DataFrame:
    disability_weights = gbd.get_auxiliary_data('disability_weight', entity.kind, 'all')
    data = disability_weights.loc[disability_weights.healthstate_id == entity.healthstate.gbd_id, :]
    return data


def extract_death(entity, location_id: int) -> pd.DataFrame:
    raise NotImplementedError()


def extract_exposure(entity, location_id: int) -> pd.DataFrame:
    raise NotImplementedError()


def extract_exposure_standard_deviation(entity, location_id: int) -> pd.DataFrame:
    raise NotImplementedError()


def extract_exposure_distribution_weights(entity, location_id: int) -> pd.DataFrame:
    raise NotImplementedError()


def extract_relative_risk(entity, location_id: int) -> pd.DataFrame:
    raise NotImplementedError()


def extract_population_attributable_fraction(entity, location_id: int) -> pd.DataFrame:
    if entity.kind == 'risk_factor':
        raise NotImplementedError()
    elif entity.kind == 'coverage_gap':
        raise NotImplementedError()
    elif entity.kind == 'etiology':
        data = gbd.get_paf(entity_id=entity.gbd_id, location_id=location_id)
        data = data[data.measure_id == MEASURES['YLDs']]
        data = data[data.metric_id == METRICS['Percent']]
        return data


def extract_mediation_factors(entity, location_id: int) -> pd.DataFrame:
    raise NotImplementedError()


def extract_estimate(entity, location_id: int) -> pd.DataFrame:
    raise NotImplementedError()


def extract_cost(entity, location_id: int) -> pd.DataFrame:
    raise NotImplementedError()


def extract_utilization(entity, location_id: int) -> pd.DataFrame:
    raise NotImplementedError()


def extract_structure(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_population(location_id)
    return data


def extract_theoretical_minimum_risk_life_expectancy(entity, location_id: int) -> pd.DataFrame:
    data = gbd.get_theoretical_minimum_risk_life_expectancy()
    return data

