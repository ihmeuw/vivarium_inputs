from typing import Union
from loguru import logger

import pandas as pd

from gbd_mapping import Cause, RiskFactor, Sequela, Covariate, Etiology
from vivarium_inputs.globals import Population


from vivarium_inputs.globals import (gbd, METRICS, MEASURES,
                                     DataAbnormalError, DataDoesNotExistError,
                                     EmptyDataFrameException, NoBestVersionError, InputsException, OTHER_MEID,
                                     RISKS_WITH_NEGATIVE_PAF)
from vivarium_inputs.utilities import filter_to_most_detailed_causes
from vivarium_inputs.mapping_extension import AlternativeRiskFactor, HealthcareEntity
import vivarium_inputs.validation.raw as validation



def extract_data(entity, measure: str, location_id: int, validate: bool = True) -> Union[pd.Series, pd.DataFrame]:
    """Check metadata for the requested entity-measure pair. Pull raw data from
    GBD. The only filtering that occurs is by applicable measure id, metric id,
    or to most detailed causes where relevant. If validate is turned on, will
    also pull any additional data needed for raw validation and call raw
    validation on the extracted data.

    Parameters
    ----------
    entity
        Entity for which to extract data.
    measure
        Measure for which to extract data.
    location_id
        Location for which to extract data.
    validate
        Flag indicating whether additional data needed for raw validation
        should be extracted and whether raw validation should be performed.
        Should only be set to False if data is being extracted for
        investigation. Never extract data for a simulation without validation.

    Returns
    -------
    Data for the entity-measure pair and specific location requested.

    Raises
    ------
    DataDoesNotExistError
        If data for the entity-measure-location set requested does not exist.

    """
    extractors = {
        # Cause-like measures
        'incidence_rate': (extract_incidence_rate, {}),
        'prevalence': (extract_prevalence, {}),
        'birth_prevalence': (extract_birth_prevalence, {}),
        'disability_weight': (extract_disability_weight, {}),
        'remission_rate': (extract_remission_rate, {}),
        'deaths': (extract_deaths, {'population': extract_structure}),
        # Risk-like measures
        'exposure': (extract_exposure, {}),
        'exposure_standard_deviation': (extract_exposure_standard_deviation, {'exposure': extract_exposure}),
        'exposure_distribution_weights': (extract_exposure_distribution_weights, {}),
        'relative_risk': (extract_relative_risk, {'exposure': extract_exposure}),
        'population_attributable_fraction': (extract_population_attributable_fraction,
                                             {'exposure': extract_exposure, 'relative_risk': extract_relative_risk}),
        'etiology_population_attributable_fraction': (extract_population_attributable_fraction, {}),
        'mediation_factors': (extract_mediation_factors, {}),
        # Covariate measures
        'estimate': (extract_estimate, {}),
        # Health system measures
        'utilization_rate': (extract_utilization_rate, {}),
        # Population measures
        'structure': (extract_structure, {}),
        'theoretical_minimum_risk_life_expectancy': (extract_theoretical_minimum_risk_life_expectancy, {}),
    }

    validation.check_metadata(entity, measure)

    try:
        main_extractor, additional_extractors = extractors[measure]
        data = main_extractor(entity, location_id)
    except (ValueError, AssertionError, EmptyDataFrameException, NoBestVersionError, InputsException) as e:
        if isinstance(e, ValueError) and f'Metadata associated with rei_id = {entity.gbd_id}' not in str(e):
            raise e
        elif (isinstance(e, AssertionError) and f'Invalid covariate_id {entity.gbd_id}' not in str(e)
                and 'No best model found' not in str(e)):
            raise e
        elif isinstance(e, InputsException) and measure != 'birth_prevalence':
            raise e
        else:
            raise DataDoesNotExistError(f'{measure.capitalize()} data for {entity.name} does not exist.')

    if validate:
        additional_data = {name: extractor(entity, location_id) for name, extractor in additional_extractors.items()}
        validation.validate_raw_data(data, entity, measure, location_id, **additional_data)

    return data


def extract_prevalence(entity: Union[Cause, Sequela], location_id: int) -> pd.DataFrame:
    data = gbd.get_incidence_prevalence(entity_id=entity.gbd_id, location_id=location_id, entity_type=entity.kind)
    data = data[data.measure_id == MEASURES['Prevalence']]
    return data


def extract_incidence_rate(entity: Union[Cause, Sequela], location_id: int) -> pd.DataFrame:
    data = gbd.get_incidence_prevalence(entity_id=entity.gbd_id, location_id=location_id, entity_type=entity.kind)
    data = data[data.measure_id == MEASURES['Incidence rate']]
    return data


def extract_birth_prevalence(entity: Union[Cause, Sequela], location_id: int) -> pd.DataFrame:
    data = gbd.get_birth_prevalence(entity_id=entity.gbd_id, location_id=location_id, entity_type=entity.kind)
    data = data[data.measure_id == MEASURES['Incidence rate']]
    return data


def extract_remission_rate(entity: Cause, location_id: int) -> pd.DataFrame:
    data = gbd.get_modelable_entity_draws(entity.me_id, location_id)
    data = data[data.measure_id == MEASURES['Remission rate']]
    return data


def extract_disability_weight(entity: Sequela, location_id: int) -> pd.DataFrame:
    disability_weights = gbd.get_auxiliary_data('disability_weight', entity.kind, 'all', location_id)
    data = disability_weights.loc[disability_weights.healthstate_id == entity.healthstate.gbd_id, :]
    return data


def extract_deaths(entity: Cause, location_id: int) -> pd.DataFrame:
    data = gbd.get_codcorrect_draws(entity.gbd_id, location_id)
    data = data[data.measure_id == MEASURES['Deaths']]
    return data


def extract_exposure(entity: Union[RiskFactor, AlternativeRiskFactor], location_id: int) -> pd.DataFrame:
    if entity.kind == 'risk_factor':
        data = gbd.get_exposure(entity.gbd_id, location_id)
        allowable_measures = [MEASURES['Proportion'], MEASURES['Continuous'], MEASURES['Prevalence']]
        proper_measure_id = set(data.measure_id).intersection(allowable_measures)
        if len(proper_measure_id) != 1:
            raise DataAbnormalError(f'Exposure data have {len(proper_measure_id)} measure id(s). Data should have'
                                    f'exactly one id out of {allowable_measures} but came back with {proper_measure_id}.')
        else:
            data = data[data.measure_id == proper_measure_id.pop()]

    else:  # alternative_risk_factor
        data = gbd.get_auxiliary_data('exposure', entity.kind, entity.name, location_id)

    return data


def extract_exposure_standard_deviation(entity: Union[RiskFactor, AlternativeRiskFactor], location_id: int) -> pd.DataFrame:
    if entity.kind == 'risk_factor' and entity.name in OTHER_MEID:
        data = gbd.get_modelable_entity_draws(OTHER_MEID[entity.name], location_id)
    elif entity.kind == 'risk_factor':
        data = gbd.get_exposure_standard_deviation(entity.gbd_id, location_id)
    else:  # alternative_risk_factor
        data = gbd.get_auxiliary_data('exposure_standard_deviation', entity.kind, entity.name, location_id)
    return data


def extract_exposure_distribution_weights(entity: Union[RiskFactor, AlternativeRiskFactor], location_id: int) -> pd.DataFrame:
    data = gbd.get_auxiliary_data('exposure_distribution_weights', entity.kind, entity.name, location_id)
    return data


def extract_relative_risk(entity: RiskFactor, location_id: int) -> pd.DataFrame:
    data = gbd.get_relative_risk(entity.gbd_id, location_id)
    data = filter_to_most_detailed_causes(data)
    return data


def extract_population_attributable_fraction(entity: Union[RiskFactor, Etiology], location_id: int) -> pd.DataFrame:
    data = gbd.get_paf(entity.gbd_id, location_id)
    data = data[data.metric_id == METRICS['Percent']]
    data = data[data.measure_id.isin([MEASURES['YLDs'], MEASURES['YLLs']])]
    data = filter_to_most_detailed_causes(data)
    return data


def extract_mediation_factors(entity: RiskFactor, location_id: int) -> pd.DataFrame:
    data = gbd.get_auxiliary_data('mediation_factor', entity.kind, entity.name, location_id)
    return data


def extract_estimate(entity: Covariate, location_id: int) -> pd.DataFrame:
    data = gbd.get_covariate_estimate(entity.gbd_id, location_id)
    return data


def extract_utilization_rate(entity: HealthcareEntity, location_id: int) -> pd.DataFrame:
    data = gbd.get_modelable_entity_draws(entity.gbd_id, location_id)
    return data


def extract_structure(entity: Population, location_id: int) -> pd.DataFrame:
    data = gbd.get_population(location_id)
    return data


def extract_theoretical_minimum_risk_life_expectancy(entity: Population, location_id: int) -> pd.DataFrame:
    data = gbd.get_theoretical_minimum_risk_life_expectancy()
    return data
