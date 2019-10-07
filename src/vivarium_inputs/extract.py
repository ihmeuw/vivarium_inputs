import pandas as pd
from urllib.parse import urlencode

from typing import Union

from gbd_mapping import ModelableEntity
from vivarium_inputs.service_utilities import (map_kind_to_idtype, build_url,
                                               dataframe_from_response, make_request, check_response)
import vivarium_inputs.validation.raw as validation
from vivarium_inputs.globals import (METRICS, MEASURES,
                                     DataAbnormalError, DataDoesNotExistError,
                                     EmptyDataFrameException, NoBestVersionError, InputsException, OTHER_MEID)
from vivarium_inputs.utilities import filter_to_most_detailed_causes


ENDPOINT_DRAWS = 'draws'
ENDPOINT_SUMMARY = 'summary'
ENDPOINT_METADATA = 'metadata'


def extract_data(entity: ModelableEntity, measure: str, location_id: int,
                 validate: bool = True)-> Union[pd.Series, pd.DataFrame]:
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
        'incidence': (extract_incidence, {}),
        'prevalence': (extract_prevalence, {}),
        'birth_prevalence': (extract_birth_prevalence, {}),
        'disability_weight': (extract_disability_weight, {}),
        'remission': (extract_remission, {}),
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
        'cost': (extract_cost, {}),
        'utilization': (extract_utilization, {}),
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


def get_age_bins() -> pd.DataFrame:
    service_endpoint = 'age_bins'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": None,
                               "kind": None,
                               "name": None,
                               "source": None,
                               "location_id": None,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def get_age_group_id() -> pd.DataFrame:
    service_endpoint = 'estimation_age_group_ids'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": None,
                               "kind": None,
                               "name": None,
                               "source": None,
                               "location_id": None,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_birth_prevalence(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    kind = map_kind_to_idtype(entity)
    service_endpoint = 'birth_prevalence'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_deaths(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    kind = map_kind_to_idtype(entity)
    service_endpoint = 'deaths'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def get_estimation_years() -> pd.DataFrame:
    service_endpoint = 'estimation_year_ids'
    url = build_url(ENDPOINT_METADATA, service_endpoint,
                    urlencode({"gbd_id": None,
                               "kind": None,
                               "name": None,
                               "source": None,
                               "location_id": None,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_incidence(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    kind = map_kind_to_idtype(entity)
    service_endpoint = 'incidence_rate'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_prevalence(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    kind = map_kind_to_idtype(entity)
    service_endpoint = 'prevalence'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_remission(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    kind = map_kind_to_idtype(entity)
    service_endpoint = 'remission_rate'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)

# TODO - what kind of entity is this?
#def extract_disability_weight(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
def extract_disability_weight(entity, location_id: int) -> pd.DataFrame:
    # disability_weights = gbd.get_auxiliary_data('disability_weight', entity.kind, 'all', location_id)
    disability_weights = get_exposure(entity.gbd_id, location_id)
    data = disability_weights.loc[disability_weights.healthstate_id == entity.healthstate.gbd_id, :]
    return data


def extract_exposure(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    if entity.kind == 'risk_factor':
        data = get_exposure(entity, location_id)
        allowable_measures = [MEASURES['Proportion'], MEASURES['Continuous'], MEASURES['Prevalence']]
        proper_measure_id = set(data.measure_id).intersection(allowable_measures)
        if len(proper_measure_id) != 1:
            raise DataAbnormalError(f'Exposure data have {len(proper_measure_id)} measure id(s). Data should have'
                                    f'exactly one id out of {allowable_measures} but came back with'
                                    f'{proper_measure_id}.')
        else:
            data = data[data.measure_id == proper_measure_id.pop()]

    else:  # alternative_risk_factor or coverage_gap
        # ks data = gbd.get_auxiliary_data('exposure', entity.kind, entity.name, location_id)
        data = get_exposure(entity.gbd_id, location_id)
    return data


def get_exposure(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    kind = map_kind_to_idtype(entity)
    service_endpoint = 'exposure'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def get_exposure_standard_deviation(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    kind = map_kind_to_idtype(entity)
    service_endpoint = 'exposure_standard_deviation'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_exposure_standard_deviation(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    if entity.kind == 'risk_factor' and entity.name in OTHER_MEID:
        data = get_exposure(OTHER_MEID[entity.name], location_id)
    elif entity.kind == 'risk_factor':
        data = get_exposure_standard_deviation(entity.gbd_id, location_id)
    else:  # alternative_risk_factor
        # data = gbd.get_auxiliary_data('exposure_standard_deviation', entity.kind, entity.name, location_id)
        data = get_exposure(entity, location_id)
    return data


def get_modelable_entity_draws(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    kind = map_kind_to_idtype(entity)
    service_endpoint = 'risk_factor'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_exposure_distribution_weights(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    # data = gbd.get_auxiliary_data('exposure_distribution_weights', entity.kind, entity.name, location_id)
    data = get_exposure(entity, location_id)
    return data


def extract_relative_risk(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    if entity.kind == 'risk_factor':
        data = get_relative_risk(entity.gbd_id, location_id)
        data = filter_to_most_detailed_causes(data)
    else:  # coverage_gap
        # data = gbd.get_auxiliary_data('relative_risk', entity.kind, entity.name, location_id)
        data = get_exposure(entity, location_id)
    return data


def get_relative_risk(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    kind = map_kind_to_idtype(entity)
    service_endpoint = 'relative_risk'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_population_attributable_fraction(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    data = get_population_attributable_fraction(entity.gbd_id, location_id)
    data = data[data.metric_id == METRICS['Percent']]
    data = data[data.measure_id.isin([MEASURES['YLDs'], MEASURES['YLLs']])]
    data = filter_to_most_detailed_causes(data)
    return data


def get_population_attributable_fraction(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    kind = map_kind_to_idtype(entity)
    service_endpoint = 'population_attributable_fraction'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_mediation_factors(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    # data = gbd.get_auxiliary_data('mediation_factor', entity.kind, entity.name, location_id)
    data = get_exposure(entity, location_id)
    return data


def extract_estimate(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    data = get_covariate_estimate(entity.gbd_id, location_id)
    return data


def get_covariate_estimate(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    kind = map_kind_to_idtype(entity)
    service_endpoint = 'covariate'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_cost(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    # data = gbd.get_auxiliary_data('cost', entity.kind, entity.name, location_id)
    data = get_exposure(entity, location_id)
    return data


def extract_utilization(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    data = get_modelable_entity_draws(entity.gbd_id, location_id)
    return data


def extract_structure(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    data = get_population(entity, location_id)
    return data


def get_population(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    kind = map_kind_to_idtype(entity)
    service_endpoint = 'population'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_theoretical_minimum_risk_life_expectancy(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    data = get_theoretical_minimum_risk_life_expectancy(entity, location_id)
    return data


def get_theoretical_minimum_risk_life_expectancy(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    kind = map_kind_to_idtype(entity)
    service_endpoint = 'theoretical_minimum_risk_life_expectancy'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)
