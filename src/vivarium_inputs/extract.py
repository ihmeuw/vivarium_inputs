import pandas as pd
from urllib.parse import urlencode
from typing import Union, List

import vivarium_inputs.validation.raw as validation

from gbd_mapping import ModelableEntity, RiskFactor
from vivarium_inputs.service_utilities import (build_url, dataframe_from_response, make_request, check_response)
from vivarium_inputs.globals import (DataDoesNotExistError, EmptyDataFrameException, NoBestVersionError,
                                     InputsException, OTHER_MEID, SEXES, NON_MAX_TMREL)


ENDPOINT_DRAWS = 'draws'
ENDPOINT_SUMMARY = 'summary'
ENDPOINT_METADATA = 'metadata'


def extract_data(entity: ModelableEntity, measure: str, location_id: int,
                 validate: bool = True) -> Union[pd.Series, pd.DataFrame]:
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
        'cost': (extract_cost, {}),
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


def extract_prevalence(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'prevalence'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": entity.kind,
                               "name": None,
                               "source": 'como',
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_incidence_rate(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'incidence_rate'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": entity.kind,
                               "name": None,
                               "source": 'como',
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_birth_prevalence(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'birth_prevalence'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": entity.kind,
                               "name": None,
                               "source": 'como',
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_remission_rate(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'remission_rate'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": entity.kind,
                               "name": None,
                               "source": 'epi',
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)

def extract_disability_weight(entity, location_id: int) -> pd.DataFrame:
    # coming soon...
    raise NotImplemented


def extract_deaths(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'deaths'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": entity.kind,
                               "name": None,
                               "source": "codcorrect",
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_exposure(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'exposure'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": entity.kind,
                               "name": None,
                               "source": 'exposure',
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_exposure_standard_deviation(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'exposure_standard_deviation'

    if entity.kind == 'risk_factor' and entity.name in OTHER_MEID:
        source = 'epi'
    elif entity.kind == 'risk_factor':
        source = 'exposure_sd'
    else:  # alternative_risk_factor
        source = None

    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": entity.kind,
                               "name": None,
                               "source": source,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_exposure_distribution_weights(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'exposure_weights'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": entity.kind,
                               "name": None,
                               "source": 'auxiliary_data',
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_relative_risk(entity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'relative_risk'
    source = 'rr' if 'relative_risk' == entity.kind else 'auxiliary_data'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": entity.kind,
                               "name": None,
                               "source": source,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_population_attributable_fraction(entity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'population_attributable_fraction'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": entity.kind,
                               "name": None,
                               "source": 'burdenator',
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_mediation_factors(entity, location_id: int) -> pd.DataFrame:
    # TODO - Not needed ???
    raise NotImplemented


def extract_cost(entity, location_id: int) -> pd.DataFrame:
    # TODO - ???
    raise NotImplemented


def extract_utilization_rate(entity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'healthcare_utilization'
    url = build_url(ENDPOINT_DRAWS, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": entity.kind,
                               "name": None,
                               "source": 'epi',
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_estimate(entity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'covariate'
    url = build_url(ENDPOINT_SUMMARY, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": entity.kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_structure(entity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'population'
    url = build_url(ENDPOINT_SUMMARY, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": entity.kind,
                               "name": None,
                               "source": None,
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_theoretical_minimum_risk_life_expectancy(entity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'theoretical_minimum_risk_life_expectancy'
    url = build_url(ENDPOINT_SUMMARY, service_endpoint,
                    urlencode({"gbd_id": entity.gbd_id,
                               "kind": entity.kind,
                               "name": None,
                               "source": 'tmrel',
                               "location_id": location_id,
                               "location": None}))
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def get_estimation_years(*_, **__) -> pd.Series:
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


def get_year_block(*_, **__) -> pd.DataFrame:
    estimation_years = get_estimation_years()
    year_block = pd.DataFrame({'year_start': range(min(estimation_years), max(estimation_years) + 1)})
    year_block['year_end'] = year_block['year_start'] + 1
    return year_block


def get_age_group_ids(*_, **__) -> List[int]:
    service_endpoint = 'estimation_age_group_ids'
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


def get_age_bins(*_, **__) -> pd.DataFrame:
    service_endpoint = 'age_bins'
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


def get_location_ids(*_, **__) -> pd.DataFrame:
    service_endpoint = 'location_ids'
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


def get_location_id(location_name):
    return {r.location_name: r.location_id for _, r in get_location_ids().iterrows()}[location_name]


def get_location_id_parents(location_id: int) -> List[int]:
    # TODO
    raise NotImplemented
    # location_metadata = gbd.get_location_path_to_global().set_index('location_id')
    # parent_ids = [int(loc) for loc in location_metadata.at[location_id, 'path_to_top_parent'].split(',')]
    # return parent_ids


def get_demographic_dimensions(location_id: int, draws: bool = False, value: float = None) -> pd.DataFrame:
    ages = get_age_group_ids()
    estimation_years = get_estimation_years()
    years = range(min(estimation_years), max(estimation_years) + 1)
    sexes = [SEXES['Male'], SEXES['Female']]
    location = [location_id]
    values = [location, sexes, ages, years]
    names = ['location_id', 'sex_id', 'age_group_id', 'year_id']

    data = (pd.MultiIndex
            .from_product(values, names=names)
            .to_frame(index=False))
    if draws:
        for i in range(1000):
            data[f'draw_{i}'] = value
    return data


def get_tmrel_category(entity: RiskFactor) -> str:
    if entity.name in NON_MAX_TMREL:
        tmrel_cat = NON_MAX_TMREL[entity.name]
    else:
        tmrel_cat = sorted(list(entity.categories.to_dict()), key=lambda x: int(x[3:]))[-1]
    return tmrel_cat
