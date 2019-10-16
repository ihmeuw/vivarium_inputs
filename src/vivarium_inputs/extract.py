import gzip
import io
from typing import Union, List
from urllib.parse import urlencode

import pandas as pd
import requests as req
from gbd_mapping import ModelableEntity
from loguru import logger

import vivarium_inputs.validation.raw as validation


TYPE_DRAWS = 'draws'
TYPE_SUMMARY = 'summary'
TYPE_METADATA = 'metadata'


SRC_AUXILIARY = 'auxiliary_data'
SRC_BURDENATOR = 'burdenator'
SRC_CODCORRECT = 'codcorrect'
SRC_COMO = 'como'
SRC_DB_QUERIES = 'db_queries'
SRC_EPI = 'epi'
SRC_EXPOSURE = 'exposure'
SRC_EXPOSURE_SD = 'exposure_sd'
SRC_RELATIVE_RISK = 'rr'
SRC_TMREL = 'tmrel'


TIMEOUT_SERVICE = None

SERVICE_VERSION = '1'
BASE_URL = f'http://microsim-rancher-p01.hosts.ihme.washington.edu:5000/v{SERVICE_VERSION}'


class GbdServiceError(Exception):
    """Error for failures accessing GBD data."""
    pass


def dataframe_from_response(resp: req.Response) -> pd.DataFrame:
    if resp.ok:
        df = pd.read_csv(io.BytesIO(gzip.decompress(resp.content)), index_col=0)
        return df
    else:
        logger.error(f'GbdServiceError: http response code {resp}')
        raise GbdServiceError


def build_url(endpoint_category: str, endpoint: str, entity: ModelableEntity, source: str,
              location_id: Union[int, None]) -> str:
    base_url = f'{BASE_URL}/{endpoint_category}/{endpoint}'
    url = f'{base_url}'
    if location_id:
        loc_name = location_name_from_id(location_id)
        params = urlencode({'gbd_id': entity.gbd_id,
                   'kind': entity.kind,
                   'name': entity.name,
                   'source': source,
                   'location_id': location_id,
                   'location': loc_name})
        url = f'{base_url}?{params}'
    return url


def make_request(url: str) -> req.Response:
    logger.info(f'Request start: {url}')
    resp = req.get(url, timeout=None)
    logger.info(f'Request end: {resp}')
    return resp


def check_response(resp: req.Response) -> None:
    if not resp.ok:
        logger.error(f'GbdServiceError: http response code {resp}')
        raise GbdServiceError



def extract_data(entity: ModelableEntity, measure: str, location_id: int,
                     validate: bool = True, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Check metadata for the requested entity-measure pair. Package inputs for
    call to REST API that in turn calls GBD. Any necessary filtering is done on
    the server side. If validate is turned on, will also pull any additional data
    needed for raw validation and call raw validation on the extracted data.

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
    kwargs
        'type', 'source', and 'id' keyword arguments can be used to override
        the default settings.

    Returns
    -------
    Data for the entity-measure pair and specific location requested.

    Raises
    ------
    GbdServiceError
        General error for unsuccessful call to service. Additional data
        helps to explain the cause of the error.

    """
    extractors = {
        # measure name
        'incidence_rate': {
            'kind_map': {
                'cause': (SRC_COMO, TYPE_DRAWS),
                'sequela': (SRC_COMO, TYPE_DRAWS)
            },
            'validation_data': {}
        },
        'prevalence': {
            'kind_map': {
                'cause': (SRC_COMO, TYPE_DRAWS),
                'sequela': (SRC_COMO, TYPE_DRAWS)
            },
            'validation_data': {}
        },
        'birth_prevalence': {
            'kind_map': {
                'cause': (SRC_COMO, TYPE_DRAWS),
                'sequela': (SRC_COMO, TYPE_DRAWS)
            },
            'validation_data': {}
        },
        'disability_weight': {
            'kind_map': {
                'cause': (SRC_AUXILIARY, TYPE_METADATA),   # TODO
                'sequela': (SRC_AUXILIARY, TYPE_METADATA),  # TODO
            },
            'validation_data': {}
        },
        'remission_rate': {
            'kind_map': {
                'cause': (SRC_EPI, TYPE_DRAWS),
            },
            'validation_data': {}
        },
        'deaths': {
            'kind_map': {
                'cause': (SRC_CODCORRECT, TYPE_DRAWS),
            },
            'validation_data': {'population': extract_population}
        },
        # Risk-like measures
        'exposure': {
            'kind_map': {
                'risk_factor': (SRC_EXPOSURE, TYPE_DRAWS),
                'coverage_gap': (SRC_AUXILIARY, TYPE_METADATA),
                'alternative_risk_factor': (SRC_AUXILIARY, TYPE_METADATA)
            },
            'validation_data': {}
        },
        'exposure_standard_deviation': {
            'kind_map': {
                'risk_factor': (SRC_EXPOSURE, TYPE_DRAWS),
                'alternative_risk_factor': (SRC_AUXILIARY, TYPE_METADATA)
            },
            'validation_data': {'exposure': extract_exposure}
        },
        'exposure_distribution_weights': {
            'kind_map': {
                'risk_factor': (SRC_AUXILIARY, TYPE_DRAWS),
                'alternative_risk_factor': (SRC_AUXILIARY, TYPE_METADATA)
            },
            'validation_data': {}
        },
        'relative_risk': {
            'kind_map': {
                'relative_risk': (SRC_RELATIVE_RISK, TYPE_DRAWS),
                'coverage_gap': (SRC_AUXILIARY, TYPE_METADATA),
            },
            'validation_data': {'exposure': extract_exposure}
        },
        'population_attributable_fraction': {
            'kind_map': {
                'risk_factor': (SRC_BURDENATOR, TYPE_DRAWS),    # TODO - check
                'etiology': (SRC_BURDENATOR, TYPE_DRAWS),
            },
            'validation_data':  {'exposure': extract_exposure, 'relative_risk': extract_relative_risk}
        },
        # this requires a change in core.py to not call this
        # 'etiology_population_attributable_fraction': {
        #     'kind_map': {
        #         'etiology': (SRC_BURDENATOR, TYPE_DRAWS),
        #     },
        #     'validation_data': {}
        # },
        # this raises NotImplemented in core.py
        # 'mediation_factors': {
        #     'kind_map': {
        #         'kind_1': (SRC_COMO, TYPE_DRAWS),
        #     },
        #     'validation_data': {}
        # },
        # Covariate measures
        'estimate': {
            'kind_map': {
                'covariate': (SRC_EPI, TYPE_SUMMARY),
            },
            'validation_data': {}
        },
        # TODO
        # 'cost': {
        #     'kind_map': {
        #         'healthcare_entity': (SRC_EPI, TYPE_SUMMARY),
        #         'health_technology': (SRC_EPI, TYPE_SUMMARY),
        #     },
        #     'validation_data': {}
        # },
        'healthcare_utilization': {
            'kind_map': {
                'healthcare_entity': (SRC_EPI, TYPE_SUMMARY),
            },
            'validation_data': {}
        },
        # was structure
        'population': {
            'kind_map': {
                'population': (SRC_EPI, TYPE_SUMMARY),
            },
            'validation_data': {}
        },
        'theoretical_minimum_risk_life_expectancy': {
            'kind_map': {
                'population': (SRC_EPI, TYPE_SUMMARY),  # TODO - check
            },
            'validation_data': {}
        }
    }

    validation.check_metadata(entity, measure)

    source_default, type_default  = extractors[measure]['kind_map'][entity.kind]
    type_final = kwargs.get('type', type_default)
    source = kwargs.get('source', source_default)
    entity.gbd_id = kwargs.get('id', entity.gbd_id)

    url = build_url(type_final, measure, entity, source, location_id)
    resp = make_request(url)
    check_response(resp)
    data = dataframe_from_response(resp)

    if validate and len(extractors[measure]['validation_data']):
        additional_extractors = extractors[measure]['validation_data']
        additional_data = {name: extractor(entity, location_id)
                           for name, extractor in additional_extractors.items()}
        validation.validate_raw_data(data, entity, measure, location_id, **additional_data)

    return data


def extract_exposure(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'exposure'
    url = build_url(TYPE_DRAWS, service_endpoint, entity, 'exposure', location_id)
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_population(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'population'
    url = build_url(TYPE_SUMMARY, service_endpoint, entity, SRC_EPI, location_id)
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_relative_risk(entity: ModelableEntity, location_id: int) -> pd.DataFrame:
    service_endpoint = 'relative_risk'
    type_final = TYPE_DRAWS if entity.kind == 'risk_factor' else TYPE_SUMMARY
    source = SRC_RELATIVE_RISK if entity.kind == 'risk_factor' else SRC_AUXILIARY
    url = build_url(type_final, service_endpoint, entity, source, location_id)
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_estimation_years(*_, **__) -> pd.Series:
    service_endpoint = 'estimation_year_ids'
    empty_entity = ModelableEntity('', '', None)
    url = build_url(TYPE_METADATA, service_endpoint, empty_entity, '', None)
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp).iloc[:,0]


def extract_age_group_ids(*_, **__) -> List[int]:
    service_endpoint = 'estimation_age_group_ids'
    empty_entity = ModelableEntity('', '', None)
    url = build_url(TYPE_METADATA, service_endpoint, empty_entity, '', None)
    resp = make_request(url)
    check_response(resp)
    return list(dataframe_from_response(resp).iloc[:,0].values)


def extract_age_bins(*_, **__) -> pd.DataFrame:
    service_endpoint = 'age_bins'
    empty_entity = ModelableEntity('', '', None)
    url = build_url(TYPE_METADATA, service_endpoint, empty_entity, '', None)
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def extract_locations_ids(*_, **__) -> pd.DataFrame:
    service_endpoint = 'location_ids'
    empty_entity = ModelableEntity('', '', None)
    url = build_url(TYPE_METADATA, service_endpoint, empty_entity, '', None)
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)


def location_name_from_id(location_id: int) -> str:
    df = extract_locations_ids()
    tmp = df[df.location_id==location_id].location_name.iloc[0]
    return tmp


def extract_location_path_to_global(*_, **__) -> pd.DataFrame:
    service_endpoint = 'location_path_to_global'
    empty_entity = ModelableEntity('', '', None)
    url = build_url(TYPE_METADATA, service_endpoint, empty_entity, '', None)
    resp = make_request(url)
    check_response(resp)
    return dataframe_from_response(resp)

