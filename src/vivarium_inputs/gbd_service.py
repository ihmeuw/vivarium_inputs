import gzip
from loguru import logger
import pandas as pd
from pathlib import Path
import requests as req
from typing import List, Callable, TypeVar
from urllib.parse import urlencode


SERVICE_VERSION = '1'
BASE_URL = f'http://microsim-rancher-p01.hosts.ihme.washington.edu:5000/v{SERVICE_VERSION}'
ARTIFACT_FOLDER = Path('/share/costeffectiveness/artifacts/')

ENDPOINT_DRAWS = 'draws'
ENDPOINT_SUMMARY = 'summary'
ENDPOINT_METADATA = 'metadata'


T = TypeVar('T')


class GbdServiceError(Exception):
    """Error for failures accessing GBD data."""
    pass


def _dataframe_from_response(resp) -> pd.DataFrame:
    if resp.ok:
        # return pickle.loads(gzip.decompress(resp.content))
        return resp.content.decode("utf-8")
        # return pd.read_csv(io.BytesIO(gzip.decompress(resp.content)), index_col=0)
        # return pd.read_json(resp.content)
    else:
        logger.error(f'GbdServiceError: http response code {resp}')
        raise GbdServiceError


def _build_url(endpoint_category: str, endpoint: str, params: str = None) -> str:
    url = f'{BASE_URL}/{endpoint_category}/{endpoint}'
    if params:
        url = f'{url}?{params}'
    return url


def _validate_args(gbd_id, name, location_id, location, source):
    pass


def _make_request(url: str, data_handler: Callable[..., T]) -> T:
    return data_handler(req.get(url))


def get_incidence_rate(gbd_id: int = None,
                       kind: str = None,
                       name: str = None,
                       source: str = None,
                       location_id: int = None,
                       location: str = None) -> pd.DataFrame:
    _validate_args(gbd_id, name, location_id, location, source)

    url = _build_url(ENDPOINT_DRAWS, "incidence_rate",
                    urlencode({"gbd_id": gbd_id,
                               "kind": kind,
                               "name": name,
                               "source": source,
                               "location_id": location_id,
                               "location": location}))
    return _make_request(url, _dataframe_from_response)

