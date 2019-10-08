import gzip
import io
from pathlib import Path
import pandas as pd
import requests as req
from loguru import logger


TIMEOUT_SERVICE = None

SERVICE_VERSION = '1'
BASE_URL = f'http://microsim-rancher-p01.hosts.ihme.washington.edu:5000/v{SERVICE_VERSION}'


class GbdServiceError(Exception):
    """Error for failures accessing GBD data."""
    pass


def dataframe_from_response(resp: req.Response) -> pd.DataFrame:
    if resp.ok:
        return pd.read_csv(io.BytesIO(gzip.decompress(resp.content)), index_col=0)
    else:
        logger.error(f'GbdServiceError: http response code {resp}')
        raise GbdServiceError


def build_url(endpoint_category: str, endpoint: str, params: str = None) -> str:
    url = f'{BASE_URL}/{endpoint_category}/{endpoint}'
    if params:
        url = f'{url}?{params}'
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
