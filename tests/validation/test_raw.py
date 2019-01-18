import numpy as np
import pandas as pd
import pytest

from vivarium_inputs.validation import raw
from vivarium_inputs.globals import DataAbnormalError


def test_check_years(mocker):
    gbd_mock_utilities = mocker.patch("vivarium_inputs.globals.gbd")
    gbd_mock_utilities.get_estimation_years.return_value = list(range(1990, 2015, 5)) + [2017]

    df = pd.DataFrame({'year_id': [1990, 1991]})
    with pytest.raises(DataAbnormalError, match='missing years') as e:
        raw.check_years(df, 'annual')

    # extra years should be allowed for annual
    df = pd.DataFrame({'year_id': range(1950, 2020)})
    raw.check_years(df, 'annual')

    df = pd.DataFrame({'year_id': [1990, 1995]})
    with pytest.raises(DataAbnormalError, match='missing years'):
        raw.check_years(df, 'binned')

    df = pd.DataFrame({'year_id': list(range(1990, 2015, 5)) + [2017, 2020]})
    with pytest.raises(DataAbnormalError, match='extra years'):
        raw.check_years(df, 'binned')

    df = pd.DataFrame({'year_id': list(range(1990, 2015, 5)) + [2017]})
    raw.check_years(df, 'binned')


