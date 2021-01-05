import operator
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger

from vivarium_inputs.globals import DRAW_COLUMNS, VivariumInputsError

###############################
# Shared validation utilities #
###############################


def check_value_columns_boundary(data: pd.DataFrame, boundary_value: Union[float, pd.Series], boundary_type: str,
                                 value_columns: list = DRAW_COLUMNS, inclusive: bool = True,
                                 error: type(VivariumInputsError) = None):
    """Check that all values in DRAW_COLUMNS in data are above or below given
    boundary_value.

    Parameters
    ----------
    data
        Dataframe containing `value_columns`.
    boundary_value
        Value against which `value_columns` values will be checked. May be a
        series of values with a matching index to data.
    boundary_type
        String 'upper' or 'lower' indicating whether `boundary_value` is upper
        or lower limit on `value_columns`.
    value_columns
        List of column names in `data`, the values of which should be checked
        against `boundary_value`.
    inclusive
        Boolean indicating whether `boundary_value` is inclusive or not.
    error
        Exception class indicating what error should be raised if values are
        found outside `boundary_value`. If none, warn instead of raising error.

    Raises
    -------
    ValueError
        If the passed `boundary_type` is neither 'upper' nor 'lower'.
    error
        If any values in `value_columns` are above/below `boundary_value`,
        depending on `boundary_type`, raise the passed error if there is one.
    """
    msg = (f'Data contains values {"below" if boundary_type == "lower" else "above"} '
           f'{"or equal to " if not inclusive else ""}the expected boundary '
           f'value{"s" if isinstance(boundary_value, pd.Series) else f" ({boundary_value})"}.')

    if boundary_type == "lower":
        op = operator.gt
        data_values = data[value_columns].min(axis=1)
    elif boundary_type == "upper":
        op = operator.lt
        data_values = data[value_columns].max(axis=1)
    else:
        raise ValueError(f'Boundary type must be either "lower" or "upper". You specified {boundary_type}.')

    if isinstance(boundary_value, pd.Series):
        data_values = data_values.sort_index()
        boundary_value = boundary_value.sort_index()

    within_boundary = op(data_values, boundary_value)
    if inclusive:
        within_boundary |= np.isclose(data_values, boundary_value)

    if not np.all(within_boundary):
        if error is not None:
            raise error(msg)
        else:
            logger.warning(msg)
