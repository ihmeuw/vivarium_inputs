import operator
from typing import Union
import warnings

import numpy as np
import pandas as pd

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
    error
        If any values in `value_columns` are above/below `boundary_value`,
        depending on `boundary_type`, raise the passed error if there is one.
    """
    msg = (f'Data contains values {"below" if boundary_type == "lower" else "above"} '
           f'{"or equal to " if not inclusive else ""}the expected boundary '
           f'value{"s" if isinstance(boundary_value, pd.Series) else f" ({boundary_value})"}.')

    if boundary_type == "lower":
        op = operator.ge if inclusive else operator.gt
        data_values = data[value_columns].min(axis=1)
    else:
        op = operator.le if inclusive else operator.lt
        data_values = data[value_columns].max(axis=1)

    if isinstance(boundary_value, pd.Series):
        data_values = data_values.sort_index()
        boundary_value = boundary_value.sort_index()

    if not np.all(op(data_values, boundary_value)):
        if error is not None:
            raise error(msg)
        else:
            warnings.warn(msg)
