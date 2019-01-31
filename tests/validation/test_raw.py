import pandas as pd
import pytest

from vivarium_inputs.validation import raw
from vivarium_inputs.globals import DataAbnormalError


@pytest.mark.parametrize('mort, morb, yld_only, yll_only, match',
                         [([0, 1, 1], [0, 1, 1], True, False, 'set to 0'),
                          ([1, 1, 0], [1, 1, 1], True, False, 'only one'),
                          ([1, 1, 1], [0, 0, 0], False, False, 'not restricted to yll_only'),
                          ([0, 0, 0], [1, 1, 1], False, False, 'not restricted to yld_only'),
                          ([1, 0, 0], [0, 1, 1], True, False, 'is restricted to yld_only'),
                          ([1, 0, 2], [0, 1, 1], False, False, 'outside the expected')])
def test_check_mort_morb_flags_fail(mort, morb, yld_only, yll_only, match):
    data = pd.DataFrame({'mortality': mort, 'morbidity': morb})
    with pytest.raises(DataAbnormalError, match=match):
        raw.check_mort_morb_flags(data, yld_only, yll_only)


@pytest.mark.parametrize('mort, morb, yld_only, yll_only',
                         [([1, 1, 1], [1, 1, 1], True, False),
                          ([1, 1, 1], [1, 1, 1], False, False),
                          ([0, 0, 0], [1, 1, 1], True, False,),
                          ([1, 1, 1], [0, 0, 0], False, True,),
                          ([1, 0, 1], [0, 1, 0], False, False)])
def test_check_mort_morb_flags_pass(mort, morb, yld_only, yll_only):
    data = pd.DataFrame({'mortality': mort, 'morbidity': morb})
    raw.check_mort_morb_flags(data, yld_only, yll_only)
