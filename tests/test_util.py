import pytest

from ceam_inputs.util import round_to_gbd_year

def test_round_to_gbd_year():
    assert round_to_gbd_year(1990) == 1990
    assert round_to_gbd_year(1990, down=False) == 1990

    assert round_to_gbd_year(1992) == 1990
    assert round_to_gbd_year(1992, down=False) == 1995

    assert round_to_gbd_year(1994) == 1990
    assert round_to_gbd_year(1994, down=False) == 1995

    assert round_to_gbd_year(1995) == 1995
    assert round_to_gbd_year(1995, down=False) == 1995
