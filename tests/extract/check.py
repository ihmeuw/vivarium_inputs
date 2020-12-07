
def _no_gbd_access():
    try:
        from vivarium_inputs.globals import GbdDummy
        return True
    except ImportError:
        pass
    return False


RUNNING_ON_CI = _no_gbd_access()