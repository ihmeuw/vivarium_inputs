
def _running_on_ci():
    try:
        from vivarium_inputs.globals import GbdDummy
        return True
    except:
        pass
    return False


RUNNING_ON_CI = _running_on_ci()