"""Access to vivarium simulation input data."""
from vivarium_inputs import core, validation, utilities


def get_measure(entity, measure, location):
    core_func = getattr(core, f'get_{measure}')
    data = core_func(entity, location)
    data = utilities.scrub_gbd_conventions(data)
    validation.validate_for_simulation(data, entity, measure, location)
    return data
