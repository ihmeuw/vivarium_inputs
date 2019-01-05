"""Access to vivarium simulation input data."""
from vivarium_inputs import core, validation, utilities


def get_measure(entity, measure, location):
    data = core.get_data(entity, measure, location)
    data = utilities.scrub_gbd_conventions(data)
    validation.validate_for_simulation(data, entity, measure, location)
    return data
