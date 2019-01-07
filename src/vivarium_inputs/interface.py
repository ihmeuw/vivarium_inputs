"""Access to vivarium simulation input data."""
from vivarium_inputs import core, utilities
import vivarium_inputs.validation.sim as validation


def get_measure(entity, measure, location):
    data = core.get_data(entity, measure, location)
    data = utilities.scrub_gbd_conventions(data, location)
    validation.validate_for_simulation(data, entity, measure, location)
    return data
