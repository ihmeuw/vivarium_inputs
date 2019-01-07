"""Access to vivarium simulation input data."""
from collections import namedtuple

from vivarium_inputs import core, utilities
import vivarium_inputs.validation.sim as validation


def get_measure(entity, measure, location):
    data = core.get_data(entity, measure, location)
    data = utilities.scrub_gbd_conventions(data, location)
    validation.validate_for_simulation(data, entity, measure, location)
    return utilities.sort_data(data)


def get_population_structure(location):
    pop = namedtuple('Population', 'kind')('population')
    data = core.get_data(pop, 'structure', location)
    data = utilities.scrub_gbd_conventions(data, location)
    validation.validate_for_simulation(data, pop, 'structure', location)
    return utilities.sort_data(data)


def get_theoretical_minimum_risk_life_expectancy():
    pop = namedtuple('Population', 'kind')('population')
    data = core.get_data(pop, 'theoretical_minimum_risk_life_expectancy', 'Global')
    validation.validate_for_simulation(data, pop, 'theoretical_minimum_risk_life_expectancy', 'Global')
    return utilities.sort_data(data)
