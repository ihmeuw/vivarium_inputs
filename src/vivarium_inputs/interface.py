"""Access to vivarium simulation input data."""
import pandas as pd

from gbd_mapping import ModelableEntity
from vivarium_inputs import core, utilities
from vivarium_inputs.globals import Population
import vivarium_inputs.validation.sim as validation


def get_measure(entity: ModelableEntity, measure: str, location: str) -> pd.DataFrame:
    """Pull GBD data for measure and entity and prep for simulation input,
    including scrubbing all GBD conventions to replace IDs with meaningful
    values or ranges and expanding over all demographic dimensions.

    Available measures:

        For entity kind 'sequela':
            incidence, prevalence, birth_prevalence, disability_weight

        For entity kind 'cause':
            incidence, prevalence, birth_prevalence, disability_weight,
            remission, cause_specific_mortality, excess_mortality

        For entity kind 'coverage_gap':
            exposure, exposure_standard_deviation, exposure_distribution_weights,
            relative_risk

        For entity kind 'risk_factor':
            exposure, exposure_standard_deviation, exposure_distribution_weights,
            relative_risk, population_attributable_fraction, mediation_factors

        For entity kind 'etiology':
            population_attributable_fraction

        For entity kind 'alternative_risk_factor':
            exposure, exposure_standard_deviation, exposure_distribution_weights

        For entity kind 'covariate':
            estimate

        For entity kind 'healthcare_entity':
            cost, utilization

        For entity kind 'health_technology':
            cost

    Parameters
    ----------
    entity
        Entity for which to pull `measure`.
    measure
        Measure for which to pull data, should be a measure available for the
        kind of entity which `entity` is.
    location
        Location for which to pull data.

    Returns
    -------
    Dataframe standardized to the format expected by `vivarium` simulations.

    """
    data = core.get_data(entity, measure, location)
    data = utilities.scrub_gbd_conventions(data, location)
    validation.validate_for_simulation(data, entity, measure, location)
    return utilities.sort_data(data)


def get_population_structure(location: str) -> pd.DataFrame:
    """Pull GBD population data for the given location and standardize to the
    expected simulation input format, including scrubbing all GBD conventions
    to replace IDs with meaningful values or ranges and expanding over all
    demographic dimensions.

    Parameters
    ----------
    location
        Location for which to pull population data.

    Returns
    -------
    Dataframe of population data for `location`, standardized to the format
    expected by `vivarium` simulations.

    """
    pop = Population()
    data = core.get_data(pop, 'structure', location)
    data = utilities.scrub_gbd_conventions(data, location)
    validation.validate_for_simulation(data, pop, 'structure', location)
    return utilities.sort_data(data)


def get_theoretical_minimum_risk_life_expectancy() -> pd.DataFrame:
    """Pull GBD theoretical minimum risk life expectancy data and standardize
    to the expected simulation input format, including binning age parameters
    as expected by simulations.

    Returns
    -------
    Dataframe of theoretical minimum risk life expectancy data, standardized
    to the format expected by `vivarium` simulations with binned age parameters.

    """
    pop = Population()
    data = core.get_data(pop, 'theoretical_minimum_risk_life_expectancy', 'Global')
    validation.validate_for_simulation(data, pop, 'theoretical_minimum_risk_life_expectancy', 'Global')
    return utilities.sort_data(data)


def get_age_bins() -> pd.DataFrame:
    """Pull GBD age bin data and standardize to the expected simulation input
    format.

    Returns
    -------
    Dataframe of age bin data, with bin start and end values as well as bin
    names.

    """
    pop = Population()
    data = core.get_data(pop, 'age_bins', 'Global')
    validation.validate_for_simulation(data, pop, 'age_bins', 'Global')
    return utilities.sort_data(data)


def get_demographic_dimensions(location: str) -> pd.DataFrame:
    """Pull the full demographic dimensions for GBD data, standardized to the
    expected simulation input format, including scrubbing all GBD conventions
    to replace IDs with with meaningful values or ranges.

    Parameters
    ----------
    location
        Location for which to pull demographic dimension data.

    Returns
    -------
    Dataframe with age and year bins from GBD, sexes, and the given location.

    """
    pop = Population()
    data = core.get_data(pop, 'demographic_dimensions', location)
    data = utilities.scrub_gbd_conventions(data, location)
    validation.validate_for_simulation(data, pop, 'demographic_dimensions', location)
    return utilities.sort_data(data)
