"""Access to vivarium simulation input data."""
from typing import List, Optional, Union

import pandas as pd
from gbd_mapping import ModelableEntity

import vivarium_inputs.validation.sim as validation
from vivarium_inputs import core, extract, utilities, utility_data
from vivarium_inputs.globals import Population


def get_measure(
    entity: ModelableEntity,
    measure: str,
    location: Union[int, str, List[Union[int, str]]],
    years: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    """Pull GBD data for measure and entity and prep for simulation input,
    including scrubbing all GBD conventions to replace IDs with meaningful
    values or ranges and expanding over all demographic dimensions. To pull data
    using this function, please have at least 50GB of memory available.

    Available measures:

        For entity kind 'sequela':
            incidence_rate, prevalence, birth_prevalence, disability_weight

        For entity kind 'cause':
            incidence_rate, prevalence, birth_prevalence, disability_weight,
            remission_rate, cause_specific_mortality_rate, excess_mortality_rate

        For entity kind 'risk_factor':
            exposure, exposure_standard_deviation, exposure_distribution_weights,
            relative_risk, population_attributable_fraction, mediation_factors

        For entity kind 'etiology':
            population_attributable_fraction

        For entity kind 'alternative_risk_factor':
            exposure, exposure_standard_deviation, exposure_distribution_weights

        For entity kind 'covariate':
            estimate

    Parameters
    ----------
    entity
        Entity for which to pull `measure`.
    measure
        Measure for which to pull data, should be a measure available for the
        kind of entity which `entity` is.
    location
        Location for which to pull data. This can be a location id as an int, the location name
        as a string, or a list of these two data types.
    years
        Years for which to extract data. If None, get most recent year. If 'all',
        get all available data. Defaults to None.

    Returns
    -------
    pandas.DataFrame
        Dataframe standardized to the format expected by `vivarium` simulations.

    """
    data = core.get_data(entity, measure, location, years)
    data = utilities.scrub_gbd_conventions(data, location)
    validation.validate_for_simulation(data, entity, measure, location, years)
    data = utilities.split_interval(data, interval_column="age", split_column_prefix="age")
    data = utilities.split_interval(data, interval_column="year", split_column_prefix="year")
    return utilities.sort_hierarchical_data(data)


def get_population_structure(
    location: Union[int, str, List[Union[int, str]]],
    years: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    """Pull GBD population data for the given location and standardize to the
    expected simulation input format, including scrubbing all GBD conventions
    to replace IDs with meaningful values or ranges and expanding over all
    demographic dimensions.

    Parameters
    ----------
    location
        Location for which to pull population data.
    years
        Years for which to extract data. If None, get most recent year. If 'all',
        get all available data. Defaults to None.

    Returns
    -------
    pandas.DataFrame
        Dataframe of population data for `location`, standardized to the format
        expected by `vivarium` simulations.

    """
    pop = Population()
    data = core.get_data(pop, "structure", location, years)
    data = utilities.scrub_gbd_conventions(data, location)
    validation.validate_for_simulation(data, pop, "structure", location, years)
    data = utilities.split_interval(data, interval_column="age", split_column_prefix="age")
    data = utilities.split_interval(data, interval_column="year", split_column_prefix="year")
    return utilities.sort_hierarchical_data(data)


def get_theoretical_minimum_risk_life_expectancy() -> pd.DataFrame:
    """Pull GBD theoretical minimum risk life expectancy data and standardize
    to the expected simulation input format, including binning age parameters
    as expected by simulations.

    Returns
    -------
    pandas.DataFrame
        Dataframe of theoretical minimum risk life expectancy data, standardized
        to the format expected by `vivarium` simulations with binned age parameters.

    """
    pop = Population()
    data = core.get_data(pop, "theoretical_minimum_risk_life_expectancy", "Global")
    data = utilities.set_age_interval(data)
    validation.validate_for_simulation(
        data, pop, "theoretical_minimum_risk_life_expectancy", "Global"
    )
    data = utilities.split_interval(data, interval_column="age", split_column_prefix="age")
    data = utilities.split_interval(data, interval_column="year", split_column_prefix="year")
    return utilities.sort_hierarchical_data(data)


def get_age_bins() -> pd.DataFrame:
    """Pull GBD age bin data and standardize to the expected simulation input
    format.

    Returns
    -------
    pandas.DataFrame
        Dataframe of age bin data, with bin start and end values as well as bin
        names.

    """
    pop = Population()
    data = core.get_data(pop, "age_bins", "Global")
    data = utilities.set_age_interval(data)
    validation.validate_for_simulation(data, pop, "age_bins", "Global")
    data = utilities.split_interval(data, interval_column="age", split_column_prefix="age")
    data = utilities.split_interval(data, interval_column="year", split_column_prefix="year")
    return utilities.sort_hierarchical_data(data)


def get_demographic_dimensions(
    location: Union[int, str, List[Union[int, str]]],
    years: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    """Pull the full demographic dimensions for GBD data, standardized to the
    expected simulation input format, including scrubbing all GBD conventions
    to replace IDs with meaningful values or ranges.

    Parameters
    ----------
    location
        Location for which to pull demographic dimension data.
    years
        Years for which to extract data. If None, get most recent year. If 'all',
        get all available data. Defaults to None.

    Returns
    -------
    pandas.DataFrame
        Dataframe with age and year bins from GBD, sexes, and the given location.

    """
    pop = Population()
    data = core.get_data(pop, "demographic_dimensions", location, years=years)
    data = utilities.scrub_gbd_conventions(data, location)
    validation.validate_for_simulation(data, pop, "demographic_dimensions", location, years)
    data = utilities.split_interval(data, interval_column="age", split_column_prefix="age")
    data = utilities.split_interval(data, interval_column="year", split_column_prefix="year")
    return utilities.sort_hierarchical_data(data)


def get_raw_data(
    entity: ModelableEntity,
    measure: str,
    location: Union[int, str, List[Union[int, str]]],
    years: Optional[Union[int, str, List[int]]] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Pull raw data from GBD for the requested entity, measure, and location.
    Skip standard raw validation checks in order to return data that can be
    investigated for oddities. The only filter that occurs is by applicable
    measure id, metric id, or to most detailed causes where relevant.

    Available measures:

        For entity kind 'sequela':
            incidence_rate, prevalence, birth_prevalence, disability_weight

        For entity kind 'cause':
            incidence_rate, prevalence, birth_prevalence, disability_weight,
            remission_rate, deaths

        For entity kind 'risk_factor':
            exposure, exposure_standard_deviation, exposure_distribution_weights,
            relative_risk, population_attributable_fraction, mediation_factors

        For entity kind 'etiology':
            population_attributable_fraction

        For entity kind 'alternative_risk_factor':
            exposure, exposure_standard_deviation, exposure_distribution_weights

        For entity kind 'covariate':
            estimate

        For entity kind 'population':
            structure, theoretical_minimum_risk_life_expectancy

    Parameters
    ----------
    entity
        Entity for which to extract data.
    measure
        Measure for which to extract data.
    location
        Location for which to extract data.
    years
        Years for which to extract data. If None, get most recent year. If 'all',
        get all available data. Defaults to None.

    Returns
    -------
    Union[pandas.Series, pandas.DataFrame]
        Data for the entity-measure pair and specific location requested, with no
        formatting or reshaping.
    """
    if not isinstance(location, list):
        location = [location]
    location_id = [
        utility_data.get_location_id(loc) if isinstance(loc, str) else loc for loc in location
    ]
    data = extract.extract_data(
        entity,
        measure,
        location_id,
        validate=False,
        years=years,
    )
    return data
