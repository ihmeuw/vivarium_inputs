"""Global constants, errors, and module imports for inputs processing."""
from typing import NamedTuple


# The purpose of this import block is to mask the dependency on internal
# IHME data and allow CI and automated testing to work.
try:
    from vivarium_gbd_access import gbd
except ModuleNotFoundError:
    class GbdDummy:
        """Mock class to wrap internal dependency."""
        def __getattr__(self, item):
            raise ModuleNotFoundError("Required package vivarium_gbd_access not found.")
    gbd = GbdDummy()


class VivariumInputsError(Exception):
    """Base exception for errors in vivarium_inputs."""
    pass


class DataDoesNotExistError(VivariumInputsError, FileNotFoundError):
    """Exception raised when requested GBD data does not exist."""
    pass


class DataAbnormalError(VivariumInputsError, ValueError):
    """Exception raised when raw GBD data does not have an expected format."""
    pass


class DataTransformationError(VivariumInputsError, ValueError):
    """Exception raised when data is incorrectly transformed."""
    pass


class InvalidQueryError(VivariumInputsError, ValueError):
    """Exception raised when the user makes an invalid query."""
    pass


METRICS = {
    # Mapping of GBD metric names to their GBD ids.
    'Number': 1,
    'Percent': 2,
    'Rate': 3,
    'Rank': 4,
    'Years': 5,
    'p-value': 6,
    'MDG p-value': 7,
    'Probability of death': 8,
    'Index score': 9
}

MEASURES = {
    # Mapping of GBD measure names to their GBD ids.
    'Deaths': 1,
    'DALYs': 2,
    'YLDs': 3,
    'YLLs': 4,
    'Prevalence': 5,
    'Incidence': 6,
    'Remission': 7,
    'Duration': 8,
    'Excess mortality rate': 9,
    'Prevalence * excess mortality rate': 10,
    'Relative risk': 11,
    'Standardized mortality ratio': 12,
    'With-condition mortality rate': 13,
    'All-cause mortality rate': 14,
    'Cause-specific mortality rate': 15,
    'Other cause mortality rate': 16,
    'Case fatality rate': 17,
    'Proportion': 18,
    'Continuous': 19,
    'Survival Rate': 20,
    'Disability Weight': 21,
    'Chronic Prevalence': 22,
    'Acute Prevalence': 23,
    'Acute Incidence': 24,
    'Maternal mortality ratio': 25,
    'Life expectancy': 26,
    'Probability of death': 27,
    'HALE (Healthy life expectancy)': 28,
    'Summary exposure value': 29,
    'Life expectancy no-shock hiv free': 30,
    'Life expectancy no-shock with hiv': 31,
    'Probability of death no-shock hiv free': 32,
    'Probability of death no-shock with hiv': 33,
    'Mortality risk': 34,
    'Short term prevalence': 35,
    'Long term prevalence': 36,
    'Life expectancy decomposition by cause': 37,
    'Birth prevalence': 38,
    'Susceptible population fraction': 39,
    'With Condition population fraction': 40,
    'Susceptible incidence': 41,
    'Total incidence': 42,
    'HAQ Index (Healthcare Access and Quality Index)': 43,
    'Population': 44,
    'Fertility': 45
}

# List of standard demographic id column names.
DEMOGRAPHIC_COLUMNS = ['location_id', 'sex_id', 'age_group_id', 'year_id']
# List of standard GBD draw column names.
DRAW_COLUMNS = [f'draw_{i}' for i in range(1000)]
# Mapping of GBD sex ids
SEXES = {'Male': 1,
         'Female': 2,
         'Combined': 3}
# Mapping of non-standard age group ids sometimes found in GBD data
SPECIAL_AGES = {'all_ages': 22,
                'age_standardized': 27}


class Population(NamedTuple):
    """Entity wrapper for querying population measures."""
    kind: str
