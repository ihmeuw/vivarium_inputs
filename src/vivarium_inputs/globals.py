
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


class GbdDataError(VivariumInputsError):
    """Base exception for errors related to GBD data."""
    pass


class DataNotExistError(VivariumInputsError, FileNotFoundError):
    """Exception raised when the gbd data for the entity-measure pair do not exist"""
    pass


class DataAbnormalError(VivariumInputsError, ValueError):
    """Exception raised when data has extra columns or values that we do not expect to have"""
    pass


class EtlError(VivariumInputsError):
    """Base exception for errors related to vivarium_inputs improperly handling data."""
    pass


class DataFormattingError(EtlError, ValueError):
    """Exception raised when data has been improperly formatted."""
    pass


class UserError(VivariumInputsError):
    """Base exception for user errors."""
    pass


class InvalidQueryError(VivariumInputsError, ValueError):
    """Exception raised when the user makes an invalid query."""
    pass


METRICS = {
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


DEMOGRAPHIC_COLUMNS = ('location_id', 'sex_id', 'age_group_id', 'year_id')
DRAW_COLUMNS = tuple([f'draw_{i}' for i in range(1000)])
