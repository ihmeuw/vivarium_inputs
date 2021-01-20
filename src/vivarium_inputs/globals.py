"""Global constants, errors, and module imports for inputs processing."""
from gbd_mapping import ModelableEntity, causes, risk_factors


# The purpose of this import block is to mask the dependency on internal
# IHME data and allow CI and automated testing to work.
try:
    from vivarium_gbd_access import gbd

    try:
        from gbd_artifacts.exceptions import NoBestVersionError
        from get_draws.api import EmptyDataFrameException, InputsException
    except  ModuleNotFoundError:
        raise RuntimeError("Problem importing gbd_artifacts.exceptions or get_draws.api.")

except ModuleNotFoundError:
    class GbdDummy:
        """Mock class to wrap internal dependency."""
        def __getattr__(self, item):
            raise ModuleNotFoundError("Required package vivarium_gbd_access not found.")
    gbd = GbdDummy()

    class NoBestVersionError(Exception):
        """Mock class for gbd exception"""
        pass

    class EmptyDataFrameException(Exception):
        """Mock class for gbd exception"""
        pass

    class InputsException(Exception):
        """Mock class for gbd exception"""
        pass


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
    'Incidence rate': 6,
    'Remission rate': 7,
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

# Cause-risk pair where risk may have a protective effect on a certain cause with negative paf
PROTECTIVE_CAUSE_RISK_PAIRS = {
    'high_body_mass_index_in_adults': [causes.neoplasms, causes.breast_cancer, causes.esophageal_cancer]
}

# Keep track of special cases for the sim validator boundary checks where the
# standard boundary in sim validators won't cut it
BOUNDARY_SPECIAL_CASES = {
    'excess_mortality_rate': {
        'Ecuador': {
            'measles': 250_000_000
        },
        'China': {
            'subarachnoid_hemorrhage': 3_775
        },
        'Russian Federation': {
            'subarachnoid_hemorrhage': 800
        },
        'South Africa': {
            'subarachnoid_hemorrhage': 500
        },
        'Bangladesh': {
            'subarachnoid_hemorrhage': 2_750
        },
        'Ukraine': {
            'subarachnoid_hemorrhage': 500
        }
    }
}

RISKS_WITH_NEGATIVE_PAF = [
    risk_factors.vitamin_a_deficiency.name,
]

# residual cat is added by get_draws but all cats modeled for lbwsg so
# has to be removed
EXTRA_RESIDUAL_CATEGORY = {risk_factors.low_birth_weight_and_short_gestation.name: 'cat125'}

# Some data contain very small (5.56e-311) values that produce floating point errors, clip them
MINIMUM_EXPOSURE_VALUE = 1e-10

PROBLEMATIC_RISKS = {risk_factors.zinc_deficiency.name:
                         "zinc deficiency relative risk data breaks central comp interpolation."}

NON_MAX_TMREL = {risk_factors.low_birth_weight_and_short_gestation.name: 'cat56'}

# LBWSG paf has data outside neonatal preterm birth age restrictions (but is all 1.0) - K.W. 4/2/19
PAF_OUTSIDE_AGE_RESTRICTIONS = {risk_factors.low_birth_weight_and_short_gestation.name: [causes.neonatal_preterm_birth]}

def EXCLUDE_ABNORMAL_DATA(entity, context):
    # Add expressions to the list that evaluate to True if the subsequent validation
    #  check should be excluded
    ABNORMAL_CONDITIONS = [
        # The data for sex is incorrect in this situation (location is Pakistan)
        (entity.kind == 'sequela'
            and entity.name == 'moderate_wasting_with_edema'
            and context.context_data['location_id'] == 165),
    ]
    for condition in ABNORMAL_CONDITIONS:
        if condition:
            return True
    return False

# GBD uses modelable entity 10488, which is attached to anemia not iron deficiency, for iron deficiency SD
OTHER_MEID = {risk_factors.iron_deficiency.name: 10488}

class Population(ModelableEntity):
    """Entity wrapper for querying population measures."""
    def __init__(self):
        super().__init__(name='population', kind='population', gbd_id=None)
