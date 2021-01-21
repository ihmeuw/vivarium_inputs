import gbd_mapping
import numpy as np
import pandas as pd
import pytest
from enum import IntFlag

from gbd_mapping import causes, risk_factors, covariates, ModelableEntity
from vivarium_inputs.interface import get_measure
from vivarium_inputs import utility_data
from vivarium_inputs.mapping_extension import healthcare_entities
from tests.extract.check import RUNNING_ON_CI


pytestmark = pytest.mark.skipif(RUNNING_ON_CI, reason="Don't run these tests on the CI server")


def success_expected(entity_name, measure_name, location):
    df = get_measure(entity_name, measure_name, location)
    return df


def fail_expected(entity_name, measure_name, location):
    with pytest.raises(Exception):
        df = get_measure(entity_name, measure_name, location)


class MCFlag(IntFlag):
    """
        Use the idea of a bit field with support from python's enum type IntFlag
        See here for general information on bit fields:
        https://en.wikipedia.org/wiki/Bit_field

        And here for python's enum type:
        https://docs.python.org/3.6/library/enum.html
    """
    INCIDENCE_RATE = 1
    RAW_INCIDENCE_RATE = 2
    PREVALENCE = 4
    BIRTH_PREVALENCE = 8
    DISABILITY_WEIGHT = 16
    REMISSION_RATE = 32
    CAUSE_SPECIFIC_MORTALITY_RATE = 64
    EXCESS_MORTALITY_RATE = 128
    DEATHS = 512


entity = [
    (causes.measles,
        MCFlag.INCIDENCE_RATE
        | MCFlag.PREVALENCE
        | MCFlag.DISABILITY_WEIGHT
        | MCFlag.CAUSE_SPECIFIC_MORTALITY_RATE
        | MCFlag.EXCESS_MORTALITY_RATE
        ),
    (causes.diarrheal_diseases,
        MCFlag.INCIDENCE_RATE
        | MCFlag.PREVALENCE
        | MCFlag.DISABILITY_WEIGHT
        | MCFlag.REMISSION_RATE
        | MCFlag.CAUSE_SPECIFIC_MORTALITY_RATE
        | MCFlag.EXCESS_MORTALITY_RATE
        ),
    (causes.diabetes_mellitus_type_2,
        MCFlag.INCIDENCE_RATE
        | MCFlag.PREVALENCE
        | MCFlag.DISABILITY_WEIGHT
        | MCFlag.CAUSE_SPECIFIC_MORTALITY_RATE
        | MCFlag.EXCESS_MORTALITY_RATE
        ),
    ]
measures = [
    ('incidence_rate', MCFlag.INCIDENCE_RATE),
    ('prevalence', MCFlag.PREVALENCE),
    ('birth_prevalence', MCFlag.BIRTH_PREVALENCE),
    ('disability_weight', MCFlag.DISABILITY_WEIGHT),
    ('remission_rate', MCFlag.REMISSION_RATE),
    ('cause_specific_mortality_rate', MCFlag.CAUSE_SPECIFIC_MORTALITY_RATE),
    ('excess_mortality_rate', MCFlag.EXCESS_MORTALITY_RATE),
    ]
locations = ['India']

@pytest.mark.parametrize('entity', entity, ids=lambda x: x[0].name)
@pytest.mark.parametrize('measure', measures, ids=lambda x: x[0])
@pytest.mark.parametrize('location', locations)
def test_get_measure_causelike(entity, measure, location):
    entity_name, entity_expected_measure_ids = entity
    measure_name, measure_id = measure
    tester = success_expected if (entity_expected_measure_ids & measure_id) else fail_expected
    df = tester(entity_name, measure_name, utility_data.get_location_id(location))


class MRFlag(IntFlag):
    """
        Use the idea of a bit field with support from python's enum type IntFlag
        See here for general information on bit fields:
        https://en.wikipedia.org/wiki/Bit_field

        And here for python's enum type:
        https://docs.python.org/3.6/library/enum.html
    """
    EXPOSURE = 1
    EXPOSURE_SD = 2
    EXPOSURE_DIST_WEIGHTS = 4
    RELATIVE_RISK = 8
    PAF = 16
    ALL = (EXPOSURE | EXPOSURE_SD | EXPOSURE_DIST_WEIGHTS
            | RELATIVE_RISK | PAF)


entity_r = [
    (risk_factors.high_systolic_blood_pressure,
        MRFlag.EXPOSURE | MRFlag.EXPOSURE_SD | MRFlag.EXPOSURE_DIST_WEIGHTS
        | MRFlag.RELATIVE_RISK | MRFlag.PAF),
    (risk_factors.low_birth_weight_and_short_gestation,
        MRFlag.EXPOSURE | MRFlag.RELATIVE_RISK | MRFlag.PAF)]
measures_r = [
    ('exposure', MRFlag.EXPOSURE),
    ('exposure_standard_deviation', MRFlag.EXPOSURE_SD),
    ('exposure_distribution_weights', MRFlag.EXPOSURE_DIST_WEIGHTS),
    ('relative_risk', MRFlag.RELATIVE_RISK),
    ('population_attributable_fraction', MRFlag.PAF),
    ]
locations_r = ['India']
@pytest.mark.parametrize('entity', entity_r, ids=lambda x: x[0].name)
@pytest.mark.parametrize('measure', measures_r, ids=lambda x: x[0])
@pytest.mark.parametrize('location', locations_r)
def test_get_measure_risklike(entity, measure, location):
    entity_name, entity_expected_measure_ids = entity
    measure_name, measure_id = measure
    tester = success_expected if (entity_expected_measure_ids & measure_id) else fail_expected
    df = tester(entity_name, measure_name, utility_data.get_location_id(location))


entity_cov = [
    covariates.systolic_blood_pressure_mmhg,
]   
measures_cov = [
    'estimate'
]
locations_cov = ['India']
@pytest.mark.parametrize('entity', entity_cov, ids=lambda x: x.name)
@pytest.mark.parametrize('measure', measures_cov, ids=lambda x: x)
@pytest.mark.parametrize('location', locations_cov)
def test_get_measure_covariatelike(entity, measure, location):
    df = get_measure(entity, measure, utility_data.get_location_id(location))
