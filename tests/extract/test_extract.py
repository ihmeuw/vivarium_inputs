import gbd_mapping
import numpy as np
import pandas as pd
import pytest
from enum import IntFlag

from gbd_mapping import causes, risk_factors, covariates, ModelableEntity
from vivarium_inputs import extract, utility_data
from tests.extract.check import RUNNING_ON_CI


pytestmark = pytest.mark.skipif(RUNNING_ON_CI, reason="Don't run these tests on the CI server")


VALIDATE_FLAG = False


def success_expected(entity_name, measure_name, location):
    df = extract.extract_data(entity_name, measure_name, location, validate=VALIDATE_FLAG)
    return df


def fail_expected(entity_name, measure_name, location):
    with pytest.raises(Exception):
        df = extract.extract_data(entity_name, measure_name, location, validate=VALIDATE_FLAG)


class MCFlag(IntFlag):
    """
        Use the idea of a bit field with support from python's enum type IntFlag
        See here for general information on bit fields:
        https://en.wikipedia.org/wiki/Bit_field

        And here for python's enum type:
        https://docs.python.org/3.6/library/enum.html
    """
    INCIDENCE_RATE = 1
    PREVALENCE = 2
    BIRTH_PREVALENCE = 4
    DISABILITY_WEIGHT = 8
    REMISSION_RATE = 16
    DEATHS = 32
    ALL = INCIDENCE_RATE | PREVALENCE | BIRTH_PREVALENCE | DISABILITY_WEIGHT | REMISSION_RATE | DEATHS


entity = [
    (causes.measles, MCFlag.INCIDENCE_RATE
        | MCFlag.PREVALENCE
        | MCFlag.BIRTH_PREVALENCE
        | MCFlag.REMISSION_RATE
        | MCFlag.DEATHS),
    (causes.diabetes_mellitus_type_2,
        MCFlag.INCIDENCE_RATE
        | MCFlag.PREVALENCE
        | MCFlag.BIRTH_PREVALENCE
        | MCFlag.DEATHS),
    ]
measures = [
    ('incidence_rate', MCFlag.INCIDENCE_RATE),
    ('prevalence', MCFlag.PREVALENCE),
    ('birth_prevalence', MCFlag.BIRTH_PREVALENCE),
    ('disability_weight', MCFlag.DISABILITY_WEIGHT),
    ('remission_rate', MCFlag.REMISSION_RATE),
    ('deaths', MCFlag.DEATHS)]
locations = ['India']

@pytest.mark.parametrize('entity', entity, ids=lambda x: x[0].name)
@pytest.mark.parametrize('measure', measures, ids=lambda x: x[0])
@pytest.mark.parametrize('location', locations)
def test_extract_causelike(entity, measure, location):
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
    ETIOLOGY_PAF = 32
    MEDIATION_FACTORS = 64
    ALL = (EXPOSURE | EXPOSURE_SD | EXPOSURE_DIST_WEIGHTS
            | RELATIVE_RISK | PAF | ETIOLOGY_PAF | MEDIATION_FACTORS)


entity_r = [
    (risk_factors.high_systolic_blood_pressure,
        MRFlag.EXPOSURE | MRFlag.EXPOSURE_SD | MRFlag.EXPOSURE_DIST_WEIGHTS
        | MRFlag.RELATIVE_RISK | MRFlag.PAF | MRFlag.ETIOLOGY_PAF | MRFlag.MEDIATION_FACTORS),
    (risk_factors.low_birth_weight_and_short_gestation,
        MRFlag.EXPOSURE | MRFlag.RELATIVE_RISK | MRFlag.PAF | MRFlag.ETIOLOGY_PAF)]
measures_r = [
    ('exposure', MRFlag.EXPOSURE),
    ('exposure_standard_deviation', MRFlag.EXPOSURE_SD),
    ('exposure_distribution_weights', MRFlag.EXPOSURE_DIST_WEIGHTS),
    ('relative_risk', MRFlag.RELATIVE_RISK),
    ('population_attributable_fraction', MRFlag.PAF),
    ('etiology_population_attributable_fraction', MRFlag.ETIOLOGY_PAF),
    ('mediation_factors', MRFlag.MEDIATION_FACTORS)]
locations_r = ['India']
@pytest.mark.parametrize('entity', entity_r, ids=lambda x: x[0].name)
@pytest.mark.parametrize('measure', measures_r, ids=lambda x: x[0])
@pytest.mark.parametrize('location', locations_r)
def test_extract_risklike(entity, measure, location):
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
@pytest.mark.parametrize('entity', entity_cov)
@pytest.mark.parametrize('measure', measures_cov)
@pytest.mark.parametrize('location', locations_cov)
def test_extract_covariatelike(entity, measure, location):
    df = extract.extract_data(entity, measure, utility_data.get_location_id(location), validate=VALIDATE_FLAG)


@pytest.mark.parametrize('measures',
    ['structure',
    'theoretical_minimum_risk_life_expectancy'])
def test_extract_population(measures):
    pop = ModelableEntity('ignored', 'population', None)
    df = extract.extract_data(pop, measures, utility_data.get_location_id('India'), validate=VALIDATE_FLAG)
