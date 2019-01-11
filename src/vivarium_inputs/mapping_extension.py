from typing import Union, Tuple

from gbd_mapping.id import meid, reiid
from gbd_mapping.base_template import GbdRecord, ModelableEntity
from gbd_mapping.cause_template import Cause
from gbd_mapping.cause import causes

class HealthcareEntity(ModelableEntity):
    """Container for healthcare system GBD ids and data."""
    __slots__ = ('name', 'kind', 'gbd_id', 'utilization', 'cost')

    def __init__(self,
                 name: str,
                 kind: str,
                 gbd_id: Union[meid, None],
                 utilization: meid=None,
                 ):
        super().__init__(name=name, kind=kind, gbd_id=gbd_id)
        self.utilization = utilization


class HealthcareEntities(GbdRecord):
    """Holder of healthcare modelable entities"""
    __slots__ = ('outpatient_visits', 'inpatient_visits')

    def __init__(self,
                 outpatient_visits: HealthcareEntity,
                 inpatient_visits: HealthcareEntity):

        super().__init__()
        self.outpatient_visits = outpatient_visits
        self.inpatient_visits = inpatient_visits


class HealthTechnology(ModelableEntity):
    """Container for health technology ids and data."""
    __slots__ = ('name', 'kind', 'gbd_id')


class HealthTechnologies(GbdRecord):
    """Holder of healthcare technologies."""
    __slots__ = ('hypertension_drugs', )

    def __init__(self,
                 hypertension_drugs: HealthTechnology):
        self.hypertension_drugs = hypertension_drugs


class AlternativeRiskFactor(ModelableEntity):
    """Container for risk factor outside of GBD and its metadata."""
    __slots__ = ('name', 'kind', 'gbd_id', 'distribution', 'affected_causes')
    def __init__(self,
                 name: str,
                 kind: str,
                 gbd_id: Union[reiid, None],
                 distribution: str,
                 affected_causes: Tuple[Cause, ...]):
        super().__init__(name=name, kind=kind, gbd_id=gbd_id)
        self.distribution=distribution
        self.affected_causes=affected_causes


class AlternativeRiskFactors(GbdRecord):
    """Holder of alternative risk factors."""
    __slots__ =('child_wasting', 'child_underweight', 'child_stunting', )

    def __init__(self,
                 child_stunting: AlternativeRiskFactor,
                 child_underweight: AlternativeRiskFactor,
                 child_wasting: AlternativeRiskFactor,):
        self.child_stunting = child_stunting
        self.child_underweight = child_underweight
        self.child_wasting = child_wasting


healthcare_entities = HealthcareEntities(
    outpatient_visits=HealthcareEntity(
        name='outpatient_visits',
        kind='healthcare_entity',
        gbd_id=meid(10333),
        utilization=meid(10333),
    ),
    inpatient_visits=HealthcareEntity(
        name='inpatient_visits',
        kind='healthcare_entity',
        gbd_id=meid(10334),
        utilization=meid(10334),
    )
)

health_technologies = HealthTechnologies(
    hypertension_drugs=HealthTechnology(
        name='hypertension_drugs',
        kind='health_technology',
        gbd_id=None,
    )
)


alternative_risk_factors = AlternativeRiskFactors(
    child_stunting=AlternativeRiskFactor(
        name='child_stunting',
        kind='alternative_risk_factor',
        gbd_id=241,
        distribution='ensemble',
        affected_causes=(causes.all_causes, causes.communicable_maternal_neonatal_and_nutritional_diseases,
                         causes.diarrheal_diseases, causes.lower_respiratory_infections, causes.measles,
                         causes.respiratory_infections_and_tuberculosis, causes.enteric_infections,
                         causes.other_infectious_diseases, ),
    ),
    child_wasting=AlternativeRiskFactor(
        name='child_wasting',
        kind='alternative_risk_factor',
        gbd_id=240,
        distribution='ensemble',
        affected_causes=(causes.all_causes, causes.communicable_maternal_neonatal_and_nutritional_diseases,
                         causes.diarrheal_diseases, causes.lower_respiratory_infections, causes.measles,
                         causes.nutritional_deficiencies, causes.protein_energy_malnutrition,
                         causes.respiratory_infections_and_tuberculosis, causes.enteric_infections,
                         causes.other_infectious_diseases, ),
    ),
    child_underweight=AlternativeRiskFactor(
        name='child_underweight',
        kind='alternative_risk_factor',
        gbd_id=94,
        distribution='ensemble',
        affected_causes=(causes.all_causes, causes.communicable_maternal_neonatal_and_nutritional_diseases,
                         causes.diarrheal_diseases, causes.lower_respiratory_infections, causes.measles,
                         causes.nutritional_deficiencies, causes.protein_energy_malnutrition,
                         causes.respiratory_infections_and_tuberculosis, causes.enteric_infections,
                         causes.other_infectious_diseases, ),
    )
)
