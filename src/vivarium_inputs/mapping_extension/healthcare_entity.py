from gbd_mapping.id import meid

from .healthcare_entity_template import HealthcareEntities, HealthcareEntity

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
