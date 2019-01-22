from gbd_mapping.id import meid

from .healthcare_entity_template import HealthcareEntities, HealthcareEntity

healthcare_entities = HealthcareEntities(
    outpatient_visits=HealthcareEntity(
        name='outpatient_visits',
        kind='healthcare_entity',
        gbd_id=meid(19797),
        utilization=meid(19797),
    ),
    inpatient_visits=HealthcareEntity(
        name='inpatient_visits',
        kind='healthcare_entity',
        gbd_id=meid(18749),
        utilization=meid(18749),
    )
)
