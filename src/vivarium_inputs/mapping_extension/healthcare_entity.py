from gbd_mapping.id import me_id

from .healthcare_entity_template import HealthcareEntities, HealthcareEntity

healthcare_entities = HealthcareEntities(
    outpatient_visits=HealthcareEntity(
        name='outpatient_visits',
        kind='healthcare_entity',
        gbd_id=me_id(19797),
        utilization=me_id(19797),
    ),
    inpatient_visits=HealthcareEntity(
        name='inpatient_visits',
        kind='healthcare_entity',
        gbd_id=me_id(18749),
        utilization=me_id(18749),
    )
)
