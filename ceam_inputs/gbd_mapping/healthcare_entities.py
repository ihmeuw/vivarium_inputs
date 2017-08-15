from .templates import HealthcareEntities, HealthcareEntity, meid

healthcare_entities = HealthcareEntities(
    outpatient_visits=HealthcareEntity(
        name='outpatient_visits',
        gbd_id=meid(9458),
        proportion=meid(9458),
    )
)
