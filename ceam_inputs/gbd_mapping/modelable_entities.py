from .templates import ModelableEntities, ModelableEntity, meid

modelable_entities = ModelableEntities(
    outpatient_visits=ModelableEntity(
        name='outpatient_visits',
        id=meid(9458),
    )
)
