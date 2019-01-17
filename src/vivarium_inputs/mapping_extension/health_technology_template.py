from gbd_mapping.base_template import ModelableEntity, GbdRecord


class HealthTechnology(ModelableEntity):
    """Container for health technology ids and data."""
    __slots__ = ('name', 'kind', 'gbd_id')


class HealthTechnologies(GbdRecord):
    """Holder of healthcare technologies."""
    __slots__ = ('hypertension_drugs', )

    def __init__(self,
                 hypertension_drugs: HealthTechnology):
        self.hypertension_drugs = hypertension_drugs
