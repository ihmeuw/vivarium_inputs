from typing import Union

from gbd_mapping.id import meid
from gbd_mapping.base_template import GbdRecord, ModelableEntity


class HealthcareEntity(ModelableEntity):
    """Container for healthcare system GBD ids and data."""
    __slots__ = ('name', 'gbd_id', 'utilization', 'cost')

    def __init__(self,
                 name: str,
                 gbd_id: Union[meid, None],
                 utilization: meid=None,
                 ):
        super().__init__(name=name, gbd_id=gbd_id)
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
    __slots__ = ('name', 'gbd_id')


class HealthTechnologies(GbdRecord):
    """Holder of healthcare technologies."""
    __slots__ = ('hypertension_drugs', )

    def __init__(self,
                 hypertension_drugs: HealthTechnology):
        self.hypertension_drugs = hypertension_drugs


healthcare_entities = HealthcareEntities(
    outpatient_visits=HealthcareEntity(
        name='outpatient_visits',
        gbd_id=meid(10333),
        utilization=meid(10333),
    ),
    inpatient_visits=HealthcareEntity(
        name='inpatient_visits',
        gbd_id=meid(10334),
        utilization=meid(10334),
    )
)

health_technologies = HealthTechnologies(
    hypertension_drugs=HealthTechnology(
        name='hypertension_drugs',
        gbd_id=None,
    )
)
