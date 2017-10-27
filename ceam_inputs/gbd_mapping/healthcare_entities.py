from typing import Union
from .templates import GbdRecord, ModelableEntity, meid


class HealthcareEntity(ModelableEntity):
    """Container for healthcare system GBD ids and data."""
    __slots__ = ('name', 'gbd_id', 'proportion', 'cost')

    def __init__(self,
                 name: str,
                 gbd_id: Union[meid, None],
                 proportion: meid = None,
                 cost: str = None,):
        super().__init__(name=name, gbd_id=gbd_id)
        self.proportion = proportion
        self.cost = cost


class HealthcareEntities(GbdRecord):
    """Holder of healthcare modelable entities"""
    __slots__ = ('outpatient_visits', 'inpatient_visits')

    def __init__(self,
                 outpatient_visits: HealthcareEntity,
                 inpatient_visits: HealthcareEntity):

        super().__init__()
        self.outpatient_visits = outpatient_visits
        self.inpatient_visits = inpatient_visits


healthcare_entities = HealthcareEntities(
    outpatient_visits=HealthcareEntity(
        name='outpatient_visits',
        gbd_id=meid(9458),
        proportion=meid(9458),
        cost='Outpatient Visit Costs'
    ),
    inpatient_visits=HealthcareEntity(
        name='inpatient_visits',
        gbd_id=None,
        cost='Inpatient Visit Costs',
    )
)
