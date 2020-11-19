from typing import Union

from gbd_mapping.base_template import ModelableEntity, GbdRecord
from gbd_mapping.id import me_id


class HealthcareEntity(ModelableEntity):
    """Container for healthcare system GBD ids and data."""
    __slots__ = ('name', 'kind', 'gbd_id', 'utilization')

    def __init__(self,
                 name: str,
                 kind: str,
                 gbd_id: Union[me_id, None],
                 utilization: me_id = None,
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
