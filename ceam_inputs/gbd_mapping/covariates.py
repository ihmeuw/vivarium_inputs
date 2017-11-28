from typing import Union
from .templates import GbdRecord, ModelableEntity, covid


class Covariate(ModelableEntity):
    """Container for covariate GBD ids and data."""
    __slots__ = ('name', 'gbd_id')

    def __init__(self,
                 name: str,
                 gbd_id: Union[covid, None],):
        super().__init__(name=name, gbd_id=gbd_id)


class Covariates(GbdRecord):
    """Holder of covariate modelable entities"""
    __slots__ = ('age_specific_fertility_rate', 'dtp3_coverage_proportion', 'live_births_by_sex')

    def __init__(self,
                 age_specific_fertility_rate: Covariate,
                 dtp3_coverage_proportion: Covariate,
                 live_births_by_sex: Covariate,):
        super().__init__()
        self.age_specific_fertility_rate = age_specific_fertility_rate
        self.dtp3_coverage_proportion = dtp3_coverage_proportion
        self.live_births_by_sex = live_births_by_sex


covariates = Covariates(
    age_specific_fertility_rate=Covariate(
        name='age_specific_fertility_rate',
        gbd_id=covid(13),
    ),
    dtp3_coverage_proportion=Covariate(
        name='dtp3_coverage_proportion',
        gbd_id=covid(32),
    ),
    live_births_by_sex=Covariate(
        name='live_births_by_sex',
        gbd_id=covid(1106),
    )
)
