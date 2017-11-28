from typing import Union

from .templates import ModelableEntity, GbdRecord, meid


class TreatmentTechnology(ModelableEntity):
    """Container for treatment technology GBD ids and data."""
    __slots__ = ('name', 'unit_cost', 'cost', 'coverage', 'protection',
                 'efficacy_mean', 'efficacy_sd', 'pafs', 'rrs', 'exposures', )

    def __init__(self,
                 name: str,
                 gbd_id: Union[meid, None],
                 unit_cost: float = None,
                 cost: str = None,
                 coverage: Union[float, str, meid] = None,
                 protection: str = None,
                 efficacy_mean: float = None,
                 efficacy_sd: float = None,
                 pafs: str = None,
                 rrs: str = None,
                 exposures: str = None,):
        super().__init__(name=name, gbd_id=gbd_id)
        self.unit_cost = unit_cost
        self.cost = cost
        self.coverage = coverage
        self.protection = protection
        self.efficacy_mean = efficacy_mean
        self.efficacy_sd = efficacy_sd
        self.pafs = pafs
        self.rrs = rrs
        self.exposures = exposures


class TreatmentTechnologies(GbdRecord):
    """Holder for treatment technology records."""
    __slots__ = ('ors', 'rota_vaccine', 'dtp3_vaccine', 'hypertension_drugs', )

    def __init__(self,
                 ors: TreatmentTechnology,
                 rota_vaccine: TreatmentTechnology,
                 dtp3_vaccine: TreatmentTechnology,
                 hypertension_drugs: TreatmentTechnology, ):
        self.ors = ors
        self.rota_vaccine = rota_vaccine
        self.dtp3_vaccine = dtp3_vaccine
        self.hypertension_drugs = hypertension_drugs


treatment_technologies = TreatmentTechnologies(
    ors=TreatmentTechnology(
        name='ors',
        gbd_id=None,
        cost='ORS Costs',
        unit_cost=0.50,  # As per Marcia 07/01/2017
        coverage=0.58,  # As per Marcia 07/01/2017
        pafs='Ors Pafs',
        rrs='Ors Relative Risks',
        exposures='Ors Exposure',
    ),
    rota_vaccine=TreatmentTechnology(
        name='rota_vaccine',
        gbd_id=None,
        coverage=meid(10596),
        protection='Rota Vaccine Protection',
        rrs='Rota Vaccine RRs',
    ),
    dtp3_vaccine=TreatmentTechnology(
        name='dtp3_vaccine',
        gbd_id=None,
        unit_cost=0,
    ),
    hypertension_drugs=TreatmentTechnology(
        name='hypertension_drugs',
        gbd_id=None,
        cost='Hypertension Drug Costs',
    )
)
