from typing import Union

from .templates import ModelableEntity, GbdRecord, meid


class TreatmentTechnology(ModelableEntity):
    """Container for treatment technology GBD ids and data."""
    __slots__ = ('name', 'unit_cost', 'daily_cost', 'coverage', 'protection',
                 'efficacy_mean', 'efficacy_sd', 'pafs', 'rrs', 'exposures', )

    def __init__(self,
                 name: str,
                 gbd_id: Union[meid, None],
                 unit_cost: float = None,
                 daily_cost: Union[float, str] = None,
                 coverage: Union[float, str, meid] = None,
                 protection: str = None,
                 efficacy_mean: float = None,
                 efficacy_sd: float = None,
                 pafs: str = None,
                 rrs: str = None,
                 exposures: str = None,):
        super().__init__(name=name, gbd_id=gbd_id)
        self.unit_cost = unit_cost
        self.daily_cost = daily_cost
        self.coverage = coverage
        self.protection = protection
        self.efficacy_mean = efficacy_mean
        self.efficacy_sd = efficacy_sd
        self.pafs = pafs
        self.rrs = rrs
        self.exposures = exposures


class TreatmentTechnologies(GbdRecord):
    """Holder for treatment technology records."""
    __slots__ = ('ors', 'rota_vaccine', 'dtp3_vaccine', 'thiazide_type_diuretics', 'beta_blockers', 'ace_inhibitors',
                 'calcium_channel_blockers', 'hypertension_drugs', )

    def __init__(self,
                 ors: TreatmentTechnology,
                 rota_vaccine: TreatmentTechnology,
                 dtp3_vaccine: TreatmentTechnology,
                 thiazide_type_diuretics: TreatmentTechnology,
                 beta_blockers: TreatmentTechnology,
                 ace_inhibitors: TreatmentTechnology,
                 calcium_channel_blockers: TreatmentTechnology,
                 hypertension_drugs: TreatmentTechnology, ):
        self.ors = ors
        self.rota_vaccine = rota_vaccine
        self.dtp3_vaccine = dtp3_vaccine
        self.thiazide_type_diuretics = thiazide_type_diuretics
        self.beta_blockers = beta_blockers
        self.ace_inhibitors = ace_inhibitors
        self.calcium_channel_blockers = calcium_channel_blockers
        self.hypertension_drugs = hypertension_drugs



treatment_technologies = TreatmentTechnologies(
    ors=TreatmentTechnology(
        name='ors',
        gbd_id=None,
        daily_cost='ORS Costs',
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
        coverage='DTP3 Coverage',
        unit_cost=0,
    ),
    thiazide_type_diuretics=TreatmentTechnology(
        name='thiazide_type_diuretics',
        gbd_id=None,
        daily_cost=0.009,
        efficacy_mean=8.8,
        efficacy_sd=0.281,
    ),
    beta_blockers=TreatmentTechnology(
        name='beta_blockers',
        gbd_id=None,
        daily_cost=0.048,
        efficacy_mean=9.2,
        efficacy_sd=0.332,
    ),
    ace_inhibitors=TreatmentTechnology(
        name='ace_inhibitors',
        gbd_id=None,
        daily_cost=0.059,
        efficacy_mean=10.3,
        efficacy_sd=0.281,
    ),
    calcium_channel_blockers=TreatmentTechnology(
        name='calcium_channel_blockers',
        gbd_id=None,
        daily_cost=0.166,
        efficacy_mean=8.8,
        efficacy_sd=0.23,
    ),
    hypertension_drugs=TreatmentTechnology(
        name='hypertenstion_drugs',
        gbd_id=None,
        daily_cost='Hypertension Drug Costs',
    )
)
