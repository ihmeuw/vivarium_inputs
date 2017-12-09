from typing import Union

from .templates import ModelableEntity, GbdRecord, meid


class TreatmentTechnology(ModelableEntity):
    """Container for treatment technology GBD ids and data."""
    __slots__ = ('name', 'unit_cost', 'cost', 'coverage', 'protection',
                 'efficacy_mean', 'efficacy_sd', 'population_attributable_fraction', 'relative_risk', 'exposure', )

    def __init__(self,
                 name: str,
                 gbd_id: Union[meid, None],
                 unit_cost: float = None,
                 cost: str = None,
                 coverage: Union[float, str, meid] = None,
                 protection: str = None,
                 efficacy_mean: float = None,
                 efficacy_sd: float = None,
                 population_attributable_fraction: str = None,
                 relative_risk: str = None,
                 exposure: str = None,):
        super().__init__(name=name, gbd_id=gbd_id)
        self.unit_cost = unit_cost
        self.cost = cost
        self.coverage = coverage
        self.protection = protection
        self.efficacy_mean = efficacy_mean
        self.efficacy_sd = efficacy_sd
        self.population_attributable_fraction = population_attributable_fraction
        self.relative_risk = relative_risk
        self.exposure = exposure


class TreatmentTechnologies(GbdRecord):
    """Holder for treatment technology records."""
    __slots__ = ('ors', 'rota_vaccine', 'hypertension_drugs', 'hiv_positive_antiretroviral_treatment')

    def __init__(self,
                 ors: TreatmentTechnology,
                 rota_vaccine: TreatmentTechnology,
                 hypertension_drugs: TreatmentTechnology,
                 hiv_positive_antiretroviral_treatment: TreatmentTechnology):
        self.ors = ors
        self.rota_vaccine = rota_vaccine
        self.hypertension_drugs = hypertension_drugs
        self.hiv_positive_antiretroviral_treatment = hiv_positive_antiretroviral_treatment


treatment_technologies = TreatmentTechnologies(
    ors=TreatmentTechnology(
        name='ors',
        gbd_id=None,
        cost='ORS Costs',
        unit_cost=0.50,  # As per Marcia 07/01/2017
        coverage=0.58,  # As per Marcia 07/01/2017
        population_attributable_fraction='Ors Pafs',
        relative_risk='Ors Relative Risks',
    ),
    rota_vaccine=TreatmentTechnology(
        name='rota_vaccine',
        gbd_id=None,
        coverage=meid(10596),
        protection='Rota Vaccine Protection',
    ),
    hypertension_drugs=TreatmentTechnology(
        name='hypertension_drugs',
        gbd_id=None,
        cost='Hypertension Drug Costs',
    ),
    hiv_positive_antiretroviral_treatment=TreatmentTechnology(
        name='hiv_positive_antiretroviral_treatment',
        gbd_id=None,
        exposure='HIV Positive Antiretroviral Therapy Exposure',
    )
)
