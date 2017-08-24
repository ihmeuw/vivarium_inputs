from .templates import TreatmentTechnology, TreatmentTechnologies, meid

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
