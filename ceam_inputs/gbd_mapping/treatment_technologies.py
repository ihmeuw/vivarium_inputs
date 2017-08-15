from .templates import TreatmentTechnology, TreatmentTechnologies, meid

treatment_technologies = TreatmentTechnologies(
    ors=TreatmentTechnology(
        name='ors',
        gbd_id=meid(1321321321),
        # Numbers as per Marcia 07/01/2017
        unit_cost=0.50,
        coverage=0.58,

    )
)
