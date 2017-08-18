from .templates import TreatmentTechnology, TreatmentTechnologies

treatment_technologies = TreatmentTechnologies(
    ors=TreatmentTechnology(
        name='ors',
        gbd_id=None,
        # Numbers as per Marcia 07/01/2017
        unit_cost=0.50,
        coverage=0.58,
    )
)
