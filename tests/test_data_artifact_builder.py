from ceam_inputs.data_artifact import ArtifactBuilder

def test_happy_path():
    builder = ArtifactBuilder({})
    for cause in ['ischemic_heart_disease', 'ischemic_stroke', 'diarrheal_diseases']:
        builder.data_container(f'cause.{cause}')
    for risk in ['high_systolic_blood_pressure', 'unsafe_water_source']:
        builder.data_container(f'risk_factor.{risk}')
    for sequela in ['acute_typhoid_infection', 'mild_upper_respiratory_infections']:
        builder.data_container(f'sequela.{sequela}')
    builder.data_container('risk_factor.correlations')
    builder.data_container('population')
    builder.data_container('healthcare_entity.outpatient_visits')
    for t in ['hypertension_drugs', 'rota_vaccine', 'hiv_positive_antiretroviral_treatment']:
        builder.data_container(f'treatment_technology.{t}')
    for t in ['low_measles_vaccine_coverage_first_dose']:
        builder.data_container(f'coverage_gap.{t}')
    for t in ['shigellosis',]:
        builder.data_container(f'etiology.{t}')
    for t in ['age_specific_fertility_rate', 'live_births_by_sex', 'dtp3_coverage_proportion']:
        builder.data_container(f'covariate.{t}')
    builder.data_container(f'subregions')
    builder.save('/tmp/test_artifact.tgz', [180], year_range=(1995, 2015), parallelism=4)
