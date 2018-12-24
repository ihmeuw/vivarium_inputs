import pytest
from vivarium_inputs.data_artifact import aggregate, disaggregate



@pytest.fixture()
def keys_mock():
    keys = ['metadata.locations', 'metadata.keyspace', 'metadata.versions',
            'cause.all_causes.cause_specific_mortality', 'cause.all_causes.restrictions',
            'cause.diarrheal_diseases.cause_specific_mortality', 'cause.diarrheal_diseases.death',
            'cause.diarrheal_diseases.etiologies', 'cause.diarrheal_diseases.excess_mortality',
            'cause.diarrheal_diseases.incidence', 'cause.diarrheal_diseases.population_attributable_fraction',
            'cause.diarrheal_diseases.prevalence', 'cause.diarrheal_diseases.remission',
            'cause.diarrheal_diseases.restrictions', 'cause.diarrheal_diseases.sequelae',
            'covariate.dtp3_coverage_proportion.estimate', 'dimensions.full_space',
            'etiology.adenovirus.population_attributable_fraction',
            'etiology.aeromonas.population_attributable_fraction',
            'etiology.amoebiasis.population_attributable_fraction', 'population.age_bins', 'population.structure',
            'population.theoretical_minimum_risk_life_expectancy', 'risk_factor.child_stunting.affected_causes',
            'risk_factor.child_stunting.distribution', 'risk_factor.child_stunting.exposure',
            'risk_factor.child_stunting.levels', 'risk_factor.child_stunting.relative_risk',
            'risk_factor.child_stunting.restrictions', 'sequela.moderate_diarrheal_diseases.disability_weight',
            'sequela.moderate_diarrheal_diseases.healthstate', 'sequela.moderate_diarrheal_diseases.incidence',
            'sequela.moderate_diarrheal_diseases.prevalence', 'no_data.key']
    return keys