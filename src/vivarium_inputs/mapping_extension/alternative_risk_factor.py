from gbd_mapping.base_template import Restrictions
from gbd_mapping.id import reiid
from gbd_mapping import causes

from .alternative_risk_factor_template import AlternativeRiskFactors, AlternativeRiskFactor

alternative_risk_factors = AlternativeRiskFactors(
    child_underweight=AlternativeRiskFactor(
        name='child_underweight',
        kind='alternative_risk_factor',
        gbd_id=reiid(94),
        level=4,
        most_detailed=True,
        distribution='ensemble',
        population_attributable_fraction_calculation_type='categorical',
        exposure_exists=True,
        exposure_standard_deviation_exists=True,
        exposure_year_type='binned',
        relative_risk_exists=True,
        relative_risk_in_range=False,
        population_attributable_fraction_yll_exists=True,
        population_attributable_fraction_yll_in_range=True,
        population_attributable_fraction_yld_exists=True,
        population_attributable_fraction_yld_in_range=True,
        restrictions=Restrictions(
            male_only=False,
            female_only=False,
            yll_only=False,
            yld_only=False,
            yll_age_group_id_start=4,
            yll_age_group_id_end=235,
            yld_age_group_id_start=2,
            yld_age_group_id_end=235,
            violated=('exposure_age_restriction_violated', 'relative_risk_age_restriction_violated',
                      'population_attributable_fraction_yll_age_restriction_violated', )
        ),
        affected_causes=(causes.all_causes, causes.communicable_maternal_neonatal_and_nutritional_diseases,
                         causes.diarrheal_diseases, causes.lower_respiratory_infections, causes.measles,
                         causes.nutritional_deficiencies, causes.protein_energy_malnutrition,
                         causes.respiratory_infections_and_tuberculosis, causes.enteric_infections,
                         causes.other_infectious_diseases, causes.all_causes,
                         causes.communicable_maternal_neonatal_and_nutritional_diseases, causes.diarrheal_diseases,
                         causes.lower_respiratory_infections, causes.measles, causes.nutritional_deficiencies,
                         causes.protein_energy_malnutrition, causes.respiratory_infections_and_tuberculosis,
                         causes.enteric_infections, causes.other_infectious_diseases, ),
        population_attributable_fraction_of_one_causes=(causes.protein_energy_malnutrition, ),
    ),
    child_wasting=AlternativeRiskFactor(
        name='child_wasting',
        kind='alternative_risk_factor',
        gbd_id=reiid(240),
        level=4,
        most_detailed=True,
        distribution='ensemble',
        population_attributable_fraction_calculation_type='categorical',
        exposure_exists=True,
        exposure_standard_deviation_exists=True,
        exposure_year_type='binned',
        relative_risk_exists=True,
        relative_risk_in_range=False,
        population_attributable_fraction_yll_exists=True,
        population_attributable_fraction_yll_in_range=True,
        population_attributable_fraction_yld_exists=True,
        population_attributable_fraction_yld_in_range=True,
        restrictions=Restrictions(
            male_only=False,
            female_only=False,
            yll_only=False,
            yld_only=False,
            yll_age_group_id_start=4,
            yll_age_group_id_end=235,
            yld_age_group_id_start=2,
            yld_age_group_id_end=235,
            violated=('exposure_age_restriction_violated', 'relative_risk_age_restriction_violated',
                      'population_attributable_fraction_yll_age_restriction_violated', )
        ),
        affected_causes=(causes.all_causes, causes.communicable_maternal_neonatal_and_nutritional_diseases,
                         causes.diarrheal_diseases, causes.lower_respiratory_infections, causes.measles,
                         causes.nutritional_deficiencies, causes.protein_energy_malnutrition,
                         causes.respiratory_infections_and_tuberculosis, causes.enteric_infections,
                         causes.other_infectious_diseases, causes.all_causes,
                         causes.communicable_maternal_neonatal_and_nutritional_diseases, causes.diarrheal_diseases,
                         causes.lower_respiratory_infections, causes.measles, causes.nutritional_deficiencies,
                         causes.protein_energy_malnutrition, causes.respiratory_infections_and_tuberculosis,
                         causes.enteric_infections, causes.other_infectious_diseases,),
        population_attributable_fraction_of_one_causes=(causes.protein_energy_malnutrition,),
    ),
    child_stunting=AlternativeRiskFactor(
        name='child_stunting',
        kind='alternative_risk_factor',
        gbd_id=reiid(241),
        level=4,
        most_detailed=True,
        distribution='ensemble',
        population_attributable_fraction_calculation_type='categorical',
        exposure_exists=True,
        exposure_standard_deviation_exists=None,
        exposure_year_type='binned',
        relative_risk_exists=True,
        relative_risk_in_range=False,
        population_attributable_fraction_yll_exists=True,
        population_attributable_fraction_yll_in_range=True,
        population_attributable_fraction_yld_exists=True,
        population_attributable_fraction_yld_in_range=True,
        restrictions=Restrictions(
            male_only=False,
            female_only=False,
            yll_only=False,
            yld_only=False,
            yll_age_group_id_start=4,
            yll_age_group_id_end=5,
            yld_age_group_id_start=4,
            yld_age_group_id_end=5,
            violated=('exposure_age_restriction_violated', )
        ),
        affected_causes=(causes.all_causes, causes.communicable_maternal_neonatal_and_nutritional_diseases,
                         causes.diarrheal_diseases, causes.lower_respiratory_infections, causes.measles,
                         causes.respiratory_infections_and_tuberculosis, causes.enteric_infections,
                         causes.other_infectious_diseases, causes.all_causes,
                         causes.communicable_maternal_neonatal_and_nutritional_diseases, causes.diarrheal_diseases,
                         causes.lower_respiratory_infections, causes.measles,
                         causes.respiratory_infections_and_tuberculosis, causes.enteric_infections,
                         causes.other_infectious_diseases, ),
        population_attributable_fraction_of_one_causes=(),
    ),
)
