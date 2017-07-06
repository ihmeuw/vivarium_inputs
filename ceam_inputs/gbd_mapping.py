from ceam.config_tree import ConfigTree


class meid(int):
    """Modelable Entity ID"""
    def __repr__(self):
        return 'meid({:d})'.format(self)


class rid(int):
    """Risk Factor ID"""
    def __repr__(self):
        return 'rid({:d})'.format(self)


class cid(int):
    """Cause ID"""
    def __repr__(self):
        return 'cid({:d})'.format(self)


class hid(int):
    """Healthstate ID"""
    def __repr__(self):
        return 'hid({:d})'.format(self)

raw_cause_mapping = {
    'all_causes': {
        'gbd_cause': cid(294),
    },
    'diarrhea': {
        'gbd_cause': cid(302),
        'disability_weight': 0.23,
        'excess_mortality': meid(1181),
        'prevalence': meid(1181),
        'incidence': meid(1181),
        'mortality': cid(302),
        'duration': meid(1181),
    },
    'mild_diarrhea': {
        'gbd_cause': cid(302),
        'incidence': meid(2608),
        'disability_weight': hid(355),
        'duration': meid(1181)
    },
    'moderate_diarrhea': {
        'gbd_cause': cid(302),
        'incidence': meid(2609),
        'disability_weight': hid(356),
        'duration': meid(1181)
    },
    'severe_diarrhea': {
        'gbd_cause': cid(302),
        'incidence': meid(2610),
        'disability_weight': hid(357),
        'mortality': cid(302),
        'excess_mortality': meid(1181),
        'duration': meid(1181)
    },
    'unattributed_diarrhea': {
        'gbd_parent_cause': cid(302),
    },
    'cholera': {
        'gbd_parent_cause': cid(302),
        'gbd_cause': rid(173),
    },
    'other_salmonella': {
        'gbd_parent_cause': cid(302),
        'gbd_cause': rid(174),
    },
    'shigellosis': {
        'gbd_parent_cause': cid(302),
        'gbd_cause': rid(175),
    },
    'EPEC': {
        'gbd_parent_cause': cid(302),
        'gbd_cause': rid(176),
    },
    'ETEC': {
        'gbd_parent_cause': cid(302),
        'gbd_cause': rid(177),
    },
    'campylobacter': {
        'gbd_parent_cause': cid(302),
        'gbd_cause': rid(178),
    },
    'amoebiasis': {
        'gbd_parent_cause': cid(302),
        'gbd_cause': rid(179),
    },
    'cryptosporidiosis': {
        'gbd_parent_cause': cid(302),
        'gbd_cause': rid(180),
    },
    'rotaviral_entiritis': {
        'gbd_parent_cause': cid(302),
        'gbd_cause': rid(181),
    },
    'aeromonas': {
        'gbd_parent_cause': cid(302),
        'gbd_cause': rid(182),
    },
    'clostridium_difficile': {
        'gbd_parent_cause': cid(302),
        'gbd_cause': rid(183),
    },
    'norovirus': {
        'gbd_parent_cause': cid(302),
        'gbd_cause': rid(184),
    },
    'adenovirus': {
        'gbd_parent_cause': cid(302),
        'gbd_cause': rid(185),
    },
    'rheumatic_heart_disease': {
        'gbd_cause': cid(492),
    },
    'ischemic_heart_disease': {
        'gbd_cause': cid(493),
        'mortality': cid(493),
    },
    'heart_attack': {
        'gbd_cause': cid(493),
        'excess_mortality': meid(1814),
        'prevalence': meid(1814),
        'mortality': meid(1814),
        'incidence': meid(1814),
    },
    'mild_heart_failure': {
        'gbd_cause': cid(493),
        'disability_weight': meid(1821),
        'excess_mortality': meid(2412),
        'prevalence': meid(1821),
        'mortality': cid(493),
    },
    'moderate_heart_failure': {
        'gbd_cause': cid(493),
        'disability_weight': meid(1822),
        'excess_mortality': meid(2412),
        'prevalence': meid(1822),
        'mortality': cid(493),
    },
    'severe_heart_failure': {
        'gbd_cause': cid(493),
        'disability_weight': meid(1823),
        'excess_mortality': meid(2412),
        'prevalence': meid(1823),
        'mortality': cid(493),
    },
    'angina_not_due_to_MI': {
        'gbd_cause': cid(493),
        'incidence': meid(1817),
    },
    'asymptomatic_angina': {
        'gbd_cause': cid(493),
        'disability_weight': meid(1823),
        'excess_mortality': meid(1817),
        'prevalence': meid(3102),
        'mortality': cid(493),
    },
    'mild_angina': {
        'gbd_cause': cid(493),
        'disability_weight': meid(1818),
        'excess_mortality': meid(1817),
        'prevalence': meid(1818),
    },
    'moderate_angina': {
        'gbd_cause': cid(493),
        'disability_weight': meid(1819),
        'excess_mortality': meid(1817),
        'prevalence': meid(1819),
        'mortality': cid(493),
    },
    'severe_angina': {
        'gbd_cause': cid(493),
        'disability_weight': meid(1820),
        'excess_mortality': meid(1817),
        'prevalence': meid(1820),
        'mortality': cid(493),
    },
    'asymptomatic_ihd': {
        'gbd_cause': cid(493),
        'disability_weight': meid(3233),
        'excess_mortality': 0.0,
        'prevalence': meid(3233),
        'mortality': cid(493),
    },
    'chronic_stroke': {
        'gbd_cause': cid(494),
        'disability_weight': 0.32,
        'excess_mortality': meid(9312),
        'prevalence': meid(9312),
        'mortality': cid(494),
    },
    'ischemic_stroke': {
        'gbd_cause': cid(495),
        'disability_weight': 0.32,
        'excess_mortality': meid(9310),
        'prevalence': meid(9310),
        'mortality': cid(495),
        'incidence': meid(9310),
    },
    'hemorrhagic_stroke': {
        'gbd_cause': cid(496),
        'disability_weight': 0.32,
        'excess_mortality': meid(9311),
        'prevalence': meid(9311),
        'mortality': cid(496),
        'incidence': meid(9311),
    },
    'hypertensive_heart_disease': {
        'gbd_cause': cid(498),
    },
    'cardiomyopathy_and_myocarditis': {
        'gbd_cause': cid(499),
    },
    'atrial_fibrillation_and_flutter': {
        'gbd_cause': cid(500),
    },
    'aortic_aneurysm': {
        'gbd_cause': cid(501),
    },
    'peripheral_vascular_disease': {
        'gbd_cause': cid(502),
    },
    'endocarditis': {
        'gbd_cause': cid(503),
    },
    'other_cardiovascular_and_circulatory_diseases': {
        'gbd_cause': cid(507),
    },
    'chronic_kidney_disease_due_to_diabetes_mellitus': {
        'gbd_cause': cid(590),
    },
    'chronic_kidney_disease_due_to_hypertension': {
        'gbd_cause': cid(591),
    },
    'chronic_kidney_disease_due_to_glomerulonephritis': {
        'gbd_cause': cid(592),
    },
    'chronic_kidney_disease_due_to_other_causes': {
        'gbd_cause': cid(593),
    },
}

for k, v in raw_cause_mapping.items():
    v['name'] = k

causes = ConfigTree(raw_cause_mapping)
causes.freeze()

raw_risk_mapping = {
    'unsafe_water_source': {
        'gbd_risk': rid(83),
        'risk_type': 'categorical',
        'effected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC, causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis, causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile, causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'unsafe_sanitation': {
        'gbd_risk': rid(84),
        'risk_type': 'categorical',
        'effected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC, causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis, causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile, causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'ambient_particulate_matter_pollution': {
        'gbd_risk': rid(86),
        'risk_type': 'continuous',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke],
    },
    'household_air_pollution_from_solid_fuels': {
        'gbd_risk': rid(87),
        'risk_type': 'categorical',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke],
    },
    'vitamin_a_deficiency': {
        'gbd_risk': rid(96),
        'risk_type': 'categorical',
        'effected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC, causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis, causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile, causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'zinc_deficiency': {
        'gbd_risk': rid(97),
        'risk_type': 'categorical',
        'effected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC, causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis, causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile, causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'secondhand_smoke': {
        'gbd_risk': rid(100),
        'risk_type': 'categorical',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke],
    },
    'alcohol_use': {
        'gbd_risk': rid(102),
        'risk_type': 'continuous',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke,
                            causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                            causes.atrial_fibrillation_and_flutter],
    },
    'high_total_cholesterol': {
        'gbd_risk': rid(106),
        'risk_type': 'continuous',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke],
        'tmrl': 3.08,
        'scale': 1
    },
    'high_systolic_blood_pressure': {
        'gbd_risk': rid(107),
        'risk_type': 'continuous',
        'effected_causes': [causes.rheumatic_heart_disease, causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                            causes.cardiomyopathy_and_myocarditis, causes.atrial_fibrillation_and_flutter,
                            causes.aortic_aneurysm, causes.peripheral_vascular_disease, causes.endocarditis,
                            causes.other_cardiovascular_and_circulatory_diseases,
                            causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                            causes.chronic_kidney_disease_due_to_hypertension,
                            causes.chronic_kidney_disease_due_to_glomerulonephritis,
                            causes.chronic_kidney_disease_due_to_other_causes],
        'tmrl': 112.5,
        'scale': 10,
    },
    'high_body_mass_index': {
        'gbd_risk': rid(108),
        'risk_type': 'continuous',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke,
                            causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                            causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                            causes.chronic_kidney_disease_due_to_hypertension,
                            causes.chronic_kidney_disease_due_to_glomerulonephritis,
                            causes.chronic_kidney_disease_due_to_other_causes],
        'tmrl': 21,
        'scale': 5,
    },
    'diet_low_in_fruits': {
        'gbd_risk': rid(111),
        'risk_type': 'continuous',
        'effected_causes': [causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke],
    },
    'diet_low_in_vegetables': {
        'gbd_risk': rid(112),
        'risk_type': 'continuous',
        'effected_causes': [causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke],
    },
    'diet_low_in_whole_grains': {
        'gbd_risk': rid(113),
        'risk_type': 'continuous',
        'effected_causes': [causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke],
    },
    'diet_low_in_nuts_and_seeds': {
        'gbd_risk': rid(114),
        'risk_type': 'continuous',
        'effected_causes': [causes.ischemic_heart_disease],
    },
    'diet_high_in_processed_meat': {
        'gbd_risk': rid(117),
        'risk_type': 'continuous',
        'effected_causes': [causes.ischemic_heart_disease],
    },
    'diet_high_in_sugar_sweetened_beverages': {
        'gbd_risk': rid(118),
        'risk_type': 'continuous',
        'effected_causes': [causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                            causes.hypertensive_heart_disease, causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                            causes.chronic_kidney_disease_due_to_hypertension,
                            causes.chronic_kidney_disease_due_to_glomerulonephritis,
                            causes.chronic_kidney_disease_due_to_other_causes],
    },
    'diet_low_in_fiber': {
        'gbd_risk': rid(119),
        'risk_type': 'continuous',
        'effected_causes': [causes.ischemic_heart_disease],
    },
    'diet_low_in_seafood_omega_3_fatty_acids': {
        'gbd_risk': rid(121),
        'risk_type': 'continuous',
        'effected_causes': [causes.ischemic_heart_disease],
    },
    'diet_low_in_polyunsaturated_fatty_acids': {
        'gbd_risk': rid(122),
        'risk_type': 'continuous',
        'effected_causes': [causes.ischemic_heart_disease],
    },
    'diet_high_in_trans_fatty_acids': {
        'gbd_risk': rid(123),
        'risk_type': 'continuous',
        'effected_causes': [causes.ischemic_heart_disease],
    },
    'diet_high_in_sodium': {
        'gbd_risk': rid(124),
        'risk_type': 'continuous',
        'effected_causes': [causes.rheumatic_heart_disease, causes.ischemic_heart_disease, causes.ischemic_stroke,
                            causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                            causes.cardiomyopathy_and_myocarditis, causes.atrial_fibrillation_and_flutter,
                            causes.aortic_aneurysm, causes.peripheral_vascular_disease, causes.endocarditis,
                            causes.other_cardiovascular_and_circulatory_diseases,
                            causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                            causes.chronic_kidney_disease_due_to_hypertension,
                            causes.chronic_kidney_disease_due_to_glomerulonephritis,
                            causes.chronic_kidney_disease_due_to_other_causes],
    },
    # 'low_physical_activity': {
    #     'gbd_risk': rid(125),
    #     'risk_type': 'continuous',
    #     'effected_causes': [causes.ischemic_heart_disease, causes.ischemic_stroke],
    # },
    'non_exclusive_breastfeeding': {
        'gbd_risk': rid(136),
        'risk_type': 'categorical',
        'effected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC, causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis, causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile, causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'discontinued_breastfeeding': {
        'gbd_risk': rid(137),
        'risk_type': 'categorical',
        'effected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC, causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis, causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile, causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'high_fasting_plasma_glucose_continuous': {
        'gbd_risk': rid(141),
        'risk_type': 'continuous',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke,
                            causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                            causes.chronic_kidney_disease_due_to_hypertension,
                            causes.chronic_kidney_disease_due_to_glomerulonephritis,
                            causes.chronic_kidney_disease_due_to_other_causes],
        'tmrl': 5.1,
        'scale': 1,
    },
    'high_fasting_plasma_glucose_categorical': {
        'gbd_risk': rid(142),
        'risk_type': 'categorical',
        'effected_causes': [causes.peripheral_vascular_disease],
    },
    # 'low_glomerular_filtration_rate': {
    #     'gbd_risk': rid(143),
    #     'effected_causes': [causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
    #                         causes.peripheral_vascular_disease, causes.chronic_kidney_disease_due_to_diabetes_mellitus,
    #                         causes.chronic_kidney_disease_due_to_hypertension,
    #                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
    #                         causes.chronic_kidney_disease_due_to_other_causes],
    # },
    'smoking_prevalence_approach': {
        'gbd_risk': rid(166),
        'risk_type': 'categorical',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke,
                            causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                            causes.atrial_fibrillation_and_flutter, causes.aortic_aneurysm,
                            causes.peripheral_vascular_disease, causes.other_cardiovascular_and_circulatory_diseases],
    },
    'no_access_to_handwashing_facility': {
        'gbd_risk': rid(238),
        'risk_type': 'categorical',
        'effected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC, causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis, causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile, causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'child_wasting': {
        'gbd_risk': rid(240),
        'risk_type': 'categorical',
        'effected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC, causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis, causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile, causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'child_stunting': {
        'gbd_risk': rid(241),
        'risk_type': 'categorical',
        'effected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC, causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis, causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile, causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'lead_exposure_in_bone': {
        'gbd_risk': rid(243),
        'risk_type': 'continuous',
        'effected_causes': [causes.rheumatic_heart_disease, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                            causes.cardiomyopathy_and_myocarditis, causes.atrial_fibrillation_and_flutter,
                            causes.aortic_aneurysm, causes.peripheral_vascular_disease, causes.endocarditis,
                            causes.other_cardiovascular_and_circulatory_diseases,
                            causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                            causes.chronic_kidney_disease_due_to_hypertension,
                            causes.chronic_kidney_disease_due_to_glomerulonephritis,
                            causes.chronic_kidney_disease_due_to_other_causes],
    },
    'low_measles_vaccine_coverage_1st_dose': {
        'gbd_risk': rid(318),
        'risk_type': 'categorical',
        'effected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC, causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis, causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile, causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
}

for k, v in raw_risk_mapping.items():
    v['name'] = k

risk_factors = ConfigTree(raw_risk_mapping)
risk_factors.freeze()
