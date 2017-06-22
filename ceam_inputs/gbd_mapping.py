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
        'distribution': 'categorical',
        'effected_causes': [causes.diarrhea],
    },
    'unsafe_sanitation': {
        'gbd_risk': rid(84),
        'distribution': 'categorical',
        'effected_causes': [causes.diarrhea],
    },
    'ambient_particulate_matter_pollution': {
        'gbd_risk': rid(86),
        'distribution': 'lognormal',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke],
        'tmred': {
            'distribution': 'uniform',
            'min': 2.4,
            'max': 5.9,
            'inverted': False,
        },
        'scale': 1,
    },
    'household_air_pollution_from_solid_fuels': {
        'gbd_risk': rid(87),
        'distribution': 'categorical',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke],
    },
    'vitamin_a_deficiency': {
        'gbd_risk': rid(96),
        'distribution': 'categorical',
        'effected_causes': [causes.diarrhea],
    },
    'zinc_deficiency': {
        'gbd_risk': rid(97),
        'distribution': 'categorical',
        'effected_causes': [causes.diarrhea],
    },
    'secondhand_smoke': {
        'gbd_risk': rid(100),
        'distribution': 'categorical',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke],
    },
    # 'alcohol_use': {
    #     'gbd_risk': rid(102),
    #     'risk_type': 'continuous',
    #     'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke,
    #                         causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
    #                         causes.atrial_fibrillation_and_flutter],
    # },
    'high_total_cholesterol': {
        'gbd_risk': rid(106),
        'distribution': 'lognormal',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke],
        'tmred': {
            'distribution': 'uniform',
            'min': 2.78,
            'max': 3.38,
            'inverted': False,
        },
        'scale': 1,
        'max_rr': 10,
    },
    'high_systolic_blood_pressure': {
        'gbd_risk': rid(107),
        'distribution': 'lognormal',
        'effected_causes': [causes.rheumatic_heart_disease, causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                            causes.cardiomyopathy_and_myocarditis, causes.atrial_fibrillation_and_flutter,
                            causes.aortic_aneurysm, causes.peripheral_vascular_disease, causes.endocarditis,
                            causes.other_cardiovascular_and_circulatory_diseases,
                            causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                            causes.chronic_kidney_disease_due_to_hypertension,
                            causes.chronic_kidney_disease_due_to_glomerulonephritis,
                            causes.chronic_kidney_disease_due_to_other_causes],
        'tmred': {
            'distribution': 'uniform',
            'min': 110,
            'max': 115,
            'inverted': False,
        },
        'scale': 10,
        'max_rr': 200,
    },
    'high_body_mass_index': {
        'gbd_risk': rid(108),
        'distribution': 'beta',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke,
                            causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                            causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                            causes.chronic_kidney_disease_due_to_hypertension,
                            causes.chronic_kidney_disease_due_to_glomerulonephritis,
                            causes.chronic_kidney_disease_due_to_other_causes],
        'tmred': {
            'distribution': 'uniform',
            'min': 20,
            'max': 25,
            'inverted': False,
        },
        'scale': 5,
        'max_rr': 50,
    },
    'diet_low_in_fruits': {
        'gbd_risk': rid(111),
        'distribution': 'lognormal',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke],
        'tmred': {
            'distribution': 'uniform',
            'min': 200,
            'max': 300,
            'inverted': True,
        },
        'scale': 100,
        'max_rr': 400,
    },
    'diet_low_in_vegetables': {
        'gbd_risk': rid(112),
        'distribution': 'lognormal',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke],
        'tmred': {
            'distribution': 'uniform',
            'min': 290,
            'max': 430,
            'inverted': True,
        },
        'scale': 100,
        'max_rr': 450,
    },
    'diet_low_in_whole_grains': {
        'gbd_risk': rid(113),
        'distribution': 'lognormal',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke],
        'tmred': {
            'distribution': 'uniform',
            'min': 100,
            'max': 150,
            'inverted': True,
        },
        'scale': 50,
        'max_rr': 150,
    },
    'diet_low_in_nuts_and_seeds': {
        'gbd_risk': rid(114),
        'distribution': 'lognormal',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease],
        'tmred': {
            'distribution': 'uniform',
            'min': 16,
            'max': 25,
            'inverted': True,
        },
        'scale': 4.05,
        'max_rr': 24,
    },
    'diet_high_in_processed_meat': {
        'gbd_risk': rid(117),
        'distribution': 'lognormal',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease],
        'tmred': {
            'distribution': 'uniform',
            'min': 0,
            'max': 4,
            'inverted': False,
        },
        'scale': 50,
        'max_rr': 1000,
    },
    'diet_high_in_sugar_sweetened_beverages': {
        'gbd_risk': rid(118),
        'distribution': 'lognormal',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke,
                            causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                            causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                            causes.chronic_kidney_disease_due_to_hypertension,
                            causes.chronic_kidney_disease_due_to_glomerulonephritis,
                            causes.chronic_kidney_disease_due_to_other_causes],
        'tmred': {
            'distribution': 'uniform',
            'min': 0,
            'max': 5,
            'inverted': False,
        },
        'scale': 226.8,
        'max_rr': 5000,
    },
    'diet_low_in_fiber': {
        'gbd_risk': rid(119),
        'distribution': 'lognormal',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease],
        'tmred': {
            'distribution': 'uniform',
            'min': 19,
            'max': 28,
            'inverted': True,
        },
        'scale': 20,
        'max_rr': 32,
    },
    'diet_low_in_seafood_omega_3_fatty_acids': {
        'gbd_risk': rid(121),
        'distribution': 'lognormal',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease],
        'tmred': {
            'distribution': 'uniform',
            'min': 200,
            'max': 300,
            'inverted': True,
        },
        'scale': 100,
        'max_rr': 1000,
    },
    'diet_low_in_polyunsaturated_fatty_acids': {
        'gbd_risk': rid(122),
        'distribution': 'lognormal',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease],
        'tmred': {
            'distribution': 'uniform',
            'min': 0.09,
            'max': 0.13,
            'inverted': True,
        },
        'scale': 0.05,
        'max_rr': 0.149,
    },
    'diet_high_in_trans_fatty_acids': {
        'gbd_risk': rid(123),
        'distribution': 'lognormal',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease],
        'tmred': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.01,
            'inverted': False,
        },
        'scale': 0.02,
        'max_rr': 1,
    },
    'diet_high_in_sodium': {
        'gbd_risk': rid(124),
        'distribution': 'lognormal',
        'effected_causes': [causes.rheumatic_heart_disease, causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                            causes.cardiomyopathy_and_myocarditis, causes.atrial_fibrillation_and_flutter,
                            causes.aortic_aneurysm, causes.peripheral_vascular_disease, causes.endocarditis,
                            causes.other_cardiovascular_and_circulatory_diseases,
                            causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                            causes.chronic_kidney_disease_due_to_hypertension,
                            causes.chronic_kidney_disease_due_to_glomerulonephritis,
                            causes.chronic_kidney_disease_due_to_other_causes],
        'tmred': {
            'distribution': 'uniform',
            'min': 1,
            'max': 5,
            'inverted': False,
        },
        'scale': 1,
        'max_rr': 50,
    },
    # 'low_physical_activity': {
    #     'gbd_risk': rid(125),
    #     'risk_type': 'continuous',
    #     'distribution': 'weibull',
    #     'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke],
    #     'tmred': {
    #         'distribution': 'uniform',
    #         'min': 3000,
    #         'max': 4500,
    #         'inverted': True,
    #     },
    #     'scale': 600,
    #     'max_rr': 5000,
    # },
    'non_exclusive_breastfeeding': {
        'gbd_risk': rid(136),
        'distribution': 'categorical',
        'effected_causes': [causes.diarrhea],
    },
    'discontinued_breastfeeding': {
        'gbd_risk': rid(137),
        'distribution': 'categorical',
        'effected_causes': [causes.diarrhea],
    },
    'high_fasting_plasma_glucose_continuous': {
        'gbd_risk': rid(141),
        'distribution': 'lognormal',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke,
                            causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                            causes.chronic_kidney_disease_due_to_hypertension,
                            causes.chronic_kidney_disease_due_to_glomerulonephritis,
                            causes.chronic_kidney_disease_due_to_other_causes],
        'tmred': {
            'distribution': 'uniform',
            'min': 4.88488,
            'max': 5.301205,
            'inverted': False,
        },
        'scale': 1,
        'max_rr': 30,
    },
    'high_fasting_plasma_glucose_categorical': {
        'gbd_risk': rid(142),
        'distribution': 'categorical',
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
        'distribution': 'categorical',
        'effected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke,
                            causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                            causes.atrial_fibrillation_and_flutter, causes.aortic_aneurysm,
                            causes.peripheral_vascular_disease, causes.other_cardiovascular_and_circulatory_diseases],
    },
    'no_access_to_handwashing_facility': {
        'gbd_risk': rid(238),
        'distribution': 'categorical',
        'effected_causes': [causes.diarrhea],
    },
    'child_wasting': {
        'gbd_risk': rid(240),
        'distribution': 'categorical',
        'effected_causes': [causes.diarrhea],
    },
    'child_stunting': {
        'gbd_risk': rid(241),
        'distribution': 'categorical',
        'effected_causes': [causes.diarrhea],
    },
    'lead_exposure_in_bone': {
        'gbd_risk': rid(243),
        'distribution': 'lognormal',
        'effected_causes': [causes.rheumatic_heart_disease, causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                            causes.cardiomyopathy_and_myocarditis, causes.atrial_fibrillation_and_flutter,
                            causes.aortic_aneurysm, causes.peripheral_vascular_disease, causes.endocarditis,
                            causes.other_cardiovascular_and_circulatory_diseases,
                            causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                            causes.chronic_kidney_disease_due_to_hypertension,
                            causes.chronic_kidney_disease_due_to_glomerulonephritis,
                            causes.chronic_kidney_disease_due_to_other_causes],
        'tmred': {
            'distribution': 'uniform',
            'min': 0,
            'max': 20,
            'inverted': False,
        },
        'scale': 10,
        'max_rr': 500,
    },
    'low_measles_vaccine_coverage_1st_dose': {
        'gbd_risk': rid(318),
        'distribution': 'categorical',
        'effected_causes': [causes.diarrhea],
    },
}

for k, v in raw_risk_mapping.items():
    v['name'] = k

risk_factors = ConfigTree(raw_risk_mapping)
risk_factors.freeze()
