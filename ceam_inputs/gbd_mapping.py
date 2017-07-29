"""Mapping of GBD ids onto vivarium conventions."""
from typing import NamedTuple, Union, NewType, List
from enum import Enum

meid = NewType('meid', int)  # Modelable entity id.
rid = NewType('rid', int)  # Risk id.
cid = NewType('cid', int)  # Cause id.
hid = NewType('hid', int)  # Healthstate id.


class Cause(NamedTuple):
    """Container type for cause GBD ids."""
    name: str
    id: cid
    incidence: meid = None
    prevalence: meid = None
    csmr: cid = None
    excess_mortality: Union[meid, float] = None
    disability_weight: Union[meid, hid, float] = None
    duration: meid = None
    sub_causes: List = []


class SubCause(NamedTuple):
    """Container type for sub-cause GBD ids."""
    name: str
    parent: Cause
    id: rid = None
    split: Union[meid, float] = None
    incidence: meid = None
    prevalence: meid = None
    excess_mortality: Union[meid, float] = None
    disability_weight: Union[meid, hid, float] = None
    duration: meid = None


class Causes(NamedTuple):
    """Holder of causes"""
    all_causes: Cause
    tuberculosis: Cause
    hiv_aids_tuberculosis: Cause
    hiv_aids_other_diseases: Cause
    diarrhea: Cause
    typhoid_fever: Cause
    paratyphoid_fever: Cause
    lower_respiratory_infections: Cause
    otitis_media: Cause
    measles: Cause
    maternal_hemorrhage: Cause
    maternal_sepsis_and_other_infections: Cause
    maternal_abortion_miscarriage_and_ectopic_pregnancy: Cause
    protein_energy_malnutrition: Cause
    vitamin_a_deficiency: Cause
    iron_deficiency_anemia: Cause
    syphilis: Cause
    chlamydia: Cause
    gonorrhea: Cause
    trichomoniasis: Cause
    genital_herpes: Cause
    other_sexually_transmitted_disease: Cause
    hepatitis_b: Cause
    hepatitis_c: Cause
    esophageal_cancer: Cause
    stomach_cancer: Cause
    tracheal_bronchus_and_lung_cancer: Cause
    rheumatic_heart_disease: Cause
    ischemic_heart_disease: Cause
    chronic_stroke: Cause
    ischemic_stroke: Cause
    hemorrhagic_stroke: Cause
    hypertensive_heart_disease: Cause
    cardiomyopathy_and_myocarditis: Cause
    atrial_fibrillation_and_flutter: Cause
    aortic_aneurysm: Cause
    peripheral_vascular_disease: Cause
    endocarditis: Cause
    other_cardiovascular_and_circulatory_diseases: Cause
    chronic_obstructive_pulmonary_disease: Cause
    chronic_kidney_disease_due_to_diabetes_mellitus: Cause
    chronic_kidney_disease_due_to_hypertension: Cause
    chronic_kidney_disease_due_to_glomerulonephritis: Cause
    chronic_kidney_disease_due_to_other_causes: Cause
    cataract: Cause


causes = Causes(
    all_causes=Cause(
        name='all_causes',
        id=cid(294)
    ),
    tuberculosis=Cause(
        name='tuberculosis',
        id=cid(297),
    ),
    hiv_aids_tuberculosis=Cause(
        name='hiv_aids_tuberculosis',
        id=cid(299),
    ),
    hiv_aids_other_diseases=Cause(
        name='hiv_aids_other_diseases',
        id=cid(300),
    ),
    diarrhea=Cause(
        name='diarrhea',
        id=cid(302),
        incidence=meid(1181),
        prevalence=meid(1181),
        csmr=cid(302),
        excess_mortality=meid(1181),
        disability_weight=0.23,
        duration=meid(1181),
    ),
    typhoid_fever=Cause(
        name='typhoid_fever',
        id=cid(319),
    ),
    paratyphoid_fever=Cause(
        name='paratyphoid_fever',
        id=cid(320),
    ),
    lower_respiratory_infections=Cause(
        name='lower_respiratory_infections',
        id=cid(322),
    ),
    otitis_media=Cause(
        name='otitis_media',
        id=cid(329),
    ),
    measles=Cause(
        name='measles',
        id=cid(341),
    ),
    maternal_hemorrhage=Cause(
        name='maternal_hemorrhage',
        id=cid(367),
    ),
    maternal_sepsis_and_other_infections=Cause(
        name='maternal_sepsis_and_other_infections',
        id=cid(368),
    ),
    maternal_abortion_miscarriage_and_ectopic_pregnancy=Cause(
        name='maternal_abortion_miscarriage_and_ectopic_pregnancy',
        id=cid(371),
    ),
    protein_energy_malnutrition=Cause(
        name='protein_energy_malnutrition',
        id=cid(387),
    ),
    vitamin_a_deficiency=Cause(
        name='vitamin_a_deficiency',
        id=cid(389),
    ),
    iron_deficiency_anemia=Cause(
        name='iron_deficiency_anemia',
        id=cid(390),
    ),
    syphilis=Cause(
        name='syphilis',
        id=cid(394),
    ),
    chlamydia=Cause(
        name='chlamydia',
        id=cid(395),
    ),
    gonorrhea=Cause(
        name='gonorrhea',
        id=cid(396),
    ),
    trichomoniasis=Cause(
        name='trichomoniasis',
        id=cid(397),
    ),
    genital_herpes=Cause(
        name='genital_herpes',
        id=cid(398),
    ),
    other_sexually_transmitted_disease=Cause(
        name='other_sexually_transmitted_disease',
        id=cid(399),
    ),
    hepatitis_b=Cause(
        name='hepatitis_b',
        id=cid(402),
    ),
    hepatitis_c=Cause(
        name='hepatitis_c',
        id=cid(403),
    ),
    esophageal_cancer=Cause(
        name='esophageal_cancer',
        id=cid(411),
    ),
    stomach_cancer=Cause(
        name='stomach_cancer',
        id=cid(414),
    ),
    tracheal_bronchus_and_lung_cancer=Cause(
        name='tracheal_bronchus_and_lung_cancer',
        id=cid(426),
    ),
    rheumatic_heart_disease=Cause(
        name='rheumatic_heart_disease',
        id=cid(492),
    ),
    ischemic_heart_disease=Cause(
        name='ischemic_heart_disease',
        id=cid(493),
        csmr=cid(493)
    ),
    chronic_stroke=Cause(
        name='chronic_stroke',
        id=cid(494),
        prevalence=cid(494),
        csmr=cid(494),
        excess_mortality=meid(9312),
        disability_weight=0.32,
    ),
    ischemic_stroke=Cause(
        name='ischemic_stroke',
        id=cid(495),
        incidence=meid(9310),
        prevalence=meid(9310),
        csmr=cid(495),
        excess_mortality=meid(9310),
        disability_weight=0.32,
    ),
    hemorrhagic_stroke=Cause(
        name='hemorrhagic_stroke',
        id=cid(496),
        incidence=meid(9311),
        prevalence=meid(9311),
        csmr=cid(496),
        excess_mortality=meid(9311),
        disability_weight=0.32,
    ),
    hypertensive_heart_disease=Cause(
        name='hypertensive_heart_disease',
        id=cid(498),
    ),
    cardiomyopathy_and_myocarditis=Cause(
        name='cardiomyopathy_and_myocarditis',
        id=cid(499),
    ),
    atrial_fibrillation_and_flutter=Cause(
        name='atrial_fibrillation_and_flutter',
        id=cid(500),
    ),
    aortic_aneurysm=Cause(
        name='aortic_aneurysm',
        id=cid(501),
    ),
    peripheral_vascular_disease=Cause(
        name='peripheral_vascular_disease',
        id=cid(502),
    ),
    endocarditis=Cause(
        name='endocarditis',
        id=cid(503),
    ),
    other_cardiovascular_and_circulatory_diseases=Cause(
        name='other_cardiovascular_and_circulatory_diseases',
        id=cid(507),
    ),
    chronic_obstructive_pulmonary_disease=Cause(
        name='chronic_obstructive_pulmonary_disease',
        id=cid(509),
    ),
    chronic_kidney_disease_due_to_diabetes_mellitus=Cause(
        name='chronic_kidney_disease_due_to_diabetes_mellitus',
        id=cid(590),
    ),
    chronic_kidney_disease_due_to_hypertension=Cause(
        name='chronic_kidney_disease_due_to_hypertension',
        id=cid(591),
    ),
    chronic_kidney_disease_due_to_glomerulonephritis=Cause(
        name='chronic_kidney_disease_due_to_glomerulonephritis',
        id=cid(592),
    ),
    chronic_kidney_disease_due_to_other_causes=Cause(
        name='chronic_kidney_disease_due_to_other_causes',
        id=cid(593),
    ),
    cataract=Cause(
        name='cataract',
        id=cid(671),
    ),
)


class SubCauses(NamedTuple):
    """Holder of sub-causes"""
    mild_diarrhea: SubCause
    moderate_diarrhea: SubCause
    severe_diarrhea: SubCause
    unattributed_diarrhea: SubCause
    cholera: SubCause
    other_salmonella: SubCause
    shigellosis: SubCause
    EPEC: SubCause
    ETEC: SubCause
    campylobacter: SubCause
    amoebiasis: SubCause
    cryptosporidiosis: SubCause
    rotaviral_entiritis: SubCause
    aeromonas: SubCause
    clostridium_difficile: SubCause
    norovirus: SubCause
    adenovirus: SubCause
    heart_attack: SubCause
    mild_heart_failure: SubCause
    moderate_heart_failure: SubCause
    severe_heart_failure: SubCause
    angina_not_due_to_MI: SubCause
    asymptomatic_angina: SubCause
    mild_angina: SubCause
    moderate_angina: SubCause
    severe_angina: SubCause
    asymptomatic_ihd: SubCause


sub_causes = SubCauses(
    mild_diarrhea=SubCause(
        name='mild_diarrhea',
        parent=causes.diarrhea,
        split=meid(2608),
        disability_weight=hid(355),
        duration=causes.diarrhea.duration,
    ),
    moderate_diarrhea=SubCause(
        name='moderate_diarrhea',
        parent=causes.diarrhea,
        split=meid(2609),
        disability_weight=hid(356),
        duration=causes.diarrhea.duration,
    ),
    severe_diarrhea=SubCause(
        name='severe_diarrhea',
        parent=causes.diarrhea,
        split=meid(2610),
        disability_weight=hid(357),
        duration=causes.diarrhea.duration,
    ),
    unattributed_diarrhea=SubCause(
        name='unattributed_diarrhea',
        parent=causes.diarrhea
    ),
    cholera=SubCause(
        name='cholera',
        parent=causes.diarrhea,
        id=rid(173),
    ),
    other_salmonella=SubCause(
        name='other_salmonella',
        parent=causes.diarrhea,
        id=rid(174),
    ),
    shigellosis=SubCause(
        name='shigellosis',
        parent=causes.diarrhea,
        id= rid(175),
    ),
    EPEC=SubCause(
        name='EPEC',
        parent=causes.diarrhea,
        id=rid(176),
    ),
    ETEC=SubCause(
        name='ETEC',
        parent=causes.diarrhea,
        id=rid(177),
    ),
    campylobacter=SubCause(
        name='campylobacter',
        parent=causes.diarrhea,
        id=rid(178),
    ),
    amoebiasis=SubCause(
        name='amoebiasis',
        parent=causes.diarrhea,
        id=rid(179),
    ),
    cryptosporidiosis=SubCause(
        name='cryptosporidiosis',
        parent=causes.diarrhea,
        id=rid(180),
    ),
    rotaviral_entiritis=SubCause(
        name='rotaviral_entiritis',
        parent=causes.diarrhea,
        id=rid(181),
    ),
    aeromonas=SubCause(
        name='aeromonas',
        parent=causes.diarrhea,
        id=rid(182),
    ),
    clostridium_difficile=SubCause(
        name='clostridium_difficile',
        parent=causes.diarrhea,
        id=rid(183),
    ),
    norovirus=SubCause(
        name='norovirus',
        parent=causes.diarrhea,
        id=rid(184),
    ),
    adenovirus=SubCause(
        name='adenovirus',
        parent=causes.diarrhea,
        id=rid(185),
    ),
    heart_attack=SubCause(
        name='heart_attack',
        parent=causes.ischemic_heart_disease,
        incidence=meid(1814),
        prevalence=meid(1814),
        excess_mortality=meid(1814),
    ),
    mild_heart_failure=SubCause(
        name='mild_heart_failure',
        parent=causes.ischemic_heart_disease,
        prevalence=meid(1821),
        excess_mortality=meid(2412),
        disability_weight=meid(1821),
    ),
    moderate_heart_failure=SubCause(
        name='moderate_heart_failure',
        parent=causes.ischemic_heart_disease,
        prevalence=meid(1822),
        excess_mortality=meid(2412),
        disability_weight=meid(1822),
    ),
    severe_heart_failure=SubCause(
        name='severe_heart_failure',
        parent=causes.ischemic_heart_disease,
        prevalence=meid(1823),
        excess_mortality=meid(2412),
        disability_weight=meid(1823)
    ),
    angina_not_due_to_MI=SubCause(
        name='angina_not_due_to_MI',
        parent=causes.ischemic_heart_disease,
        incidence=meid(1817),
    ),
    asymptomatic_angina=SubCause(
        name='asymptomatic_angina',
        parent=causes.ischemic_heart_disease,
        prevalence=meid(3102),
        excess_mortality=meid(1817),
        disability_weight=meid(1823),
    ),
    mild_angina=SubCause(
        name='mild_angina',
        parent=causes.ischemic_heart_disease,
        prevalence=meid(1818),
        excess_mortality=meid(1817),
        disability_weight=meid(1818),
    ),
    moderate_angina=SubCause(
        name='moderate_angina',
        parent=causes.ischemic_heart_disease,
        prevalence=meid(1819),
        excess_mortality=meid(1817),
        disability_weight=meid(1819),
    ),
    severe_angina=SubCause(
        name='severe_angina',
        parent=causes.ischemic_heart_disease,
        prevalence=meid(1820),
        excess_mortality=meid(1817),
        disability_weight=meid(1820)
    ),
    asymptomatic_ihd=SubCause(
        name='asymptomatic_ihd',
        parent=causes.ischemic_heart_disease,
        prevalence=meid(3233),
        excess_mortality=0.0,
        disability_weight=meid(3233)
    ),
)


class Risk(NamedTuple):
    name: str
    id: rid
    distribution:

raw_risk_mapping = {
    'unsafe_water_source': {
        'gbd_risk': rid(83),
        'distribution': 'categorical',
        'levels': {
            'cat1': 'unimproved and untreated',
            'cat2': 'unimproved and chlorinated',
            'cat3': 'unimproved and filtered',
            'cat4': 'improved and untreated',
            'cat5': 'improved and chlorinated',
            'cat6': 'improved and filtered',
            'cat7': 'piped and untreated',
            'cat8': 'piped and chlorinated',
            'cat9': 'piped and filtered',
        },
        'affected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC,
                            causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis,
                            causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile,
                            causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea,
                            causes.typhoid_fever, causes.paratyphoid_fever],
    },
    'unsafe_sanitation': {
        'gbd_risk': rid(84),
        'distribution': 'categorical',
        'levels': {
            'cat1': 'unimproved and untreated',
            'cat2': 'improved',
            'cat3': 'sewer'
        },
        'affected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC,
                            causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis,
                            causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile,
                            causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea,
                            causes.typhoid_fever, causes.paratyphoid_fever],
    },
    'ambient_particulate_matter_pollution': {
        'gbd_risk': rid(86),
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke,
                            causes.hemorrhagic_stroke, causes.lower_respiratory_infections,
                            causes.tracheal_bronchus_and_lung_cancer, causes.chronic_obstructive_pulmonary_disease],
        'scale': 1,
        'max_rr': 500,
    },
    'household_air_pollution_from_solid_fuels': {
        'gbd_risk': rid(87),
        'distribution': 'categorical',
        'levels': {
            'cat1': 'Exposed',
            'cat2': 'Unexposed',
        },
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke,
                            causes.hemorrhagic_stroke, causes.lower_respiratory_infections,
                            causes.tracheal_bronchus_and_lung_cancer, causes.chronic_obstructive_pulmonary_disease],
    },
    'ambient_ozone_pollution': {
        'gbd_risk': rid(88),
        'distribution': '',
        'affected_causes': [causes.chronic_obstructive_pulmonary_disease],
        'tmred': {
            'distribution': 'uniform',
            'min': 30,
            'max': 50,
            'inverted': False,
        },
        'scale': 10,
        'max_rr': 100,
    },
    'residential_radon': {
        'gbd_risk': rid(90),
        'distribution': 'lognormal',
        'affected_causes': [causes.tracheal_bronchus_and_lung_cancer],
        'tmred': {
            'distribution': 'uniform',
            'min': 7,
            'max': 14.8,
            'inverted': False,
        },
        'scale': 1,
        'max_rr': 10000,
    },
    'childhood_underweight': {
        'gbd_risk': rid(94),
        'distribution': 'categorical',
        'levels': {
            'cat1': 'more than 3 std. dev. below target',
            'cat2': '2 to 3 std. dev. below target',
            'cat3': '1 to 2 std. dev. below target',
            'cat4': 'less than 1 std. dev. below target',
        },
        'affected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC,
                            causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis,
                            causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile,
                            causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea]
    },
    'vitamin_a_deficiency': {
        'gbd_risk': rid(96),
        'distribution': 'categorical',
        'levels': {
            'cat1': 'Exposed',
            'cat2': 'Unexposed',
        },
        'affected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC,
                            causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis,
                            causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile,
                            causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'zinc_deficiency': {
        'gbd_risk': rid(97),
        'distribution': 'categorical',
        'levels': 2,
        'affected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC,
                            causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis,
                            causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile,
                            causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'secondhand_smoke': {
        'gbd_risk': rid(100),
        'distribution': 'categorical',
        'levels': 2,
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
                            causes.ischemic_stroke, causes.hemorrhagic_stroke],
    },
    # 'alcohol_use': {
    #     'gbd_risk': rid(102),
    #     'distribution': 'continuous',
    #     'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke,
    #                         causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
    #                         causes.atrial_fibrillation_and_flutter],
    # },
    'high_total_cholesterol': {
        'gbd_risk': rid(106),
        'distribution': 'lognormal',
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke],
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
        'affected_causes': [causes.rheumatic_heart_disease, causes.heart_attack, causes.ischemic_heart_disease,
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
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke,
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
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
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
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
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
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
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
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease],
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
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease],
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
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke,
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
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease],
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
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease],
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
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease],
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
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease],
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
        'affected_causes': [causes.rheumatic_heart_disease, causes.heart_attack, causes.ischemic_heart_disease,
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
    #     'distribution': 'weibull',
    #     'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke],
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
        'affected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC,
                            causes.ETEC, causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis,
                            causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile,
                            causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'discontinued_breastfeeding': {
        'gbd_risk': rid(137),
        'distribution': 'categorical',
        'affected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC,
                            causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis,
                            causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile,
                            causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'high_fasting_plasma_glucose_continuous': {
        'gbd_risk': rid(141),
        'distribution': 'lognormal',
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease,
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
        'affected_causes': [causes.peripheral_vascular_disease],
    },
    # 'low_glomerular_filtration_rate': {
    #     'gbd_risk': rid(143),
    #     'affected_causes': [causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
    #                         causes.peripheral_vascular_disease, causes.chronic_kidney_disease_due_to_diabetes_mellitus,
    #                         causes.chronic_kidney_disease_due_to_hypertension,
    #                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
    #                         causes.chronic_kidney_disease_due_to_other_causes],
    # },
    'smoking_prevalence_approach': {
        'gbd_risk': rid(166),
        'distribution': 'categorical',
        'affected_causes': [causes.heart_attack, causes.ischemic_heart_disease, causes.ischemic_stroke,
                            causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                            causes.atrial_fibrillation_and_flutter, causes.aortic_aneurysm,
                            causes.peripheral_vascular_disease, causes.other_cardiovascular_and_circulatory_diseases],
    },
    'no_access_to_handwashing_facility': {
        'gbd_risk': rid(238),
        'distribution': 'categorical',
        'affected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC,
                            causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis,
                            causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile,
                            causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'child_wasting': {
        'gbd_risk': rid(240),
        'distribution': 'categorical',
        'affected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC,
                            causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis,
                            causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile,
                            causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'child_stunting': {
        'gbd_risk': rid(241),
        'distribution': 'categorical',
        'affected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC, causes.ETEC,
                            causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis,
                            causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile,
                            causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
    'lead_exposure_in_bone': {
        'gbd_risk': rid(243),
        'distribution': 'lognormal',
        'affected_causes': [causes.rheumatic_heart_disease, causes.heart_attack, causes.ischemic_heart_disease,
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
        'affected_causes': [causes.cholera, causes.other_salmonella, causes.shigellosis, causes.EPEC,
                            causes.ETEC, causes.campylobacter, causes.amoebiasis, causes.cryptosporidiosis,
                            causes.rotaviral_entiritis, causes.aeromonas, causes.clostridium_difficile,
                            causes.norovirus, causes.adenovirus, causes.unattributed_diarrhea],
    },
}

for k, v in raw_risk_mapping.items():
    v['name'] = k

risk_factors = ConfigTree(raw_risk_mapping)
risk_factors.freeze()
