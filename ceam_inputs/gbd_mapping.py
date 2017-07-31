"""Mapping of GBD ids onto vivarium conventions."""
from typing import NamedTuple, Union, NewType, List

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
    sequelae: List[Sequela] = None
    etiologies: List[Etiology] = None
    severity_splits: SeveritySplits = None


class Sequela(NamedTuple):
    name: str
    incidence: meid = None
    prevalence: meid = None
    excess_mortality: Union[meid, float] = None
    disability_weight: meid = None
    severity_splits: SeveritySplits = None


class Etiology(NamedTuple):
    name: str
    id: rid = None


class SeveritySplit(NamedTuple):
    split: Union[meid, float]
    prevalence: meid = None
    disability_weight: hid = None


class SeveritySplits(NamedTuple):
    mild: SeveritySplit
    moderate: SeveritySplit
    severe: SeveritySplit
    asymptomatic: SeveritySplit = None


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


class Etioloties(NamedTuple):
    """Holder of Etiologies"""
    unattributed_diarrhea: Etiology
    cholera: Etiology
    other_salmonella: Etiology
    shigellosis: Etiology
    EPEC: Etiology
    ETEC: Etiology
    campylobacter: Etiology
    amoebiasis: Etiology
    cryptosporidiosis: Etiology
    rotaviral_entiritis: Etiology
    aeromonas: Etiology
    clostridium_difficile: Etiology
    norovirus: Etiology
    adenovirus: Etiology


class Sequelae(NamedTuple):
    """Holder of Sequelae"""
    heart_attack: Sequela
    heart_failure: Sequela
    angina: Sequela
    asymptomatic_ihd: Sequela


etiologies = Etioloties(
    unattributed_diarrhea=Etiology(
        name='unattributed_diarrhea',
    ),
    cholera=Etiology(
        name='cholera',
        id=rid(173),
    ),
    other_salmonella=Etiology(
        name='other_salmonella',
        id=rid(174),
    ),
    shigellosis=Etiology(
        name='shigellosis',
    ),
    EPEC=Etiology(
        name='EPEC',
        id=rid(176),
    ),
    ETEC=Etiology(
        name='ETEC',
        id=rid(177),
    ),
    campylobacter=Etiology(
        name='campylobacter',
        id=rid(178),
    ),
    amoebiasis=Etiology(
        name='amoebiasis',
        id=rid(179),
    ),
    cryptosporidiosis=Etiology(
        name='cryptosporidiosis',
        id=rid(180),
    ),
    rotaviral_entiritis=Etiology(
        name='rotaviral_entiritis',
        id=rid(181),
    ),
    aeromonas=Etiology(
        name='aeromonas',
        id=rid(182),
    ),
    clostridium_difficile=Etiology(
        name='clostridium_difficile',
        id=rid(183),
    ),
    norovirus=Etiology(
        name='norovirus',
        id=rid(184),
    ),
    adenovirus=Etiology(
        name='adenovirus',
        id=rid(185),
    ),
)

sequelae = Sequelae(
    heart_attack=Sequela(
        name='heart_attack',
        incidence=meid(1814),
        prevalence=meid(1814),
        excess_mortality=meid(1814),
    ),
    heart_failure=Sequela(
        name='heart_failure',
        excess_mortality=meid(2412),
        severity_splits=SeveritySplits(
            mild=SeveritySplit(
                split=0.182074,
                prevalence=meid(1821),
                disability_weight=meid(1821),
            ),
            moderate=SeveritySplit(
                split=0.149771,
                prevalence=meid(1822),
                disability_weight=meid(1822),
            ),
            severe=SeveritySplit(
                split=0.402838,
                prevalence=meid(1823),
                disability_weight=meid(1823),
            ),
        ),
    ),
    angina=Sequela(
        name='angina',
        incidence=meid(1817),
        excess_mortality=meid(1817),
        severity_splits=SeveritySplits(
            asymptomatic=SeveritySplit(
                split=0.304553,
                prevalence=meid(3102),
                disability_weight=meid(1823),
            ),
            mild=SeveritySplit(
                split=0.239594,
                prevalence=meid(1818),
                disability_weight=meid(1818),
            ),
            moderate=SeveritySplit(
                split=0.126273,
                prevalence=meid(1819),
                disability_weight=meid(1819),
            ),
            severe=SeveritySplit(
                split=0.32958,
                prevalence=meid(1820),
                disability_weight=meid(1820),
            ),
        ),
    ),
    asymptomatic_ihd=Sequela(
        name='asymptomatic_ihd',
        prevalence=meid(3233),
        excess_mortality=0.0,
        disability_weight=meid(3233)
    ),
)


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
        severity_splits=SeveritySplits(
            mild=SeveritySplit(
                split=meid(2608),
                disability_weight=hid(355),
            ),
            moderate=SeveritySplit(
                split=meid(2609),
                disability_weight=hid(356)
            ),
            severe=SeveritySplit(
                split=meid(2610),
                disability_weight=hid(357),
            ),
        ),
        etiologies=[etiologies.adenovirus, etiologies.aeromonas, etiologies.amoebiasis, etiologies.campylobacter,
                    etiologies.cholera, etiologies.clostridium_difficile, etiologies.cryptosporidiosis,
                    etiologies.EPEC, etiologies.ETEC, etiologies.norovirus, etiologies.other_salmonella,
                    etiologies.rotaviral_entiritis, etiologies.shigellosis, etiologies.unattributed_diarrhea]
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
        csmr=cid(493),
        sequelae=[sequelae.heart_attack, sequelae.angina, sequelae.asymptomatic_ihd, sequelae.heart_failure]
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


class Tmred(NamedTuple):
    distribution: str
    min: float
    max: float
    inverted: bool

class Levels(NamedTuple):
    cat1: str
    cat2: str
    cat3: str = None
    cat4: str = None
    cat5: str = None
    cat6: str = None
    cat7: str = None
    cat8: str = None
    cat9: str = None


class Risk(NamedTuple):
    """Container type for risk factor GBD ids."""
    name: str
    id: rid
    distribution: str
    levels: Levels = None
    affected_causes: List[Cause] = None
    tmred: Tmred = None
    scale: float = None
    max_rr: float = None

class Risks(NamedTuple):
    unsafe_water_source: Risk
    unsafe_sanitation: Risk
    ambient_particulate_matter_pollution: Risk
    household_air_pollution_from_solid_fuels: Risk
    ambient_ozone_pollution: Risk
    residential_radon: Risk
    childhood_underweight: Risk
    vitamin_a_deficiency: Risk
    zinc_deficiency: Risk
    secondhand_smoke: Risk
    alcohol_use: Risk
    high_total_cholesterol: Risk
    high_systolic_blood_pressure: Risk
    high_body_mass_index: Risk
    diet_low_in_fruits: Risk
    diet_low_in_vegetables: Risk
    diet_low_in_whole_grains: Risk
    diet_low_in_nuts_and_seeds: Risk
    diet_high_in_processed_meat: Risk
    diet_high_in_sugar_sweetened_beverages: Risk
    diet_low_in_fiber: Risk
    diet_low_in_seafood_omega_3_fatty_acids: Risk
    diet_low_in_polyunsaturated_fatty_acids: Risk
    diet_high_in_trans_fatty_acids: Risk
    diet_high_in_sodium: Risk
    low_physical_activity: Risk
    non_exclusive_breastfeeding: Risk
    discontinued_breastfeeding: Risk
    high_fasting_plasma_glucose_continuous: Risk
    high_fasting_plasma_glucose_categorical: Risk
    low_glomerular_filtration_rate: Risk
    smoking_prevalence_approach: Risk
    no_access_to_handwashing_facility: Risk
    child_wasting: Risk
    child_stunting: Risk
    lead_exposure_in_bone: Risk
    low_measles_vaccine_coverage_first_dose: Risk

risk_factors = Risks(
    unsafe_water_source=Risk(
        name='unsafe_water_source',
        id=rid(83),
        distribution='categorical',
        levels=Levels(
            cat1='unimproved and untreated',
            cat2='unimproved and chlorinated',
            cat3='unimproved and filtered',
            cat4='improved and untreated',
            cat5='improved and chlorinated',
            cat6='improved and filtered',
            cat7='piped and untreated',
            cat8='piped and chlorinated',
            cat9='piped and filtered',
        ),
        affected_causes=[causes.diarrhea, causes.typhoid_fever, causes.paratyphoid_fever],
    ),
    unsafe_sanitation=Risk(
        name='unsafe_sanitation',
        id=rid(84),
        distribution='categorical',
        levels=Levels(
            cat1='unimproved and untreated',
            cat2='improved',
            cat3='sewer',
        ),
        affected_causes=[causes.diarrhea, causes.typhoid_fever, causes.paratyphoid_fever],
    ),
    ambient_particulate_matter_pollution=Risk(
        name='ambient_particulate_matter_pollution',
        id=rid(86),
        distribution='unknown',
        affected_causes=[causes.ischemic_heart_disease, causes.ischemic_stroke,
                         causes.hemorrhagic_stroke, causes.lower_respiratory_infections,
                         causes.tracheal_bronchus_and_lung_cancer, causes.chronic_obstructive_pulmonary_disease],
        scale=1,
        max_rr=500,
    ),
    household_air_pollution_from_solid_fuels=Risk(
        name='household_air_pollution_from_solid_fuels',
        id=rid(87),
        distribution='categorical',
        levels=Levels(
            cat1='exposed',
            cat2='unexposed',
        ),
        affected_causes=[causes.ischemic_heart_disease, causes.ischemic_stroke,
                         causes.hemorrhagic_stroke, causes.lower_respiratory_infections,
                         causes.tracheal_bronchus_and_lung_cancer, causes.chronic_obstructive_pulmonary_disease],
    ),
    ambient_ozone_pollution=Risk(
        name='ambient_ozone_pollution',
        id=rid(88),
        distribution='unknown',
        affected_causes=[causes.chronic_obstructive_pulmonary_disease],
        tmred=Tmred(
            distribution='uniform',
            min=30,
            max=50,
            inverted=False,
        ),
        scale=10,
        max_rr=100,
    ),
    residential_radon=Risk(
        name='residential_radon',
        id=rid(90),
        distribution= 'lognormal',
        affected_causes=[causes.tracheal_bronchus_and_lung_cancer],
        tmred=Tmred(
            distribution='uniform',
            min=7,
            max=14.8,
            inverted=False,
        ),
        scale=1,
        max_rr=10000,
    ),
    childhood_underweight=Risk(
        name='childhood_underweight',
        id=rid(94),
        distribution='categorical',
        levels=Levels(
            cat1='more than 3 std. dev. below target',
            cat2='2 to 3 std. dev. below target',
            cat3='1 to 2 std. dev. below target',
            cat4='less than 1 std. dev. below target',
        ),
        affected_causes=[causes.diarrhea],
    ),
    vitamin_a_deficiency=Risk(
        name='vitamin_a_deficiency',
        id=rid(96),
        distribution='categorical',
        levels=Levels(
            cat1='deficient',
            cat2='not deficient',
        ),
        affected_causes=[causes.diarrhea],
    ),
    zinc_deficiency=Risk(
        name='zinc_deficiency',
        id=rid(97),
        distribution='categorical',
        levels= Levels(
            cat1='deficient',
            cat2='not deficient',
        ),
        affected_causes=[causes.diarrhea],
    ),
    secondhand_smoke=Risk(
        name='secondhand_smoke',
        id=rid(100),
        distribution='categorical',
        levels=Levels(
            cat1='exposed',
            cat2='unexposed',
        ),
        affected_causes=[causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke],
    ),
    alcohol_use=Risk(
        name='alcohol_use',
        id=rid(102),
        distribution='unknown',
        affected_causes=[causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.hypertensive_heart_disease,causes.atrial_fibrillation_and_flutter],
        scale=1,
        max_rr=100,
    ),
    high_total_cholesterol=Risk(
        name='high_total_cholesterol',
        id=rid(106),
        distribution='lognormal',
        affected_causes=[causes.ischemic_heart_disease, causes.ischemic_stroke],
        tmred=Tmred(
            distribution='uniform',
            min=2.78,
            max=3.38,
            inverted=False,
        ),
        scale=1,
        max_rr=10,
    ),
    high_systolic_blood_pressure=Risk(
        name='high_systolic_blood_pressure',
        id=rid(107),
        distribution='lognormal',
        affected_causes=[causes.rheumatic_heart_disease, causes.ischemic_heart_disease, causes.ischemic_stroke,
                         causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                         causes.cardiomyopathy_and_myocarditis, causes.atrial_fibrillation_and_flutter,
                         causes.aortic_aneurysm, causes.peripheral_vascular_disease, causes.endocarditis,
                         causes.other_cardiovascular_and_circulatory_diseases,
                         causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_hypertension,
                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
                         causes.chronic_kidney_disease_due_to_other_causes],
        tmred=Tmred(
            distribution='uniform',
            min=110,
            max=115,
            inverted=False,
        ),
        scale=10,
        max_rr=200,
    ),
    high_body_mass_index=Risk(
        name='high_body_mass_index',
        id=rid(108),
        distribution='beta',
        affected_causes=[causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.hypertensive_heart_disease, causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_hypertension,
                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
                         causes.chronic_kidney_disease_due_to_other_causes],
        tmred=Tmred(
            distribution='uniform',
            min=20,
            max=25,
            inverted=False,
        ),
        scale=5,
        max_rr=50,
    ),
    diet_low_in_fruits=Risk(
        name='diet_low_in_fruits',
        id=rid(111),
        distribution='lognormal',
        affected_causes=[causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke],
        tmred=Tmred(
            distribution='uniform',
            min=200,
            max=300,
            inverted=True,
        ),
        scale=100,
        max_rr=400,
    ),
    diet_low_in_vegetables=Risk(
        name='diet_low_in_vegetables',
        id=rid(112),
        distribution='lognormal',
        affected_causes=[causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke],
        tmred=Tmred(
            distribution='uniform',
            min=290,
            max=430,
            inverted=True,
        ),
        scale=100,
        max_rr=450,
    ),
    diet_low_in_whole_grains=Risk(
        name='diet_low_in_whole_grains',
        id=rid(113),
        distribution='lognormal',
        affected_causes=[causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke],
        tmred=Tmred(
            distribution='uniform',
            min=100,
            max=150,
            inverted=True,
        ),
        scale=50,
        max_rr=150,
    ),
    diet_low_in_nuts_and_seeds=Risk(
        name='diet_low_in_nuts_and_seeds',
        id=rid(114),
        distribution='lognormal',
        affected_causes=[causes.ischemic_heart_disease],
        tmred=Tmred(
            distribution='uniform',
            min=16,
            max=25,
            inverted=True,
        ),
        scale=4.05,
        max_rr=24,
    ),
    diet_high_in_processed_meat=Risk(
        name='diet_high_in_processed_meat',
        id=rid(117),
        distribution='lognormal',
        affected_causes=[causes.ischemic_heart_disease],
        tmred=Tmred(
            distribution='uniform',
            min=0,
            max=4,
            inverted=False,
        ),
        scale=50,
        max_rr=1000,
    ),
    diet_high_in_sugar_sweetened_beverages=Risk(
        name='diet_high_in_sugar_sweetened_beverages',
        id=rid(118),
        distribution='lognormal',
        affected_causes=[causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.hypertensive_heart_disease, causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_hypertension,
                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
                         causes.chronic_kidney_disease_due_to_other_causes],
        tmred=Tmred(
            distribution='uniform',
            min=0,
            max=5,
            inverted=False,
        ),
        scale=226.8,
        max_rr=5000,
    ),
    diet_low_in_fiber=Risk(
        name='diet_low_in_fiber',
        id=rid(119),
        distribution='lognormal',
        affected_causes=[causes.ischemic_heart_disease],
        tmred=Tmred(
            distribution='uniform',
            min=19,
            max=28,
            inverted=True,
        ),
        scale=20,
        max_rr=32,
    ),
    diet_low_in_seafood_omega_3_fatty_acids=Risk(
        name='diet_low_in_seafood_omega_3_fatty_acids',
        id=rid(121),
        distribution='lognormal',
        affected_causes=[causes.ischemic_heart_disease],
        tmred=Tmred(
            distribution='uniform',
            min=200,
            max=300,
            inverted=True,
        ),
        scale=100,
        max_rr=1000,
    ),
    diet_low_in_polyunsaturated_fatty_acids=Risk(
        name='diet_low_in_polyunsaturated_fatty_acids',
        id=rid(122),
        distribution='lognormal',
        affected_causes=[causes.ischemic_heart_disease],
        tmred=Tmred(
            distribution='uniform',
            min=0.09,
            max=0.13,
            inverted=True,
        ),
        scale=0.05,
        max_rr=0.149,
    ),
    diet_high_in_trans_fatty_acids=Risk(
        name='diet_high_in_trans_fatty_acids',
        id=rid(123),
        distribution='lognormal',
        affected_causes=[causes.ischemic_heart_disease],
        tmred=Tmred(
            distribution='uniform',
            min=0,
            max=0.01,
            inverted=False,
        ),
        scale=0.02,
        max_rr=1,
    ),
    diet_high_in_sodium=Risk(
        name='diet_high_in_sodium',
        id=rid(124),
        distribution='lognormal',
        affected_causes=[causes.rheumatic_heart_disease, causes.ischemic_heart_disease, causes.ischemic_stroke,
                         causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                         causes.cardiomyopathy_and_myocarditis, causes.atrial_fibrillation_and_flutter,
                         causes.aortic_aneurysm, causes.peripheral_vascular_disease, causes.endocarditis,
                         causes.other_cardiovascular_and_circulatory_diseases,
                         causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_hypertension,
                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
                         causes.chronic_kidney_disease_due_to_other_causes],
        tmred=Tmred(
            distribution='uniform',
            min=1,
            max=5,
            inverted=False,
        ),
        scale=1,
        max_rr=50,
    ),
    low_physical_activity=Risk(
        name='low_physical_activity',
        id=rid(125),
        distribution='weibull',
        affected_causes=[causes.ischemic_heart_disease, causes.ischemic_stroke],
        tmred=Tmred(
            distribution='uniform',
            min=3000,
            max=4500,
            inverted=True,
        ),
        scale=600,
        max_rr=5000,
    ),
    non_exclusive_breastfeeding=Risk(
        name='non_exclusive_breastfeeding',
        id=rid(136),
        distribution='categorical',
        affected_causes=[causes.diarrhea],
    ),
    discontinued_breastfeeding=Risk(
        name='discontinued_breastfeeding',
        id=rid(137),
        distribution='categorical',
        affected_causes=[causes.diarrhea],
    ),
    high_fasting_plasma_glucose_continuous=Risk(
        name='high_fasting_plasma_glucose_continuous',
        id=rid(141),
        distribution='lognormal',
        affected_causes=[causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_hypertension,
                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
                         causes.chronic_kidney_disease_due_to_other_causes],
        tmred=Tmred(
            distribution='uniform',
            min=4.88488,
            max=5.301205,
            inverted=False,
        ),
        scale=1,
        max_rr=30,
    ),
    high_fasting_plasma_glucose_categorical=Risk(
        name='high_fasting_plasma_glucose_categorical',
        id=rid(142),
        distribution='categorical',
        affected_causes=[causes.peripheral_vascular_disease],
    ),
    low_glomerular_filtration_rate=Risk(
        name='low_glomerular_filtration_rate',
        id=rid(143),
        distribution='unknown',
        affected_causes=[causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.peripheral_vascular_disease, causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_hypertension,
                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
                         causes.chronic_kidney_disease_due_to_other_causes],
    ),
    smoking_prevalence_approach=Risk(
        name='smoking_prevalence_approach',
        id=rid(166),
        distribution='categorical',
        affected_causes=[causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.hypertensive_heart_disease, causes.atrial_fibrillation_and_flutter,
                         causes.aortic_aneurysm, causes.peripheral_vascular_disease,
                         causes.other_cardiovascular_and_circulatory_diseases],
    ),
    no_access_to_handwashing_facility=Risk(
        name='no_access_to_handwashing_facility',
        id=rid(238),
        distribution='categorical',
        affected_causes=[causes.diarrhea],
    ),
    child_wasting=Risk(
        name='child_wasting',
        id=rid(240),
        distribution='categorical',
        affected_causes=[causes.diarrhea],
    ),
    child_stunting=Risk(
        name='child_stunting',
        id=rid(241),
        distribution='categorical',
        affected_causes=[causes.diarrhea],
    ),
    lead_exposure_in_bone=Risk(
        name='lead_exposure_in_bone',
        id=rid(243),
        distribution='lognormal',
        affected_causes=[causes.rheumatic_heart_disease, causes.ischemic_heart_disease, causes.ischemic_stroke,
                         causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                         causes.cardiomyopathy_and_myocarditis, causes.atrial_fibrillation_and_flutter,
                         causes.aortic_aneurysm, causes.peripheral_vascular_disease, causes.endocarditis,
                         causes.other_cardiovascular_and_circulatory_diseases,
                         causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_hypertension,
                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
                         causes.chronic_kidney_disease_due_to_other_causes],
        tmred=Tmred(
            distribution='uniform',
            min=0,
            max=20,
            inverted=False,
        ),
        scale=10,
        max_rr=500,
    ),
    low_measles_vaccine_coverage_first_dose=Risk(
        name='low_measles_vaccine_coverage_first_dose',
        id=rid(318),
        distribution='categorical',
        affected_causes=[causes.diarrhea],
    ),
)
