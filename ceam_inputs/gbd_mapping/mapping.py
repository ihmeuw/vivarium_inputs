"""Mapping of GBD ids onto vivarium conventions."""

from gbd_mapping.templates import *

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
        etiologies={etiologies.adenovirus, etiologies.aeromonas, etiologies.amoebiasis, etiologies.campylobacter,
                    etiologies.cholera, etiologies.clostridium_difficile, etiologies.cryptosporidiosis,
                    etiologies.EPEC, etiologies.ETEC, etiologies.norovirus, etiologies.other_salmonella,
                    etiologies.rotaviral_entiritis, etiologies.shigellosis, etiologies.unattributed_diarrhea},
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
    liver_cancer_due_to_hepatitis_b=Cause(
        name='liver_cancer_due_to_hepatitis_b',
        id=cid(418),
    ),
    liver_cancer_due_to_hepatitis_c=Cause(
        name='liver_cancer_due_to_hepatitis_c',
        id=cid(419),
    ),
    liver_cancer_due_to_alcohol_use=Cause(
        name='liver_cancer_due_to_alcohol_use',
        id=cid(420),
    ),
    liver_cancer_due_to_other_causes=Cause(
        name='liver_cancer_due_to_other_causes',
        id=cid(421)
    ),
    larynx_cancer=Cause(
        name='larynx_cancer',
        id=cid(423)
    ),
    tracheal_bronchus_and_lung_cancer=Cause(
        name='tracheal_bronchus_and_lung_cancer',
        id=cid(426),
    ),
    breast_cancer=Cause(
        name='breast_cancer',
        id=cid(429),
    ),
    cervical_cancer=Cause(
        name='cervical_cancer',
        id=cid(432),
    ),
    uterine_cancer=Cause(
        name='uterine_cancer',
        id=cid(435),
    ),
    colon_and_rectum_cancer=Cause(
        name='colon_and_rectum_cancer',
        id=cid(441),
    ),
    lip_and_oral_cavity_cancer=Cause(
        name='lip_and_oral_cavity_cancer',
        id=cid(444),
    ),
    nasopharynx_cancer=Cause(
        name='nasopharynx_cancer',
        id=cid(447),
    ),
    other_pharynx_cancer=Cause(
        name='other_pharynx_cancer',
        id=cid(450),
    ),
    gallbladder_and_biliary_tract_cancer=Cause(
        name='gallbladder_and_biliary_tract_cancer',
        id=cid(453),
    ),
    pancreatic_cancer=Cause(
        name='pancreatic_cancer',
        id=cid(456),
    ),
    ovarian_cancer=Cause(
        name='ovarian_cancer',
        id=cid(465)
    ),
    kidney_cancer=Cause(
        name='kidney_cancer',
        id=cid(471)
    ),
    bladder_cancer=Cause(
        name='bladder_cancer',
        id=cid(474),
    ),
    thyroid_cancer=Cause(
        name='thyroid_cancer',
        id=cid(480),
    ),
    mesothelioma=Cause(
        name='mesothelioma',
        id=cid(483),
    ),
    rheumatic_heart_disease=Cause(
        name='rheumatic_heart_disease',
        id=cid(492),
    ),
    ischemic_heart_disease=Cause(
        name='ischemic_heart_disease',
        id=cid(493),
        csmr=cid(493),
        sequelae={sequelae.heart_attack, sequelae.angina, sequelae.asymptomatic_ihd, sequelae.heart_failure},
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
    silicosis=Cause(
        name='silicosis',
        id=cid(511),
    ),
    asbestosis=Cause(
        name='asbestosis',
        id=cid(512),
    ),
    coal_workers_pneumoconiosis=Cause(
        name='coal_workers_pneumoconiosis',
        id=cid(513),
    ),
    other_pneumoconiosis=Cause(
        name='other_pneumoconiosis',
        id=cid(514)
    ),
    asthma=Cause(
        name='asthma',
        id=cid(515),
    ),
    interstitial_lung_disease_and_pulmonary_sarcoidosis=Cause(
        name='interstitial_lung_disease_and_pulmonary_sarcoidosis',
        id=cid(516),
    ),
    other_chronic_respiratory_diseases=Cause(
        name='other_chronic_respiratory_diseases',
        id=cid(520),
    ),
    cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_b=Cause(
        name='cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_b',
        id=cid(522),
    ),
    cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_c=Cause(
        name='cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_c',
        id=cid(523),
    ),
    cirrhosis_and_other_chronic_liver_diseases_due_to_alcohol_use=Cause(
        name='cirrhosis_and_other_chronic_liver_diseases_due_to_alcohol_use',
        id=cid(524),
    ),
    cirrhosis_and_other_chronic_liver_diseases_due_to_other_causes=Cause(
        name='cirrhosis_and_other_chronic_liver_diseases_due_to_other_causes',
        id=cid(525),
    ),
    peptic_ulcer_disease=Cause(
        name='peptic_ulcer_disease',
        id=cid(527),
    ),
    pancreatitis=Cause(
        name='pancreatitis',
        id=cid(535),
    ),
    epilepsy=Cause(
        name='epilepsy',
        id=cid(545),
    ),
    alcohol_use_disorders=Cause(
        name='alcohol_use_disorders',
        id=cid(560),
    ),
    opioid_use_disorders=Cause(
        name='opioid_use_disorders',
        id=cid(562),
    ),
    cocaine_use_disorders=Cause(
        name='cocaine_use_disorders',
        id=cid(563),
    ),
    amphetamine_use_disorders=Cause(
        name='amphetamine_use_disorders',
        id=cid(564),
    ),
    cannabis_use_disorders=Cause(
        name='cannabis_use_disorders',
        id=cid(565),
    ),
    other_drug_use_disorders=Cause(
        name='other_drug_use_disorders',
        id=cid(566),
    ),
    major_depressive_disorder=Cause(
        name='major_depressive_disorder',
        id=cid(568),
    ),
    dysthymia=Cause(
        name='dysthymia',
        id=cid(569),
    ),
    idiopathic_developmental_intellectual_disability=Cause(
        name='idiopathic_developmental_intellectual_disability',
        id=cid(582),
    ),
    diabetes_mellitus=Cause(
        name='diabetes_mellitus',
        id=cid(587),
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
    rheumatoid_arthritis=Cause(
        name='rheumatoid_arthritis',
        id=cid(627),
    ),
    osteoarthritis=Cause(
        name='osteoarthritis',
        id=cid(628),
    ),
    low_back_pain=Cause(
        name='low_back_pain',
        id=cid(630),
    ),
    gout=Cause(
        name='gout',
        id=cid(632),
    ),
    cataract=Cause(
        name='cataract',
        id=cid(671),
    ),
    macular_degeneration=Cause(
        name='macular_degeneration',
        id=cid(672),
    ),
    age_related_and_other_hearing_loss=Cause(
        name='age_related_and_other_hearing_loss',
        id=cid(674),
    ),
    pedestrian_road_injuries=Cause(
        name='pedestrian_road_injuries',
        id=cid(690),
    ),
    cyclist_road_injuries=Cause(
        name='cyclist_road_injuries',
        id=cid(691),
    ),
    motorcyclist_road_injuries=Cause(
        name='motorcyclist_road_injuries',
        id=cid(692),
    ),
    motor_vehicle_road_injuries=Cause(
        name='motor_vehicle_road_injuries',
        id=cid(693),
    ),
    other_road_injuries=Cause(
        name='other_road_injuries',
        id=cid(694),
    ),
    other_transport_injuries=Cause(
        name='other_transport_injuries',
        id=cid(695),
    ),
    falls=Cause(
        name='falls',
        id=cid(697),
    ),
    drowning=Cause(
        name='drowning',
        id=cid(698),
    ),
    fire_heat_and_hot_substances=Cause(
        name='fire_heat_and_hot_substances',
        id=cid(699),
    ),
    poisonings=Cause(
        name='poisonings',
        id=cid(700),
    ),
    unintentional_firearm_injuries=Cause(
        name='unintentional_firearm_injuries',
        id=cid(705),
    ),
    unintentional_suffocation=Cause(
        name='unintentional_suffocation',
        id=cid(706),
    ),
    other_exposure_to_mechanical_forces=Cause(
        name='other_exposure_to_mechanical_forces',
        id=cid(707),
    ),
    venomous_animal_contact=Cause(
        name='venomous_animal_contact',
        id=cid(710),
    ),
    non_venomous_animal_contact=Cause(
        name='non_venomous_animal_contact',
        id=cid(711),
    ),
    pulmonary_aspiration_and_foreign_body_in_airway=Cause(
        name='pulmonary_aspiration_and_foreign_body_in_airway',
        id=cid(713),
    ),
    foreign_body_in_eyes=Cause(
        name='foreign_body_in_eyes',
        id=cid(714),
    ),
    foreign_body_in_other_body_part=Cause(
        name='foreign_body_in_other_body_part',
        id=cid(715),
    ),
    other_unintentional_injuries=Cause(
        name='other_unintentional_injuries',
        id=cid(716),
    ),
    self_harm=Cause(
        name='self_harm',
        id=cid(718),
    ),
    assault_by_firearm=Cause(
        name='assault_by_firearm',
        id=cid(725),
    ),
    assault_by_sharp_object=Cause(
        name='assault_by_sharp_object',
        id=cid(726),
    ),
    assault_by_other_means=Cause(
        name='assault_by_other_means',
        id=cid(727),
    ),
    exposure_to_forces_of_nature=Cause(
        name='exposure_to_forces_of_nature',
        id=cid(729),
    ),
)


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
        affected_causes={causes.diarrhea, causes.typhoid_fever, causes.paratyphoid_fever},
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
        affected_causes={causes.diarrhea, causes.typhoid_fever, causes.paratyphoid_fever},
    ),
    ambient_particulate_matter_pollution=Risk(
        name='ambient_particulate_matter_pollution',
        id=rid(86),
        distribution='unknown',
        affected_causes={causes.lower_respiratory_infections, causes.tracheal_bronchus_and_lung_cancer,
                         causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.chronic_obstructive_pulmonary_disease},
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
        affected_causes={causes.lower_respiratory_infections, causes.tracheal_bronchus_and_lung_cancer,
                         causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.chronic_obstructive_pulmonary_disease, causes.cataract},
    ),
    ambient_ozone_pollution=Risk(
        name='ambient_ozone_pollution',
        id=rid(88),
        distribution='unknown',
        affected_causes={causes.chronic_obstructive_pulmonary_disease},
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
        distribution='lognormal',
        affected_causes={causes.tracheal_bronchus_and_lung_cancer},
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
        affected_causes={causes.diarrhea, causes.lower_respiratory_infections, causes.measles,
                         causes.protein_energy_malnutrition},
    ),
    iron_deficiency=Risk(
        name='iron_deficiency',
        id=rid(95),
        affected_causes={causes.maternal_hemorrhage, causes.maternal_sepsis_and_other_infections,
                         causes.iron_deficiency_anemia}
    ),
    vitamin_a_deficiency=Risk(
        name='vitamin_a_deficiency',
        id=rid(96),
        distribution='categorical',
        levels=Levels(
            cat1='deficient',
            cat2='not deficient',
        ),
        affected_causes={causes.diarrhea, causes.measles, causes.vitamin_a_deficiency},
    ),
    zinc_deficiency=Risk(
        name='zinc_deficiency',
        id=rid(97),
        distribution='categorical',
        levels=Levels(
            cat1='deficient',
            cat2='not deficient',
        ),
        affected_causes={causes.diarrhea, causes.lower_respiratory_infections},
    ),
    secondhand_smoke=Risk(
        name='secondhand_smoke',
        id=rid(100),
        distribution='categorical',
        levels=Levels(
            cat1='exposed',
            cat2='unexposed',
        ),
        affected_causes={causes.lower_respiratory_infections, causes.otitis_media,
                         causes.tracheal_bronchus_and_lung_cancer, causes.ischemic_heart_disease,
                         causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.chronic_obstructive_pulmonary_disease},
    ),
    alcohol_use=Risk(
        name='alcohol_use',
        id=rid(102),
        distribution='unknown',
        affected_causes={causes.tuberculosis, causes.lower_respiratory_infections, causes.lip_and_oral_cavity_cancer,
                         causes.nasopharynx_cancer, causes.other_pharynx_cancer, causes.esophageal_cancer,
                         causes.colon_and_rectum_cancer, causes.liver_cancer_due_to_hepatitis_b,
                         causes.liver_cancer_due_to_hepatitis_c, causes.liver_cancer_due_to_alcohol_use,
                         causes.liver_cancer_due_to_other_causes, causes.larynx_cancer, causes.breast_cancer,
                         causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.hypertensive_heart_disease, causes.atrial_fibrillation_and_flutter,
                         causes.cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_b,
                         causes.cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_c,
                         causes.cirrhosis_and_other_chronic_liver_diseases_due_to_alcohol_use,
                         causes.cirrhosis_and_other_chronic_liver_diseases_due_to_other_causes,
                         causes.pancreatitis, causes.epilepsy, causes.alcohol_use_disorders, causes.diabetes_mellitus,
                         causes.pedestrian_road_injuries, causes.cyclist_road_injuries,
                         causes.motorcyclist_road_injuries, causes.motor_vehicle_road_injuries, causes.falls,
                         causes.drowning, causes.fire_heat_and_hot_substances, causes.poisonings,
                         causes.unintentional_firearm_injuries, causes.unintentional_suffocation,
                         causes.other_exposure_to_mechanical_forces, causes.self_harm, causes.assault_by_firearm,
                         causes.assault_by_sharp_object, causes.assault_by_other_means},
        scale=1,
        max_rr=100,
    ),
    high_total_cholesterol=Risk(
        name='high_total_cholesterol',
        id=rid(106),
        distribution='lognormal',
        affected_causes={causes.ischemic_heart_disease, causes.ischemic_stroke},
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
        affected_causes={causes.rheumatic_heart_disease, causes.ischemic_heart_disease, causes.ischemic_stroke,
                         causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                         causes.cardiomyopathy_and_myocarditis, causes.atrial_fibrillation_and_flutter,
                         causes.aortic_aneurysm, causes.peripheral_vascular_disease, causes.endocarditis,
                         causes.other_cardiovascular_and_circulatory_diseases,
                         causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_hypertension,
                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
                         causes.chronic_kidney_disease_due_to_other_causes},
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
        affected_causes={causes.esophageal_cancer, causes.colon_and_rectum_cancer,
                         causes.liver_cancer_due_to_hepatitis_b, causes.liver_cancer_due_to_hepatitis_c,
                         causes.liver_cancer_due_to_alcohol_use, causes.liver_cancer_due_to_other_causes,
                         causes.gallbladder_and_biliary_tract_cancer, causes.pancreatic_cancer, causes.breast_cancer,
                         causes.uterine_cancer, causes.ovarian_cancer, causes.kidney_cancer, causes.thyroid_cancer,
                         causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.hypertensive_heart_disease, causes.diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_hypertension,
                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
                         causes.chronic_kidney_disease_due_to_other_causes, causes.osteoarthritis,
                         causes.low_back_pain},
        tmred=Tmred(
            distribution='uniform',
            min=20,
            max=25,
            inverted=False,
        ),
        scale=5,
        max_rr=50,
    ),
    low_bone_mineral_density=Risk(
        name='low_bone_mineral_density',
        id=rid(109),
        affected_causes={causes.pedestrian_road_injuries, causes.cyclist_road_injuries,
                         causes.motorcyclist_road_injuries, causes.motor_vehicle_road_injuries,
                         causes.other_road_injuries, causes.other_transport_injuries, causes.falls,
                         causes.other_exposure_to_mechanical_forces, causes.non_venomous_animal_contact,
                         causes.assault_by_other_means, causes.exposure_to_forces_of_nature},
    ),
    diet_low_in_fruits=Risk(
        name='diet_low_in_fruits',
        id=rid(111),
        distribution='lognormal',
        affected_causes={causes.lip_and_oral_cavity_cancer, causes.nasopharynx_cancer, causes.other_pharynx_cancer,
                         causes.esophageal_cancer, causes.larynx_cancer, causes.tracheal_bronchus_and_lung_cancer,
                         causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.diabetes_mellitus},
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
        affected_causes={causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke},
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
        affected_causes={causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.diabetes_mellitus},
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
        affected_causes={causes.ischemic_heart_disease, causes.diabetes_mellitus},
        tmred=Tmred(
            distribution='uniform',
            min=16,
            max=25,
            inverted=True,
        ),
        scale=4.05,
        max_rr=24,
    ),
    diet_low_in_milk=Risk(
        name='diet_low_in_milk',
        id=rid(115),
        affected_causes={causes.colon_and_rectum_cancer},
    ),
    diet_high_in_red_meat=Risk(
        name='diet_high_in_red_meat',
        id=rid(116),
        affected_causes={causes.colon_and_rectum_cancer, causes.diabetes_mellitus},
    ),
    diet_high_in_processed_meat=Risk(
        name='diet_high_in_processed_meat',
        id=rid(117),
        distribution='lognormal',
        affected_causes={causes.colon_and_rectum_cancer, causes.ischemic_heart_disease, causes.diabetes_mellitus},
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
        affected_causes={causes.esophageal_cancer, causes.colon_and_rectum_cancer,
                         causes.liver_cancer_due_to_hepatitis_b, causes.liver_cancer_due_to_hepatitis_c,
                         causes.liver_cancer_due_to_alcohol_use, causes.liver_cancer_due_to_other_causes,
                         causes.gallbladder_and_biliary_tract_cancer, causes.pancreatic_cancer, causes.breast_cancer,
                         causes.uterine_cancer, causes.ovarian_cancer, causes.kidney_cancer, causes.thyroid_cancer,
                         causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.hypertensive_heart_disease, causes.diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_hypertension,
                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
                         causes.chronic_kidney_disease_due_to_other_causes, causes.osteoarthritis,
                         causes.low_back_pain},
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
        affected_causes={causes.colon_and_rectum_cancer, causes.ischemic_heart_disease},
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
        affected_causes={causes.ischemic_heart_disease},
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
        affected_causes={causes.ischemic_heart_disease},
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
        affected_causes={causes.ischemic_heart_disease},
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
        affected_causes={causes.stomach_cancer, causes.rheumatic_heart_disease, causes.ischemic_heart_disease,
                         causes.ischemic_stroke, causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                         causes.cardiomyopathy_and_myocarditis, causes.atrial_fibrillation_and_flutter,
                         causes.aortic_aneurysm, causes.peripheral_vascular_disease, causes.endocarditis,
                         causes.other_cardiovascular_and_circulatory_diseases,
                         causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_hypertension,
                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
                         causes.chronic_kidney_disease_due_to_other_causes},
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
        affected_causes={causes.colon_and_rectum_cancer, causes.breast_cancer, causes.ischemic_heart_disease,
                         causes.ischemic_stroke, causes.diabetes_mellitus},
        tmred=Tmred(
            distribution='uniform',
            min=3000,
            max=4500,
            inverted=True,
        ),
        scale=600,
        max_rr=5000,
    ),
    occupational_asthmagens=Risk(
        name='occupational_asthmagens',
        id=rid(128),
        affected_causes={causes.asthma},
    ),
    occupational_particulate_matter_gases_and_fumes=Risk(
        name='occupational_particulate_matter_gases_and_fumes',
        id=rid(129),
        affected_causes={causes.chronic_obstructive_pulmonary_disease, causes.coal_workers_pneumoconiosis}
    ),
    occupational_noise=Risk(
        name='occupational_noise',
        id=rid(130),
        affected_causes={causes.age_related_and_other_hearing_loss},
    ),
    occupational_injuries=Risk(
        name='occupational_injuries',
        id=rid(131),
        affected_causes={causes.pedestrian_road_injuries, causes.cyclist_road_injuries,
                         causes.motorcyclist_road_injuries, causes.motor_vehicle_road_injuries,
                         causes.other_road_injuries, causes.other_transport_injuries, causes.falls, causes.drowning,
                         causes.fire_heat_and_hot_substances, causes.poisonings, causes.unintentional_firearm_injuries,
                         causes.unintentional_suffocation, causes.other_exposure_to_mechanical_forces,
                         causes.venomous_animal_contact, causes.non_venomous_animal_contact,
                         causes.pulmonary_aspiration_and_foreign_body_in_airway, causes.foreign_body_in_eyes,
                         causes.foreign_body_in_other_body_part, causes.other_unintentional_injuries},
    ),
    occupational_ergonomic_factors=Risk(
        name='occupational_ergonomic_factors',
        id=rid(132),
        affected_causes={causes.low_back_pain}
    ),
    non_exclusive_breastfeeding=Risk(
        name='non_exclusive_breastfeeding',
        id=rid(136),
        distribution='categorical',
        affected_causes={causes.diarrhea, causes.lower_respiratory_infections},
    ),
    discontinued_breastfeeding=Risk(
        name='discontinued_breastfeeding',
        id=rid(137),
        distribution='categorical',
        affected_causes={causes.diarrhea},
    ),
    drug_use_dependence_and_blood_borne_viruses=Risk(
        name='drug_use_dependence_and_blood_borne_viruses',
        id=rid(138),
        affected_causes={causes.hiv_aids_tuberculosis, causes.hiv_aids_other_diseases, causes.hepatitis_b,
                         causes.hepatitis_c, causes.liver_cancer_due_to_hepatitis_b,
                         causes.liver_cancer_due_to_hepatitis_c,
                         causes.cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_b,
                         causes.cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_c,
                         causes.opioid_use_disorders, causes.cocaine_use_disorders, causes.amphetamine_use_disorders,
                         causes.cannabis_use_disorders, causes.other_drug_use_disorders},
    ),
    suicide_due_to_drug_use_disorders=Risk(
        name='suicide_due_to_drug_use_disorders',
        id=rid(140),
        affected_causes={causes.self_harm},
    ),
    high_fasting_plasma_glucose_continuous=Risk(
        name='high_fasting_plasma_glucose_continuous',
        id=rid(141),
        distribution='lognormal',
        affected_causes={causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.diabetes_mellitus, causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_hypertension,
                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
                         causes.chronic_kidney_disease_due_to_other_causes},
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
        affected_causes={causes.tuberculosis, causes.peripheral_vascular_disease},
    ),
    low_glomerular_filtration_rate=Risk(
        name='low_glomerular_filtration_rate',
        id=rid(143),
        distribution='unknown',
        affected_causes={causes.ischemic_heart_disease, causes.ischemic_stroke, causes.hemorrhagic_stroke,
                         causes.peripheral_vascular_disease, causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_hypertension,
                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
                         causes.chronic_kidney_disease_due_to_other_causes, causes.gout},
    ),
    occupational_exposure_to_asbestos=Risk(
        name='occupational_exposure_to_asbestos',
        id=rid(150),
        affected_causes={causes.larynx_cancer, causes.tracheal_bronchus_and_lung_cancer, causes.ovarian_cancer,
                         causes.mesothelioma},
    ),
    occupational_exposure_to_arsenic=Risk(
        name='occupational_exposure_to_arsenic',
        id=rid(151),
        affected_causes={causes.tracheal_bronchus_and_lung_cancer},
    ),
    occupational_exposure_to_beryllium=Risk(
        name='occupational_exposure_to_beryllium',
        id=rid(153),
        affected_causes={causes.tracheal_bronchus_and_lung_cancer},
    ),
    occupational_exposure_to_cadmium=Risk(
        name='occupational_exposure_to_cadmium',
        id=rid(154),
        affected_causes={causes.tracheal_bronchus_and_lung_cancer},
    ),
    occupational_exposure_to_chromium=Risk(
        name='occupational_exposure_to_chromium',
        id=rid(155),
        affected_causes={causes.tracheal_bronchus_and_lung_cancer},
    ),
    occupational_exposure_to_diesel_engine_exhaust=Risk(
        name='occupational_exposure_to_diesel_engine_exhaust',
        id=rid(156),
        affected_causes={causes.tracheal_bronchus_and_lung_cancer},
    ),
    occupational_exposure_to_secondhand_smoke=Risk(
        name='occupational_exposure_to_secondhand_smoke',
        id=rid(157),
        affected_causes={causes.tracheal_bronchus_and_lung_cancer},
    ),
    occupational_exposure_to_formaldehyde=Risk(
        name='occupational_exposure_to_formaldehyde',
        id=rid(158),
        affected_causes={causes.nasopharynx_cancer},
    ),
    occupational_exposure_to_nickel=Risk(
        name='occupational_exposure_to_nickel',
        id=rid(159),
        affected_causes={causes.tracheal_bronchus_and_lung_cancer},
    ),
    occupational_exposure_to_polycyclic_aromatic_hydrocarbons=Risk(
        name='occupational_exposure_to_polycyclic_aromatic_hydrocarbons',
        id=rid(160),
        affected_causes={causes.tracheal_bronchus_and_lung_cancer},
    ),
    occupational_exposure_to_silica=Risk(
        name='occupational_exposure_to_silica',
        id=rid(161),
        affected_causes={causes.tracheal_bronchus_and_lung_cancer},
    ),
    occupational_exposure_to_sulfuric_acid=Risk(
        name='occupational_exposure_to_sulfuric_acid',
        id=rid(162),
        affected_causes={causes.larynx_cancer},
    ),
    smoking_sir_approach=Risk(
        name='smoking_sir_approach',
        id=rid(165),
        affected_causes={causes.lip_and_oral_cavity_cancer, causes.nasopharynx_cancer, causes.esophageal_cancer,
                         causes.stomach_cancer, causes.colon_and_rectum_cancer, causes.liver_cancer_due_to_hepatitis_b,
                         causes.liver_cancer_due_to_hepatitis_c, causes.liver_cancer_due_to_alcohol_use,
                         causes.liver_cancer_due_to_other_causes, causes.pancreatic_cancer, causes.larynx_cancer,
                         causes.tracheal_bronchus_and_lung_cancer, causes.cervical_cancer, causes.kidney_cancer,
                         causes.bladder_cancer, causes.chronic_obstructive_pulmonary_disease, causes.silicosis,
                         causes.asbestosis, causes.coal_workers_pneumoconiosis, causes.other_pneumoconiosis,
                         causes.interstitial_lung_disease_and_pulmonary_sarcoidosis,
                         causes.other_chronic_respiratory_diseases},
    ),
    smoking_prevalence_approach=Risk(
        name='smoking_prevalence_approach',
        id=rid(166),
        distribution='categorical',
        affected_causes={causes.tuberculosis, causes.lower_respiratory_infections, causes.ischemic_heart_disease,
                         causes.ischemic_stroke, causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                         causes.atrial_fibrillation_and_flutter, causes.aortic_aneurysm,
                         causes.peripheral_vascular_disease, causes.other_cardiovascular_and_circulatory_diseases,
                         causes.asthma, causes.peptic_ulcer_disease, causes.diabetes_mellitus,
                         causes.rheumatoid_arthritis, causes.cataract, causes.macular_degeneration,
                         causes.pedestrian_road_injuries, causes.cyclist_road_injuries,
                         causes.motorcyclist_road_injuries, causes.motor_vehicle_road_injuries,
                         causes.other_road_injuries, causes.other_transport_injuries, causes.falls,
                         causes.other_exposure_to_mechanical_forces, causes.non_venomous_animal_contact,
                         causes.assault_by_other_means, causes.exposure_to_forces_of_nature},
    ),
    intimate_partner_violence_exposure_approach=Risk(
        name='intimate_partner_violence_exposure_approach',
        id=rid(167),
        affected_causes={causes.maternal_abortion_miscarriage_and_ectopic_pregnancy, causes.major_depressive_disorder,
                         causes.dysthymia, causes.self_harm, causes.assault_by_firearm, causes.assault_by_sharp_object,
                         causes.assault_by_other_means},
    ),
    intimate_partner_violence_direct_paf_approach=Risk(
        name='intimate_partner_violence_direct_paf_approach',
        id=rid(168),
        affected_causes={causes.assault_by_firearm, causes.assault_by_sharp_object, causes.assault_by_other_means},
    ),
    unsafe_sex=Risk(
        name='unsafe_sex',
        id=rid(170),
        affected_causes={causes.hiv_aids_tuberculosis, causes.hiv_aids_other_diseases, causes.syphilis,
                         causes.chlamydia, causes.gonorrhea, causes.trichomoniasis, causes.genital_herpes,
                         causes.other_sexually_transmitted_disease, causes.cervical_cancer},
    ),
    intimate_partner_violence_hiv_paf_approach=Risk(
        name='intimate_partner_violence_hiv_paf_approach',
        id=rid(201),
        affected_causes={causes.hiv_aids_tuberculosis, causes.hiv_aids_other_diseases},
    ),
    occupational_exposure_to_trichloroethylene=Risk(
        name='occupational_exposure_to_trichloroethylene',
        id=rid(237),
        affected_causes={causes.kidney_cancer},
    ),
    no_handwashing_with_soap=Risk(
        name='no_handwashing_with_soap',
        id=rid(238),
        affected_causes={causes.diarrhea, causes.typhoid_fever, causes.paratyphoid_fever,
                         causes.lower_respiratory_infections},
    ),
    childhood_wasting=Risk(
        name='childhood_wasting',
        id=rid(240),
        affected_causes={causes.diarrhea, causes.lower_respiratory_infections, causes.measles,
                         causes.protein_energy_malnutrition},
    ),
    childhood_stunting=Risk(
        name='childhood_stunting',
        id=rid(241),
        affected_causes={causes.diarrhea, causes.lower_respiratory_infections, causes.measles},
    ),
    lead_exposure_in_blood=Risk(
        name='lead_exposure_in_blood',
        id=rid(242),
        affected_causes={causes.idiopathic_developmental_intellectual_disability},
    ),
    lead_exposure_in_bone=Risk(
        name='lead_exposure_in_bone',
        id=rid(243),
        distribution='lognormal',
        affected_causes={causes.rheumatic_heart_disease, causes.ischemic_heart_disease, causes.ischemic_stroke,
                         causes.hemorrhagic_stroke, causes.hypertensive_heart_disease,
                         causes.cardiomyopathy_and_myocarditis, causes.atrial_fibrillation_and_flutter,
                         causes.aortic_aneurysm, causes.peripheral_vascular_disease, causes.endocarditis,
                         causes.other_cardiovascular_and_circulatory_diseases,
                         causes.chronic_kidney_disease_due_to_diabetes_mellitus,
                         causes.chronic_kidney_disease_due_to_hypertension,
                         causes.chronic_kidney_disease_due_to_glomerulonephritis,
                         causes.chronic_kidney_disease_due_to_other_causes},
        tmred=Tmred(
            distribution='uniform',
            min=0,
            max=20,
            inverted=False,
        ),
        scale=10,
        max_rr=500,
    ),
    childhood_sexual_abuse_against_females=Risk(
        name='childhood_sexual_abuse_against_females',
        id=rid(244),
        affected_causes={causes.alcohol_use_disorders, causes.major_depressive_disorder, causes.dysthymia,
                         causes.self_harm},
    ),
    childhood_sexual_abuse_against_males=Risk(
        name='childhood_sexual_abuse_against_males',
        id=rid(245),
        affected_causes={causes.alcohol_use_disorders, causes.major_depressive_disorder, causes.dysthymia,
                         causes.self_harm},
    ),
)
