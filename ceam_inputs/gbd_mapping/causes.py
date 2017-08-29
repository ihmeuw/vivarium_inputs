"""Mapping of GBD ids onto vivarium conventions."""
from .templates import (Etiology, Etioloties, Sequela, Sequelae, Cause, Causes,
                        SeveritySplit, SeveritySplits, cid, sid, meid, rid, hid, scalar)

etiologies = Etioloties(
    unattributed_diarrhea=Etiology(
        name='unattributed_diarrhea',
        gbd_id=None,
        prevalence=None,
        disability_weight=None,
    ),
    cholera=Etiology(
        name='cholera',
        gbd_id=rid(173),
        prevalence=None,
        disability_weight=None,
    ),
    other_salmonella=Etiology(
        name='other_salmonella',
        gbd_id=rid(174),
        prevalence=None,
        disability_weight=None,
    ),
    shigellosis=Etiology(
        name='shigellosis',
        gbd_id=rid(175),
        prevalence=None,
        disability_weight=None,
    ),
    EPEC=Etiology(
        name='EPEC',
        gbd_id=rid(176),
        prevalence=None,
        disability_weight=None,
    ),
    ETEC=Etiology(
        name='ETEC',
        gbd_id=rid(177),
        prevalence=None,
        disability_weight=None,
    ),
    campylobacter=Etiology(
        name='campylobacter',
        gbd_id=rid(178),
        prevalence=None,
        disability_weight=None,
    ),
    amoebiasis=Etiology(
        name='amoebiasis',
        gbd_id=rid(179),
        prevalence=None,
        disability_weight=None,
    ),
    cryptosporidiosis=Etiology(
        name='cryptosporidiosis',
        gbd_id=rid(180),
        prevalence=None,
        disability_weight=None,
    ),
    rotaviral_entiritis=Etiology(
        name='rotaviral_entiritis',
        gbd_id=rid(181),
        prevalence=None,
        disability_weight=None,
    ),
    aeromonas=Etiology(
        name='aeromonas',
        gbd_id=rid(182),
        prevalence=None,
        disability_weight=None,
    ),
    clostridium_difficile=Etiology(
        name='clostridium_difficile',
        gbd_id=rid(183),
        prevalence=None,
        disability_weight=None,
    ),
    norovirus=Etiology(
        name='norovirus',
        gbd_id=rid(184),
        prevalence=None,
        disability_weight=None,
    ),
    adenovirus=Etiology(
        name='adenovirus',
        gbd_id=rid(185),
        prevalence=None,
        disability_weight=None,
    ),
)

sequelae = Sequelae(
    heart_attack=Sequela(
        name='heart_attack',
        gbd_id=meid(1814),
        incidence=meid(1814),
        prevalence=meid(1814),
        excess_mortality=meid(1814),
        duration=scalar(28),
    ),
    acute_myocardial_infarction_first_2_days=Sequela(
        name='acute_myocardial_infarction_first_2_days',
        gbd_id=sid(378),
        prevalence=meid(1815),
    ),
    acute_myocardial_infarction_3_to_28_days=Sequela(
        name='acute_myocardial_infarction_3_to_28_days',
        gbd_id=sid(379),
        prevalence=meid(1816),
    ),
    heart_failure=Sequela(
        name='heart_failure',
        gbd_id=meid(2412),
        incidence=meid(2412),
        # Post MI proportion, I think?  Based on Everett's discussion with modelers. Use with caution.
        proportion=meid(2414),
        excess_mortality=None,
        severity_splits=SeveritySplits(
            # Split data obtained from Catherine Johnson (johnsoco@uw.edu)
            mild=SeveritySplit(
                name='mild_heart_failure',
                gbd_id=meid(1821),
                proportion=scalar(0.182074),
                prevalence=meid(1821),
                disability_weight=meid(1821),
                excess_mortality=meid(2412),
            ),
            moderate=SeveritySplit(
                name='moderate_heart_failure',
                gbd_id=meid(1822),
                proportion=scalar(0.149771),
                prevalence=meid(1822),
                disability_weight=meid(1822),
                excess_mortality=meid(2412),
            ),
            severe=SeveritySplit(
                name='severe_heart_failure',
                gbd_id=meid(1823),
                proportion=scalar(0.402838),
                prevalence=meid(1823),
                disability_weight=meid(1823),
                excess_mortality=meid(2412),
            ),
        ),
    ),
    angina=Sequela(
        name='angina',
        gbd_id=meid(1817),
        incidence=meid(1817),
        excess_mortality=None,
        severity_splits=SeveritySplits(
            # Split data obtained from Catherine Johnson (johnsoco@uw.edu)
            asymptomatic=SeveritySplit(
                name='asymptomatic_angina',
                gbd_id=meid(3102),
                proportion=scalar(0.304553),
                prevalence=meid(3102),
                disability_weight=meid(3102),
                excess_mortality=meid(1817),
            ),
            mild=SeveritySplit(
                name='mild_angina',
                gbd_id=meid(1818),
                proportion=scalar(0.239594),
                prevalence=meid(1818),
                disability_weight=meid(1818),
                excess_mortality=meid(1817),
            ),
            moderate=SeveritySplit(
                name='moderate_angina',
                gbd_id=meid(1819),
                proportion=scalar(0.126273),
                prevalence=meid(1819),
                disability_weight=meid(1819),
                excess_mortality=meid(1817),
            ),
            severe=SeveritySplit(
                name='severe_angina',
                gbd_id=meid(1820),
                proportion=scalar(0.32958),
                prevalence=meid(1820),
                disability_weight=meid(1820),
                excess_mortality=meid(1817),
            ),
        ),
    ),
    asymptomatic_ihd=Sequela(
        name='asymptomatic_ihd',
        gbd_id=meid(3233),
        prevalence=meid(3233),
        excess_mortality=scalar(0),
        disability_weight=meid(3233),
    ),
    chronic_ischemic_stroke=Sequela(
        name='chronic_ischemic_stroke',
        gbd_id=None,
        severity_splits=SeveritySplits(
            asymptomatic=SeveritySplit(
                name='asymptomatic_chronic_ischemic_stroke',
                gbd_id=sid(946),
                prevalence=meid(3095),
            ),
            level_1=SeveritySplit(
                name='chronic_ischemic_stroke_level_1',
                gbd_id=sid(391),
                prevalence=meid(1833),
            ),
            level_2=SeveritySplit(
                name='chronic_ischemic_stroke_level_2',
                gbd_id=sid(392),
                prevalence=meid(1834),
            ),
            level_3=SeveritySplit(
                name='chronic_ischemic_stroke_level_3',
                gbd_id=sid(393),
                prevalence=meid(1835),
            ),
            level_4=SeveritySplit(
                name='chronic_ischemic_stroke_level_4',
                gbd_id=sid(394),
                prevalence=meid(1836),
            ),
            level_5=SeveritySplit(
                name='chronic_ischemic_stroke_level_5',
                gbd_id=sid(395),
                prevalence=meid(1837),
            ),
        ),
    ),
    acute_ischemic_stroke=Sequela(
        name='acute_ischemic_stroke',
        gbd_id=None,
        severity_splits=SeveritySplits(
            level_1=SeveritySplit(
                name='acute_ischemic_stroke_level_1',
                gbd_id=sid(386),
                prevalence=meid(1827),
            ),
            level_2=SeveritySplit(
                name='acute_ischemic_stroke_level_2',
                gbd_id=sid(387),
                prevalence=meid(1828),
            ),
            level_3=SeveritySplit(
                name='acute_ischemic_stroke_level_3',
                gbd_id=sid(388),
                prevalence=meid(1829),
            ),
            level_4=SeveritySplit(
                name='acute_ischemic_stroke_level_4',
                gbd_id=sid(389),
                prevalence=meid(1830),
            ),
            level_5=SeveritySplit(
                name='acute_ischemic_stroke_level_5',
                gbd_id=sid(390),
                prevalence=meid(1831),
            ),
        ),
    ),
    chronic_hemorrhagic_stroke=Sequela(
        name='chronic_hemorrhagic_stroke',
        gbd_id=None,
        severity_splits=SeveritySplits(
            asymptomatic=SeveritySplit(
                name='asymptomatic_chronic_hemorrhagic_stroke',
                gbd_id=sid(947),
                prevalence=meid(3096),
            ),
            level_1=SeveritySplit(
                name='chronic_hemorrhagic_stroke_level_1',
                gbd_id=sid(401),
                prevalence=meid(1845),
            ),
            level_2=SeveritySplit(
                name='chronic_hemorrhagic_stroke_level_2',
                gbd_id=sid(402),
                prevalence=meid(1846),
            ),
            level_3=SeveritySplit(
                name='chronic_hemorrhagic_stroke_level_3',
                gbd_id=sid(403),
                prevalence=meid(1847),
            ),
            level_4=SeveritySplit(
                name='chronic_hemorrhagic_stroke_level_4',
                gbd_id=sid(404),
                prevalence=meid(1848),
            ),
            level_5=SeveritySplit(
                name='chronic_hemorrhagic_stroke_level_5',
                gbd_id=sid(405),
                prevalence=meid(1849),
            ),
        ),
    ),
    acute_hemorrhagic_stroke=Sequela(
        name='acute_hemorrhagic_stroke',
        gbd_id=None,
        severity_splits=SeveritySplits(
            level_1=SeveritySplit(
                name='acute_hemorrhagic_stroke_level_1',
                gbd_id=sid(396),
                prevalence=meid(1839),
            ),
            level_2=SeveritySplit(
                name='acute_hemorrhagic_stroke_level_2',
                gbd_id=sid(397),
                prevalence=meid(1840),
            ),
            level_3=SeveritySplit(
                name='acute_hemorrhagic_stroke_level_3',
                gbd_id=sid(398),
                prevalence=meid(1841),
            ),
            level_4=SeveritySplit(
                name='acute_hemorrhagic_stroke_level_4',
                gbd_id=sid(399),
                prevalence=meid(1842),
            ),
            level_5=SeveritySplit(
                name='acute_hemorrhagic_stroke_level_5',
                gbd_id=sid(400),
                prevalence=meid(1843),
            ),
        ),
    ),
)


causes = Causes(
    all_causes=Cause(
        name='all_causes',
        gbd_id=cid(294),
        csmr=cid(294),
    ),
    tuberculosis=Cause(
        name='tuberculosis',
        gbd_id=cid(297),
    ),
    hiv_aids_tuberculosis=Cause(
        name='hiv_aids_tuberculosis',
        gbd_id=cid(299),
    ),
    hiv_aids_other_diseases=Cause(
        name='hiv_aids_other_diseases',
        gbd_id=cid(300),
    ),
    diarrhea=Cause(
        name='diarrhea',
        gbd_id=cid(302),
        incidence=meid(1181),
        prevalence=meid(1181),
        csmr=cid(302),
        excess_mortality=meid(1181),
        disability_weight=scalar(0.23),
        remission=meid(1181),
        severity_splits=SeveritySplits(
            mild=SeveritySplit(
                name='mild_diarrhea',
                gbd_id=meid(2608),
                proportion=meid(2608),
                disability_weight=hid(355),
                remission=meid(1181),
            ),
            moderate=SeveritySplit(
                name='moderate_diarrhea',
                gbd_id=meid(2609),
                proportion=meid(2609),
                disability_weight=hid(356),
                remission=meid(1181),
            ),
            severe=SeveritySplit(
                name='severe_diarrhea',
                gbd_id=meid(2610),
                proportion=meid(2610),
                disability_weight=hid(357),
                remission=meid(1181),
            ),
        ),
        etiologies=(etiologies.adenovirus, etiologies.aeromonas, etiologies.amoebiasis, etiologies.campylobacter,
                    etiologies.cholera, etiologies.clostridium_difficile, etiologies.cryptosporidiosis,
                    etiologies.EPEC, etiologies.ETEC, etiologies.norovirus, etiologies.other_salmonella,
                    etiologies.rotaviral_entiritis, etiologies.shigellosis, etiologies.unattributed_diarrhea),
    ),
    typhoid_fever=Cause(
        name='typhoid_fever',
        gbd_id=cid(319),
    ),
    paratyphoid_fever=Cause(
        name='paratyphoid_fever',
        gbd_id=cid(320),
    ),
    lower_respiratory_infections=Cause(
        name='lower_respiratory_infections',
        gbd_id=cid(322),
    ),
    otitis_media=Cause(
        name='otitis_media',
        gbd_id=cid(329),
    ),
    measles=Cause(
        name='measles',
        gbd_id=cid(341),
    ),
    maternal_hemorrhage=Cause(
        name='maternal_hemorrhage',
        gbd_id=cid(367),
    ),
    maternal_sepsis_and_other_infections=Cause(
        name='maternal_sepsis_and_other_infections',
        gbd_id=cid(368),
    ),
    maternal_abortion_miscarriage_and_ectopic_pregnancy=Cause(
        name='maternal_abortion_miscarriage_and_ectopic_pregnancy',
        gbd_id=cid(371),
    ),
    protein_energy_malnutrition=Cause(
        name='protein_energy_malnutrition',
        gbd_id=cid(387),
    ),
    vitamin_a_deficiency=Cause(
        name='vitamin_a_deficiency',
        gbd_id=cid(389),
    ),
    iron_deficiency_anemia=Cause(
        name='iron_deficiency_anemia',
        gbd_id=cid(390),
    ),
    syphilis=Cause(
        name='syphilis',
        gbd_id=cid(394),
    ),
    chlamydia=Cause(
        name='chlamydia',
        gbd_id=cid(395),
    ),
    gonorrhea=Cause(
        name='gonorrhea',
        gbd_id=cid(396),
    ),
    trichomoniasis=Cause(
        name='trichomoniasis',
        gbd_id=cid(397),
    ),
    genital_herpes=Cause(
        name='genital_herpes',
        gbd_id=cid(398),
    ),
    other_sexually_transmitted_disease=Cause(
        name='other_sexually_transmitted_disease',
        gbd_id=cid(399),
    ),
    hepatitis_b=Cause(
        name='hepatitis_b',
        gbd_id=cid(402),
    ),
    hepatitis_c=Cause(
        name='hepatitis_c',
        gbd_id=cid(403),
    ),
    esophageal_cancer=Cause(
        name='esophageal_cancer',
        gbd_id=cid(411),
    ),
    stomach_cancer=Cause(
        name='stomach_cancer',
        gbd_id=cid(414),
    ),
    liver_cancer_due_to_hepatitis_b=Cause(
        name='liver_cancer_due_to_hepatitis_b',
        gbd_id=cid(418),
    ),
    liver_cancer_due_to_hepatitis_c=Cause(
        name='liver_cancer_due_to_hepatitis_c',
        gbd_id=cid(419),
    ),
    liver_cancer_due_to_alcohol_use=Cause(
        name='liver_cancer_due_to_alcohol_use',
        gbd_id=cid(420),
    ),
    liver_cancer_due_to_other_causes=Cause(
        name='liver_cancer_due_to_other_causes',
        gbd_id=cid(421)
    ),
    larynx_cancer=Cause(
        name='larynx_cancer',
        gbd_id=cid(423)
    ),
    tracheal_bronchus_and_lung_cancer=Cause(
        name='tracheal_bronchus_and_lung_cancer',
        gbd_id=cid(426),
    ),
    breast_cancer=Cause(
        name='breast_cancer',
        gbd_id=cid(429),
    ),
    cervical_cancer=Cause(
        name='cervical_cancer',
        gbd_id=cid(432),
    ),
    uterine_cancer=Cause(
        name='uterine_cancer',
        gbd_id=cid(435),
    ),
    colon_and_rectum_cancer=Cause(
        name='colon_and_rectum_cancer',
        gbd_id=cid(441),
    ),
    lip_and_oral_cavity_cancer=Cause(
        name='lip_and_oral_cavity_cancer',
        gbd_id=cid(444),
    ),
    nasopharynx_cancer=Cause(
        name='nasopharynx_cancer',
        gbd_id=cid(447),
    ),
    other_pharynx_cancer=Cause(
        name='other_pharynx_cancer',
        gbd_id=cid(450),
    ),
    gallbladder_and_biliary_tract_cancer=Cause(
        name='gallbladder_and_biliary_tract_cancer',
        gbd_id=cid(453),
    ),
    pancreatic_cancer=Cause(
        name='pancreatic_cancer',
        gbd_id=cid(456),
    ),
    ovarian_cancer=Cause(
        name='ovarian_cancer',
        gbd_id=cid(465)
    ),
    kidney_cancer=Cause(
        name='kidney_cancer',
        gbd_id=cid(471)
    ),
    bladder_cancer=Cause(
        name='bladder_cancer',
        gbd_id=cid(474),
    ),
    thyroid_cancer=Cause(
        name='thyroid_cancer',
        gbd_id=cid(480),
    ),
    mesothelioma=Cause(
        name='mesothelioma',
        gbd_id=cid(483),
    ),
    rheumatic_heart_disease=Cause(
        name='rheumatic_heart_disease',
        gbd_id=cid(492),
    ),
    ischemic_heart_disease=Cause(
        name='ischemic_heart_disease',
        gbd_id=cid(493),
        csmr=cid(493),
        excess_mortality=cid(493),
        prevalence=cid(493),
        incidence=cid(493),
        disability_weight=scalar(0),
        sequelae=(sequelae.heart_attack, sequelae.angina, sequelae.asymptomatic_ihd, sequelae.heart_failure),
    ),
    chronic_stroke=Cause(
        name='chronic_stroke',
        gbd_id=cid(494),
        prevalence=cid(494),
        csmr=cid(494),
        excess_mortality=meid(9312),
        disability_weight=scalar(0.32),
    ),
    ischemic_stroke=Cause(
        name='ischemic_stroke',
        gbd_id=cid(495),
        incidence=meid(9310),
        prevalence=meid(9310),
        csmr=cid(495),
        excess_mortality=meid(9310),
        disability_weight=scalar(0.32),
        duration=scalar(28),
        sequelae=(sequelae.acute_ischemic_stroke, sequelae.chronic_ischemic_stroke),
    ),
    hemorrhagic_stroke=Cause(
        name='hemorrhagic_stroke',
        gbd_id=cid(496),
        incidence=meid(9311),
        prevalence=meid(9311),
        csmr=cid(496),
        excess_mortality=meid(9311),
        disability_weight=scalar(0.32),
        duration=scalar(28),
        sequelae=(sequelae.acute_hemorrhagic_stroke, sequelae.chronic_hemorrhagic_stroke)
    ),
    hypertensive_heart_disease=Cause(
        name='hypertensive_heart_disease',
        gbd_id=cid(498),
    ),
    cardiomyopathy_and_myocarditis=Cause(
        name='cardiomyopathy_and_myocarditis',
        gbd_id=cid(499),
    ),
    atrial_fibrillation_and_flutter=Cause(
        name='atrial_fibrillation_and_flutter',
        gbd_id=cid(500),
    ),
    aortic_aneurysm=Cause(
        name='aortic_aneurysm',
        gbd_id=cid(501),
    ),
    peripheral_vascular_disease=Cause(
        name='peripheral_vascular_disease',
        gbd_id=cid(502),
    ),
    endocarditis=Cause(
        name='endocarditis',
        gbd_id=cid(503),
    ),
    other_cardiovascular_and_circulatory_diseases=Cause(
        name='other_cardiovascular_and_circulatory_diseases',
        gbd_id=cid(507),
    ),
    chronic_obstructive_pulmonary_disease=Cause(
        name='chronic_obstructive_pulmonary_disease',
        gbd_id=cid(509),
    ),
    silicosis=Cause(
        name='silicosis',
        gbd_id=cid(511),
    ),
    asbestosis=Cause(
        name='asbestosis',
        gbd_id=cid(512),
    ),
    coal_workers_pneumoconiosis=Cause(
        name='coal_workers_pneumoconiosis',
        gbd_id=cid(513),
    ),
    other_pneumoconiosis=Cause(
        name='other_pneumoconiosis',
        gbd_id=cid(514)
    ),
    asthma=Cause(
        name='asthma',
        gbd_id=cid(515),
    ),
    interstitial_lung_disease_and_pulmonary_sarcoidosis=Cause(
        name='interstitial_lung_disease_and_pulmonary_sarcoidosis',
        gbd_id=cid(516),
    ),
    other_chronic_respiratory_diseases=Cause(
        name='other_chronic_respiratory_diseases',
        gbd_id=cid(520),
    ),
    cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_b=Cause(
        name='cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_b',
        gbd_id=cid(522),
    ),
    cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_c=Cause(
        name='cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_c',
        gbd_id=cid(523),
    ),
    cirrhosis_and_other_chronic_liver_diseases_due_to_alcohol_use=Cause(
        name='cirrhosis_and_other_chronic_liver_diseases_due_to_alcohol_use',
        gbd_id=cid(524),
    ),
    cirrhosis_and_other_chronic_liver_diseases_due_to_other_causes=Cause(
        name='cirrhosis_and_other_chronic_liver_diseases_due_to_other_causes',
        gbd_id=cid(525),
    ),
    peptic_ulcer_disease=Cause(
        name='peptic_ulcer_disease',
        gbd_id=cid(527),
    ),
    pancreatitis=Cause(
        name='pancreatitis',
        gbd_id=cid(535),
    ),
    epilepsy=Cause(
        name='epilepsy',
        gbd_id=cid(545),
    ),
    alcohol_use_disorders=Cause(
        name='alcohol_use_disorders',
        gbd_id=cid(560),
    ),
    opioid_use_disorders=Cause(
        name='opioid_use_disorders',
        gbd_id=cid(562),
    ),
    cocaine_use_disorders=Cause(
        name='cocaine_use_disorders',
        gbd_id=cid(563),
    ),
    amphetamine_use_disorders=Cause(
        name='amphetamine_use_disorders',
        gbd_id=cid(564),
    ),
    cannabis_use_disorders=Cause(
        name='cannabis_use_disorders',
        gbd_id=cid(565),
    ),
    other_drug_use_disorders=Cause(
        name='other_drug_use_disorders',
        gbd_id=cid(566),
    ),
    major_depressive_disorder=Cause(
        name='major_depressive_disorder',
        gbd_id=cid(568),
    ),
    dysthymia=Cause(
        name='dysthymia',
        gbd_id=cid(569),
    ),
    idiopathic_developmental_intellectual_disability=Cause(
        name='idiopathic_developmental_intellectual_disability',
        gbd_id=cid(582),
    ),
    diabetes_mellitus=Cause(
        name='diabetes_mellitus',
        gbd_id=cid(587),
    ),
    chronic_kidney_disease_due_to_diabetes_mellitus=Cause(
        name='chronic_kidney_disease_due_to_diabetes_mellitus',
        gbd_id=cid(590),
    ),
    chronic_kidney_disease_due_to_hypertension=Cause(
        name='chronic_kidney_disease_due_to_hypertension',
        gbd_id=cid(591),
    ),
    chronic_kidney_disease_due_to_glomerulonephritis=Cause(
        name='chronic_kidney_disease_due_to_glomerulonephritis',
        gbd_id=cid(592),
    ),
    chronic_kidney_disease_due_to_other_causes=Cause(
        name='chronic_kidney_disease_due_to_other_causes',
        gbd_id=cid(593),
    ),
    rheumatoid_arthritis=Cause(
        name='rheumatoid_arthritis',
        gbd_id=cid(627),
    ),
    osteoarthritis=Cause(
        name='osteoarthritis',
        gbd_id=cid(628),
    ),
    low_back_pain=Cause(
        name='low_back_pain',
        gbd_id=cid(630),
    ),
    gout=Cause(
        name='gout',
        gbd_id=cid(632),
    ),
    cataract=Cause(
        name='cataract',
        gbd_id=cid(671),
    ),
    macular_degeneration=Cause(
        name='macular_degeneration',
        gbd_id=cid(672),
    ),
    age_related_and_other_hearing_loss=Cause(
        name='age_related_and_other_hearing_loss',
        gbd_id=cid(674),
    ),
    pedestrian_road_injuries=Cause(
        name='pedestrian_road_injuries',
        gbd_id=cid(690),
    ),
    cyclist_road_injuries=Cause(
        name='cyclist_road_injuries',
        gbd_id=cid(691),
    ),
    motorcyclist_road_injuries=Cause(
        name='motorcyclist_road_injuries',
        gbd_id=cid(692),
    ),
    motor_vehicle_road_injuries=Cause(
        name='motor_vehicle_road_injuries',
        gbd_id=cid(693),
    ),
    other_road_injuries=Cause(
        name='other_road_injuries',
        gbd_id=cid(694),
    ),
    other_transport_injuries=Cause(
        name='other_transport_injuries',
        gbd_id=cid(695),
    ),
    falls=Cause(
        name='falls',
        gbd_id=cid(697),
    ),
    drowning=Cause(
        name='drowning',
        gbd_id=cid(698),
    ),
    fire_heat_and_hot_substances=Cause(
        name='fire_heat_and_hot_substances',
        gbd_id=cid(699),
    ),
    poisonings=Cause(
        name='poisonings',
        gbd_id=cid(700),
    ),
    unintentional_firearm_injuries=Cause(
        name='unintentional_firearm_injuries',
        gbd_id=cid(705),
    ),
    unintentional_suffocation=Cause(
        name='unintentional_suffocation',
        gbd_id=cid(706),
    ),
    other_exposure_to_mechanical_forces=Cause(
        name='other_exposure_to_mechanical_forces',
        gbd_id=cid(707),
    ),
    venomous_animal_contact=Cause(
        name='venomous_animal_contact',
        gbd_id=cid(710),
    ),
    non_venomous_animal_contact=Cause(
        name='non_venomous_animal_contact',
        gbd_id=cid(711),
    ),
    pulmonary_aspiration_and_foreign_body_in_airway=Cause(
        name='pulmonary_aspiration_and_foreign_body_in_airway',
        gbd_id=cid(713),
    ),
    foreign_body_in_eyes=Cause(
        name='foreign_body_in_eyes',
        gbd_id=cid(714),
    ),
    foreign_body_in_other_body_part=Cause(
        name='foreign_body_in_other_body_part',
        gbd_id=cid(715),
    ),
    other_unintentional_injuries=Cause(
        name='other_unintentional_injuries',
        gbd_id=cid(716),
    ),
    self_harm=Cause(
        name='self_harm',
        gbd_id=cid(718),
    ),
    assault_by_firearm=Cause(
        name='assault_by_firearm',
        gbd_id=cid(725),
    ),
    assault_by_sharp_object=Cause(
        name='assault_by_sharp_object',
        gbd_id=cid(726),
    ),
    assault_by_other_means=Cause(
        name='assault_by_other_means',
        gbd_id=cid(727),
    ),
    exposure_to_forces_of_nature=Cause(
        name='exposure_to_forces_of_nature',
        gbd_id=cid(729),
    ),
)
