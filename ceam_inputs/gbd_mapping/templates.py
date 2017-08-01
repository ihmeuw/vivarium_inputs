from typing import NewType, NamedTuple, Union, Set

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
    sequelae: Set[Sequela] = None
    etiologies: Set[Etiology] = None
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
    liver_cancer_due_to_hepatitis_b: Cause
    liver_cancer_due_to_hepatitis_c: Cause
    liver_cancer_due_to_alcohol_use: Cause
    liver_cancer_due_to_other_causes: Cause
    larynx_cancer: Cause
    tracheal_bronchus_and_lung_cancer: Cause
    breast_cancer: Cause
    cervical_cancer: Cause
    uterine_cancer: Cause
    colon_and_rectum_cancer: Cause
    lip_and_oral_cavity_cancer: Cause
    nasopharynx_cancer: Cause
    other_pharynx_cancer: Cause
    gallbladder_and_biliary_tract_cancer: Cause
    pancreatic_cancer: Cause
    ovarian_cancer: Cause
    kidney_cancer: Cause
    bladder_cancer: Cause
    thyroid_cancer: Cause
    mesothelioma: Cause
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
    silicosis: Cause
    asbestosis: Cause
    coal_workers_pneumoconiosis: Cause
    other_pneumoconiosis: Cause
    asthma: Cause
    interstitial_lung_disease_and_pulmonary_sarcoidosis: Cause
    other_chronic_respiratory_diseases: Cause
    cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_b: Cause
    cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_c: Cause
    cirrhosis_and_other_chronic_liver_diseases_due_to_alcohol_use: Cause
    cirrhosis_and_other_chronic_liver_diseases_due_to_other_causes: Cause
    peptic_ulcer_disease: Cause
    pancreatitis: Cause
    epilepsy: Cause
    alcohol_use_disorders: Cause
    opioid_use_disorders: Cause
    cocaine_use_disorders: Cause
    amphetamine_use_disorders: Cause
    cannabis_use_disorders: Cause
    other_drug_use_disorders: Cause
    major_depressive_disorder: Cause
    dysthymia: Cause
    idiopathic_developmental_intellectual_disability: Cause
    diabetes_mellitus: Cause
    chronic_kidney_disease_due_to_diabetes_mellitus: Cause
    chronic_kidney_disease_due_to_hypertension: Cause
    chronic_kidney_disease_due_to_glomerulonephritis: Cause
    chronic_kidney_disease_due_to_other_causes: Cause
    rheumatoid_arthritis: Cause
    osteoarthritis: Cause
    low_back_pain: Cause
    gout: Cause
    cataract: Cause
    macular_degeneration: Cause
    age_related_and_other_hearing_loss: Cause
    pedestrian_road_injuries: Cause
    cyclist_road_injuries: Cause
    motorcyclist_road_injuries: Cause
    motor_vehicle_road_injuries: Cause
    other_road_injuries: Cause
    other_transport_injuries: Cause
    falls: Cause
    drowning: Cause
    fire_heat_and_hot_substances: Cause
    poisonings: Cause
    unintentional_firearm_injuries: Cause
    unintentional_suffocation: Cause
    other_exposure_to_mechanical_forces: Cause
    venomous_animal_contact: Cause
    non_venomous_animal_contact: Cause
    pulmonary_aspiration_and_foreign_body_in_airway: Cause
    foreign_body_in_eyes: Cause
    foreign_body_in_other_body_part: Cause
    other_unintentional_injuries: Cause
    self_harm: Cause
    assault_by_firearm: Cause
    assault_by_sharp_object: Cause
    assault_by_other_means: Cause
    exposure_to_forces_of_nature: Cause


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
    affected_causes: Set[Cause] = None
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
    iron_deficiency: Risk
    vitamin_a_deficiency: Risk
    zinc_deficiency: Risk
    secondhand_smoke: Risk
    alcohol_use: Risk
    high_total_cholesterol: Risk
    high_systolic_blood_pressure: Risk
    high_body_mass_index: Risk
    low_bone_mineral_density: Risk
    diet_low_in_fruits: Risk
    diet_low_in_vegetables: Risk
    diet_low_in_whole_grains: Risk
    diet_low_in_nuts_and_seeds: Risk
    diet_low_in_milk: Risk
    diet_high_in_red_meat: Risk
    diet_high_in_processed_meat: Risk
    diet_high_in_sugar_sweetened_beverages: Risk
    diet_low_in_fiber: Risk
    diet_low_in_seafood_omega_3_fatty_acids: Risk
    diet_low_in_polyunsaturated_fatty_acids: Risk
    diet_high_in_trans_fatty_acids: Risk
    diet_high_in_sodium: Risk
    low_physical_activity: Risk
    occupational_asthmagens: Risk
    occupational_particulate_matter_gases_and_fumes: Risk
    occupational_noise: Risk
    occupational_injuries: Risk
    occupational_ergonomic_factors: Risk
    non_exclusive_breastfeeding: Risk
    discontinued_breastfeeding: Risk
    drug_use_dependence_and_blood_borne_viruses: Risk
    suicide_due_to_drug_use_disorders: Risk
    high_fasting_plasma_glucose_continuous: Risk
    high_fasting_plasma_glucose_categorical: Risk
    low_glomerular_filtration_rate: Risk
    occupational_exposure_to_asbestos: Risk
    occupational_exposure_to_arsenic: Risk
    occupational_exposure_to_beryllium: Risk
    occupational_exposure_to_cadmium: Risk
    occupational_exposure_to_chromium: Risk
    occupational_exposure_to_diesel_engine_exhaust: Risk
    occupational_exposure_to_secondhand_smoke: Risk
    occupational_exposure_to_formaldehyde: Risk
    occupational_exposure_to_nickel: Risk
    occupational_exposure_to_polycyclic_aromatic_hydrocarbons: Risk
    occupational_exposure_to_silica: Risk
    occupational_exposure_to_sulfuric_acid: Risk
    smoking_sir_approach: Risk
    smoking_prevalence_approach: Risk
    intimate_partner_violence_exposure_approach: Risk
    intimate_partner_violence_direct_paf_approach: Risk
    unsafe_sex: Risk
    intimate_partner_violence_hiv_paf_approach: Risk
    occupational_exposure_to_trichloroethylene: Risk
    no_handwashing_with_soap: Risk
    childhood_wasting: Risk
    childhood_stunting: Risk
    lead_exposure_in_blood: Risk
    lead_exposure_in_bone: Risk
    childhood_sexual_abuse_against_females: Risk
    childhood_sexual_abuse_against_males: Risk
