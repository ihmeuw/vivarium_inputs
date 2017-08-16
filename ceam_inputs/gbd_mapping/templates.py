from typing import Union, Tuple


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


class scalar(float):
    """Raw measure value"""
    def __repr__(self):
        return 'scalar({:d})'.format(self)


class _Unknown:
    """Marker for unknown values."""
    def __repr__(self):
        return 'UNKNOWN'

UNKNOWN = _Unknown()


class UnknownEntityError(Exception):
    """Exception raised when a quantity is requested from ceam_inputs with an `UNKNOWN` id."""
    pass


class GbdRecord:
    """Base class for entities modeled in the GBD."""
    __slots__ = ()

    def __contains__(self, item):
        return item in self.__slots__

    def __getitem__(self, item):
        if item in self.__slots__:
            return getattr(self, item)
        else:
            raise KeyError

    def __iter__(self):
        for item in self.__slots__:
            yield getattr(self, item)

    def __repr__(self):

        return "{}({})".format(self.__class__.__name__,
                               ',\n'.join(['{}={}'.format(name, self[name])
                                          for name in self.__slots__]))


class ModelableEntity(GbdRecord):
    """Container for general GBD ids."""
    __slots__ = ('name', 'gbd_id',)

    def __init__(self,
                 name: str,
                 gbd_id: Union[meid, rid, cid],):
        self.name = name
        self.gbd_id = gbd_id


class CauseLike(ModelableEntity):
    """Container for cause-like entity GBD ids."""
    __slots__ = ('name', 'gbd_id', 'prevalence', 'disability_weight',
                 'excess_mortality', 'remission', 'duration', 'proportion')

    def __init__(self,
                 name: str,
                 gbd_id: Union[cid, rid, meid],
                 prevalence: Union[cid, meid, None] = UNKNOWN,
                 disability_weight: Union[hid, meid, None] = UNKNOWN,
                 excess_mortality: Union[cid, meid, None] = UNKNOWN,
                 remission: Union[meid, None] = UNKNOWN,
                 proportion: Union[meid, scalar] = UNKNOWN,
                 duration: scalar = scalar(0),):

        super().__init__(name=name,
                         gbd_id=gbd_id)
        self.prevalence = prevalence
        self.disability_weight = disability_weight
        self.excess_mortality = excess_mortality
        self.remission = remission
        self.duration = duration
        self.proportion = proportion


class SeveritySplit(CauseLike):
    """Container for severity split GBD ids."""
    __slots__ = ('name', 'gbd_id', 'proportion', 'prevalence', 'disability_weight',
                 'excess_mortality', 'remission', 'duration',)

    def __init__(self,
                 name: str,
                 gbd_id: meid,
                 proportion: Union[meid, scalar],
                 prevalence: Union[meid, None] = UNKNOWN,
                 disability_weight: Union[hid, meid, None] = UNKNOWN,
                 excess_mortality: Union[meid, None] = UNKNOWN,
                 remission: Union[meid, None] = UNKNOWN,
                 duration: scalar = scalar(0),):

        super().__init__(name=name,
                         gbd_id=gbd_id,
                         prevalence=prevalence,
                         disability_weight=disability_weight,
                         excess_mortality=excess_mortality,
                         remission=remission,
                         duration=duration,
                         proportion=proportion)


class SeveritySplits(GbdRecord):
    """Holder of severity splits."""
    __slots__ = ('mild', 'moderate', 'severe', 'asymptomatic')

    def __init__(self,
                 mild: SeveritySplit,
                 moderate: SeveritySplit,
                 severe: SeveritySplit,
                 asymptomatic: SeveritySplit = None):
        self.mild = mild
        self.moderate = moderate
        self.severe = severe
        self.asymptomatic = asymptomatic


class Sequela(CauseLike):
    """Container for sequela GBD ids."""
    __slots__ = ('name', 'gbd_id', 'incidence', 'proportion', 'prevalence', 'disability_weight',
                 'excess_mortality', 'remission', 'duration', 'severity_splits')

    def __init__(self,
                 name: str,
                 gbd_id: meid,
                 incidence: Union[meid, None] = UNKNOWN,
                 proportion: Union[meid, str, None] = UNKNOWN,
                 prevalence: Union[meid, None] = UNKNOWN,
                 disability_weight: Union[hid, meid, None] = UNKNOWN,
                 excess_mortality: Union[meid, float, None] = UNKNOWN,
                 remission: Union[meid, None] = UNKNOWN,
                 duration: scalar = scalar(0),
                 severity_splits: SeveritySplits = None,):

        super().__init__(name=name,
                         gbd_id=gbd_id,
                         prevalence=prevalence,
                         disability_weight=disability_weight,
                         excess_mortality=excess_mortality,
                         remission=remission,
                         duration=duration)
        self.incidence = incidence
        self.proportion = proportion
        self.severity_splits = severity_splits


class Etiology(CauseLike):
    """Container for etiology GBD ids."""
    __slots__ = ('name', 'gbd_id', 'prevalence', 'disability_weight',
                 'excess_mortality', 'remission', 'duration',)

    def __init__(self, name: str,
                 gbd_id: Union[rid, None],
                 prevalence: Union[meid, None] = UNKNOWN,
                 disability_weight: Union[hid, meid, None] = UNKNOWN,
                 excess_mortality: Union[meid, None] = UNKNOWN,
                 remission: Union[meid, None] = UNKNOWN,
                 duration: scalar = scalar(0),):
        super().__init__(name=name,
                         gbd_id=gbd_id,
                         prevalence=prevalence,
                         disability_weight=disability_weight,
                         excess_mortality=excess_mortality,
                         remission=remission,
                         duration=duration)


class Cause(CauseLike):
    """Container for cause GBD ids."""
    __slots__ = ('name', 'gbd_id', 'incidence', 'prevalence', 'disability_weight',
                 'excess_mortality', 'remission', 'duration', 'csmr', 'sequelae',
                 'etiologies', 'severity_splits')

    def __init__(self, name: str,
                 gbd_id: cid,
                 incidence: Union[cid, meid] = UNKNOWN,
                 prevalence: Union[cid, meid] = UNKNOWN,
                 disability_weight: Union[hid, meid, float, None] = UNKNOWN,
                 csmr: Union[cid, meid] = UNKNOWN,
                 excess_mortality: Union[cid, meid, None] = UNKNOWN,
                 remission: Union[meid, None] = UNKNOWN,
                 duration: scalar = scalar(0),
                 sequelae: Tuple[Sequela, ...] = None,
                 etiologies: Tuple[Etiology, ...] = None,
                 severity_splits: SeveritySplits = None,):
        super().__init__(name=name,
                         gbd_id=gbd_id,
                         prevalence=prevalence,
                         disability_weight=disability_weight,
                         excess_mortality=excess_mortality,
                         remission=remission,
                         duration=duration)
        self.incidence = incidence
        self.csmr = csmr
        self.sequelae = sequelae
        self.etiologies = etiologies
        self.severity_splits = severity_splits


class Etioloties(GbdRecord):
    """Holder of Etiologies"""
    __slots__ = ('unattributed_diarrhea', 'cholera', 'other_salmonella', 'shigellosis', 'EPEC', 'ETEC',
                 'campylobacter', 'amoebiasis', 'cryptosporidiosis', 'rotaviral_entiritis', 'aeromonas',
                 'clostridium_difficile', 'norovirus', 'adenovirus')

    def __init__(self,
                 unattributed_diarrhea: Etiology,
                 cholera: Etiology,
                 other_salmonella: Etiology,
                 shigellosis: Etiology,
                 EPEC: Etiology,
                 ETEC: Etiology,
                 campylobacter: Etiology,
                 amoebiasis: Etiology,
                 cryptosporidiosis: Etiology,
                 rotaviral_entiritis: Etiology,
                 aeromonas: Etiology,
                 clostridium_difficile: Etiology,
                 norovirus: Etiology,
                 adenovirus: Etiology,):

        super().__init__()
        self.unattributed_diarrhea = unattributed_diarrhea
        self.cholera = cholera
        self.other_salmonella = other_salmonella
        self.shigellosis = shigellosis
        self.EPEC = EPEC
        self.ETEC = ETEC
        self.campylobacter = campylobacter
        self.amoebiasis = amoebiasis
        self.cryptosporidiosis = cryptosporidiosis
        self.rotaviral_entiritis = rotaviral_entiritis
        self.aeromonas = aeromonas
        self.clostridium_difficile = clostridium_difficile
        self.norovirus = norovirus
        self.adenovirus = adenovirus


class Sequelae(GbdRecord):
    """Holder of Sequelae"""
    __slots__ = ('heart_attack', 'heart_failure', 'angina', 'asymptomatic_ihd')

    def __init__(self,
                 heart_attack: Sequela,
                 heart_failure: Sequela,
                 angina: Sequela,
                 asymptomatic_ihd: Sequela,):

        super().__init__()
        self.heart_attack = heart_attack
        self.heart_failure = heart_failure
        self.angina = angina
        self.asymptomatic_ihd = asymptomatic_ihd


class Tmred(GbdRecord):
    """Container for theoretical minimum risk exposure distribution data."""
    __slots__ = ('distribution', 'min', 'max', 'inverted')

    def __init__(self,
                 distribution: str,
                 min: float,
                 max: float,
                 inverted: bool,):

        super().__init__()
        self.distribution = distribution
        self.min = min
        self.max = max
        self.inverted = inverted


class Levels(GbdRecord):
    """Container for categorical risk exposure levels."""
    __slots__ = ('cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9')

    def __init__(self,
                 cat1: str,
                 cat2: str,
                 cat3: str = None,
                 cat4: str = None,
                 cat5: str = None,
                 cat6: str = None,
                 cat7: str = None,
                 cat8: str = None,
                 cat9: str = None,):

        super().__init__()
        self.cat1 = cat1
        self.cat2 = cat2
        self.cat3 = cat3
        self.cat4 = cat4
        self.cat5 = cat5
        self.cat6 = cat6
        self.cat7 = cat7
        self.cat8 = cat8
        self.cat9 = cat9


class Risk(ModelableEntity):
    """Container type for risk factor GBD ids."""
    __slots__ = ('name', 'gbd_id', 'affected_causes', 'distribution', 'levels', 'tmred',
                 'scale', 'max_rr', 'min_rr', 'max_val', 'min_val', 'max_var')

    def __init__(self,
                 name: str,
                 gbd_id: rid,
                 affected_causes: Tuple[Cause, ...],
                 distribution: str = UNKNOWN,
                 levels: Levels = None,
                 tmred: Tmred = None,
                 scale: float = None,
                 max_rr: float = None,
                 min_rr: float = None,
                 max_val: float = None,
                 min_val: float = None,
                 max_var: float = None,):

        super().__init__(name=name, gbd_id=gbd_id)
        self.affected_causes = affected_causes
        self.distribution = distribution
        self.levels = levels
        self.tmred = tmred
        self.scale = scale
        self.max_rr = max_rr
        self.min_rr = min_rr
        self.max_val = max_val
        self.min_val = min_val
        self.max_var = max_var


class Causes(GbdRecord):
    """Holder of causes"""
    __slots__ = ('all_causes', 'tuberculosis', 'hiv_aids_tuberculosis', 'hiv_aids_other_diseases', 'diarrhea',
                 'typhoid_fever', 'paratyphoid_fever', 'lower_respiratory_infections', 'otitis_media', 'measles',
                 'maternal_hemorrhage', 'maternal_sepsis_and_other_infections',
                 'maternal_abortion_miscarriage_and_ectopic_pregnancy', 'protein_energy_malnutrition',
                 'vitamin_a_deficiency', 'iron_deficiency_anemia', 'syphilis', 'chlamydia', 'gonorrhea',
                 'trichomoniasis', 'genital_herpes', 'other_sexually_transmitted_disease', 'hepatitis_b',
                 'hepatitis_c', 'esophageal_cancer', 'stomach_cancer', 'liver_cancer_due_to_hepatitis_b',
                 'liver_cancer_due_to_hepatitis_c', 'liver_cancer_due_to_alcohol_use',
                 'liver_cancer_due_to_other_causes', 'larynx_cancer', 'tracheal_bronchus_and_lung_cancer',
                 'breast_cancer', 'cervical_cancer', 'uterine_cancer', 'colon_and_rectum_cancer',
                 'lip_and_oral_cavity_cancer', 'nasopharynx_cancer', 'other_pharynx_cancer',
                 'gallbladder_and_biliary_tract_cancer', 'pancreatic_cancer', 'ovarian_cancer', 'kidney_cancer',
                 'bladder_cancer', 'thyroid_cancer', 'mesothelioma', 'rheumatic_heart_disease',
                 'ischemic_heart_disease', 'chronic_stroke', 'ischemic_stroke', 'hemorrhagic_stroke',
                 'hypertensive_heart_disease', 'cardiomyopathy_and_myocarditis', 'atrial_fibrillation_and_flutter',
                 'aortic_aneurysm', 'peripheral_vascular_disease', 'endocarditis',
                 'other_cardiovascular_and_circulatory_diseases', 'chronic_obstructive_pulmonary_disease', 'silicosis',
                 'asbestosis', 'coal_workers_pneumoconiosis', 'other_pneumoconiosis', 'asthma',
                 'interstitial_lung_disease_and_pulmonary_sarcoidosis', 'other_chronic_respiratory_diseases',
                 'cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_b',
                 'cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_c',
                 'cirrhosis_and_other_chronic_liver_diseases_due_to_alcohol_use',
                 'cirrhosis_and_other_chronic_liver_diseases_due_to_other_causes', 'peptic_ulcer_disease',
                 'pancreatitis', 'epilepsy', 'alcohol_use_disorders', 'opioid_use_disorders', 'cocaine_use_disorders',
                 'amphetamine_use_disorders', 'cannabis_use_disorders', 'other_drug_use_disorders',
                 'major_depressive_disorder', 'dysthymia', 'idiopathic_developmental_intellectual_disability',
                 'diabetes_mellitus', 'chronic_kidney_disease_due_to_diabetes_mellitus',
                 'chronic_kidney_disease_due_to_hypertension', 'chronic_kidney_disease_due_to_glomerulonephritis',
                 'chronic_kidney_disease_due_to_other_causes', 'rheumatoid_arthritis', 'osteoarthritis',
                 'low_back_pain', 'gout', 'cataract', 'macular_degeneration', 'age_related_and_other_hearing_loss',
                 'pedestrian_road_injuries', 'cyclist_road_injuries', 'motorcyclist_road_injuries',
                 'motor_vehicle_road_injuries', 'other_road_injuries', 'other_transport_injuries', 'falls', 'drowning',
                 'fire_heat_and_hot_substances', 'poisonings', 'unintentional_firearm_injuries',
                 'unintentional_suffocation', 'other_exposure_to_mechanical_forces', 'venomous_animal_contact',
                 'non_venomous_animal_contact', 'pulmonary_aspiration_and_foreign_body_in_airway',
                 'foreign_body_in_eyes', 'foreign_body_in_other_body_part', 'other_unintentional_injuries',
                 'self_harm', 'assault_by_firearm', 'assault_by_sharp_object', 'assault_by_other_means',
                 'exposure_to_forces_of_nature')

    def __init__(self,
                 all_causes: Cause,
                 tuberculosis: Cause,
                 hiv_aids_tuberculosis: Cause,
                 hiv_aids_other_diseases: Cause,
                 diarrhea: Cause,
                 typhoid_fever: Cause,
                 paratyphoid_fever: Cause,
                 lower_respiratory_infections: Cause,
                 otitis_media: Cause,
                 measles: Cause,
                 maternal_hemorrhage: Cause,
                 maternal_sepsis_and_other_infections: Cause,
                 maternal_abortion_miscarriage_and_ectopic_pregnancy: Cause,
                 protein_energy_malnutrition: Cause,
                 vitamin_a_deficiency: Cause,
                 iron_deficiency_anemia: Cause,
                 syphilis: Cause,
                 chlamydia: Cause,
                 gonorrhea: Cause,
                 trichomoniasis: Cause,
                 genital_herpes: Cause,
                 other_sexually_transmitted_disease: Cause,
                 hepatitis_b: Cause,
                 hepatitis_c: Cause,
                 esophageal_cancer: Cause,
                 stomach_cancer: Cause,
                 liver_cancer_due_to_hepatitis_b: Cause,
                 liver_cancer_due_to_hepatitis_c: Cause,
                 liver_cancer_due_to_alcohol_use: Cause,
                 liver_cancer_due_to_other_causes: Cause,
                 larynx_cancer: Cause,
                 tracheal_bronchus_and_lung_cancer: Cause,
                 breast_cancer: Cause,
                 cervical_cancer: Cause,
                 uterine_cancer: Cause,
                 colon_and_rectum_cancer: Cause,
                 lip_and_oral_cavity_cancer: Cause,
                 nasopharynx_cancer: Cause,
                 other_pharynx_cancer: Cause,
                 gallbladder_and_biliary_tract_cancer: Cause,
                 pancreatic_cancer: Cause,
                 ovarian_cancer: Cause,
                 kidney_cancer: Cause,
                 bladder_cancer: Cause,
                 thyroid_cancer: Cause,
                 mesothelioma: Cause,
                 rheumatic_heart_disease: Cause,
                 ischemic_heart_disease: Cause,
                 chronic_stroke: Cause,
                 ischemic_stroke: Cause,
                 hemorrhagic_stroke: Cause,
                 hypertensive_heart_disease: Cause,
                 cardiomyopathy_and_myocarditis: Cause,
                 atrial_fibrillation_and_flutter: Cause,
                 aortic_aneurysm: Cause,
                 peripheral_vascular_disease: Cause,
                 endocarditis: Cause,
                 other_cardiovascular_and_circulatory_diseases: Cause,
                 chronic_obstructive_pulmonary_disease: Cause,
                 silicosis: Cause,
                 asbestosis: Cause,
                 coal_workers_pneumoconiosis: Cause,
                 other_pneumoconiosis: Cause,
                 asthma: Cause,
                 interstitial_lung_disease_and_pulmonary_sarcoidosis: Cause,
                 other_chronic_respiratory_diseases: Cause,
                 cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_b: Cause,
                 cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_c: Cause,
                 cirrhosis_and_other_chronic_liver_diseases_due_to_alcohol_use: Cause,
                 cirrhosis_and_other_chronic_liver_diseases_due_to_other_causes: Cause,
                 peptic_ulcer_disease: Cause,
                 pancreatitis: Cause,
                 epilepsy: Cause,
                 alcohol_use_disorders: Cause,
                 opioid_use_disorders: Cause,
                 cocaine_use_disorders: Cause,
                 amphetamine_use_disorders: Cause,
                 cannabis_use_disorders: Cause,
                 other_drug_use_disorders: Cause,
                 major_depressive_disorder: Cause,
                 dysthymia: Cause,
                 idiopathic_developmental_intellectual_disability: Cause,
                 diabetes_mellitus: Cause,
                 chronic_kidney_disease_due_to_diabetes_mellitus: Cause,
                 chronic_kidney_disease_due_to_hypertension: Cause,
                 chronic_kidney_disease_due_to_glomerulonephritis: Cause,
                 chronic_kidney_disease_due_to_other_causes: Cause,
                 rheumatoid_arthritis: Cause,
                 osteoarthritis: Cause,
                 low_back_pain: Cause,
                 gout: Cause,
                 cataract: Cause,
                 macular_degeneration: Cause,
                 age_related_and_other_hearing_loss: Cause,
                 pedestrian_road_injuries: Cause,
                 cyclist_road_injuries: Cause,
                 motorcyclist_road_injuries: Cause,
                 motor_vehicle_road_injuries: Cause,
                 other_road_injuries: Cause,
                 other_transport_injuries: Cause,
                 falls: Cause,
                 drowning: Cause,
                 fire_heat_and_hot_substances: Cause,
                 poisonings: Cause,
                 unintentional_firearm_injuries: Cause,
                 unintentional_suffocation: Cause,
                 other_exposure_to_mechanical_forces: Cause,
                 venomous_animal_contact: Cause,
                 non_venomous_animal_contact: Cause,
                 pulmonary_aspiration_and_foreign_body_in_airway: Cause,
                 foreign_body_in_eyes: Cause,
                 foreign_body_in_other_body_part: Cause,
                 other_unintentional_injuries: Cause,
                 self_harm: Cause,
                 assault_by_firearm: Cause,
                 assault_by_sharp_object: Cause,
                 assault_by_other_means: Cause,
                 exposure_to_forces_of_nature: Cause,):

        super().__init__()
        self.all_causes = all_causes
        self.tuberculosis = tuberculosis
        self.hiv_aids_tuberculosis = hiv_aids_tuberculosis
        self.hiv_aids_other_diseases = hiv_aids_other_diseases
        self.diarrhea = diarrhea
        self.typhoid_fever = typhoid_fever
        self.paratyphoid_fever = paratyphoid_fever
        self.lower_respiratory_infections = lower_respiratory_infections
        self.otitis_media = otitis_media
        self.measles = measles
        self.maternal_hemorrhage = maternal_hemorrhage
        self.maternal_sepsis_and_other_infections = maternal_sepsis_and_other_infections
        self.maternal_abortion_miscarriage_and_ectopic_pregnancy = maternal_abortion_miscarriage_and_ectopic_pregnancy
        self.protein_energy_malnutrition = protein_energy_malnutrition
        self.vitamin_a_deficiency = vitamin_a_deficiency
        self.iron_deficiency_anemia = iron_deficiency_anemia
        self.syphilis = syphilis
        self.chlamydia = chlamydia
        self.gonorrhea = gonorrhea
        self.trichomoniasis = trichomoniasis
        self.genital_herpes = genital_herpes
        self.other_sexually_transmitted_disease = other_sexually_transmitted_disease
        self.hepatitis_b = hepatitis_b
        self.hepatitis_c = hepatitis_c
        self.esophageal_cancer = esophageal_cancer
        self.stomach_cancer = stomach_cancer
        self.liver_cancer_due_to_hepatitis_b = liver_cancer_due_to_hepatitis_b
        self.liver_cancer_due_to_hepatitis_c = liver_cancer_due_to_hepatitis_c
        self.liver_cancer_due_to_alcohol_use = liver_cancer_due_to_alcohol_use
        self.liver_cancer_due_to_other_causes = liver_cancer_due_to_other_causes
        self.larynx_cancer = larynx_cancer
        self.tracheal_bronchus_and_lung_cancer = tracheal_bronchus_and_lung_cancer
        self.breast_cancer = breast_cancer
        self.cervical_cancer = cervical_cancer
        self.uterine_cancer = uterine_cancer
        self.colon_and_rectum_cancer = colon_and_rectum_cancer
        self.lip_and_oral_cavity_cancer = lip_and_oral_cavity_cancer
        self.nasopharynx_cancer = nasopharynx_cancer
        self.other_pharynx_cancer = other_pharynx_cancer
        self.gallbladder_and_biliary_tract_cancer = gallbladder_and_biliary_tract_cancer
        self.pancreatic_cancer = pancreatic_cancer
        self.ovarian_cancer = ovarian_cancer
        self.kidney_cancer = kidney_cancer
        self.bladder_cancer = bladder_cancer
        self.thyroid_cancer = thyroid_cancer
        self.mesothelioma = mesothelioma
        self.rheumatic_heart_disease = rheumatic_heart_disease
        self.ischemic_heart_disease = ischemic_heart_disease
        self.chronic_stroke = chronic_stroke
        self.ischemic_stroke = ischemic_stroke
        self.hemorrhagic_stroke = hemorrhagic_stroke
        self.hypertensive_heart_disease = hypertensive_heart_disease
        self.cardiomyopathy_and_myocarditis = cardiomyopathy_and_myocarditis
        self.atrial_fibrillation_and_flutter = atrial_fibrillation_and_flutter
        self.aortic_aneurysm = aortic_aneurysm
        self.peripheral_vascular_disease = peripheral_vascular_disease
        self.endocarditis = endocarditis
        self.other_cardiovascular_and_circulatory_diseases = other_cardiovascular_and_circulatory_diseases
        self.chronic_obstructive_pulmonary_disease = chronic_obstructive_pulmonary_disease
        self.silicosis = silicosis
        self.asbestosis = asbestosis
        self.coal_workers_pneumoconiosis = coal_workers_pneumoconiosis
        self.other_pneumoconiosis = other_pneumoconiosis
        self.asthma = asthma
        self.interstitial_lung_disease_and_pulmonary_sarcoidosis = interstitial_lung_disease_and_pulmonary_sarcoidosis
        self.other_chronic_respiratory_diseases = other_chronic_respiratory_diseases
        self.cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_b = \
            cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_b
        self.cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_c = \
            cirrhosis_and_other_chronic_liver_diseases_due_to_hepatitis_c
        self.cirrhosis_and_other_chronic_liver_diseases_due_to_alcohol_use = \
            cirrhosis_and_other_chronic_liver_diseases_due_to_alcohol_use
        self.cirrhosis_and_other_chronic_liver_diseases_due_to_other_causes = \
            cirrhosis_and_other_chronic_liver_diseases_due_to_other_causes
        self.peptic_ulcer_disease = peptic_ulcer_disease
        self.pancreatitis = pancreatitis
        self.epilepsy = epilepsy
        self.alcohol_use_disorders = alcohol_use_disorders
        self.opioid_use_disorders = opioid_use_disorders
        self.cocaine_use_disorders = cocaine_use_disorders
        self.amphetamine_use_disorders = amphetamine_use_disorders
        self.cannabis_use_disorders = cannabis_use_disorders
        self.other_drug_use_disorders = other_drug_use_disorders
        self.major_depressive_disorder = major_depressive_disorder
        self.dysthymia = dysthymia
        self.idiopathic_developmental_intellectual_disability = idiopathic_developmental_intellectual_disability
        self.diabetes_mellitus = diabetes_mellitus
        self.chronic_kidney_disease_due_to_diabetes_mellitus = chronic_kidney_disease_due_to_diabetes_mellitus
        self.chronic_kidney_disease_due_to_hypertension = chronic_kidney_disease_due_to_hypertension
        self.chronic_kidney_disease_due_to_glomerulonephritis = chronic_kidney_disease_due_to_glomerulonephritis
        self.chronic_kidney_disease_due_to_other_causes = chronic_kidney_disease_due_to_other_causes
        self.rheumatoid_arthritis = rheumatoid_arthritis
        self.osteoarthritis = osteoarthritis
        self.low_back_pain = low_back_pain
        self.gout = gout
        self.cataract = cataract
        self.macular_degeneration = macular_degeneration
        self.age_related_and_other_hearing_loss = age_related_and_other_hearing_loss
        self.pedestrian_road_injuries = pedestrian_road_injuries
        self.cyclist_road_injuries = cyclist_road_injuries
        self.motorcyclist_road_injuries = motorcyclist_road_injuries
        self.motor_vehicle_road_injuries = motor_vehicle_road_injuries
        self.other_road_injuries = other_road_injuries
        self.other_transport_injuries = other_transport_injuries
        self.falls = falls
        self.drowning = drowning
        self.fire_heat_and_hot_substances = fire_heat_and_hot_substances
        self.poisonings = poisonings
        self.unintentional_firearm_injuries = unintentional_firearm_injuries
        self.unintentional_suffocation = unintentional_suffocation
        self.other_exposure_to_mechanical_forces = other_exposure_to_mechanical_forces
        self.venomous_animal_contact = venomous_animal_contact
        self.non_venomous_animal_contact = non_venomous_animal_contact
        self.pulmonary_aspiration_and_foreign_body_in_airway = pulmonary_aspiration_and_foreign_body_in_airway
        self.foreign_body_in_eyes = foreign_body_in_eyes
        self.foreign_body_in_other_body_part = foreign_body_in_other_body_part
        self.other_unintentional_injuries = other_unintentional_injuries
        self.self_harm = self_harm
        self.assault_by_firearm = assault_by_firearm
        self.assault_by_sharp_object = assault_by_sharp_object
        self.assault_by_other_means = assault_by_other_means
        self.exposure_to_forces_of_nature = exposure_to_forces_of_nature


class Risks(GbdRecord):
    """Holder of risks"""
    __slots__ = ('unsafe_water_source', 'unsafe_sanitation', 'ambient_particulate_matter_pollution',
                 'household_air_pollution_from_solid_fuels', 'ambient_ozone_pollution', 'residential_radon',
                 'childhood_underweight', 'iron_deficiency', 'vitamin_a_deficiency', 'zinc_deficiency',
                 'secondhand_smoke', 'alcohol_use', 'high_total_cholesterol', 'high_systolic_blood_pressure',
                 'high_body_mass_index', 'low_bone_mineral_density', 'diet_low_in_fruits', 'diet_low_in_vegetables',
                 'diet_low_in_whole_grains', 'diet_low_in_nuts_and_seeds', 'diet_low_in_milk', 'diet_high_in_red_meat',
                 'diet_high_in_processed_meat', 'diet_high_in_sugar_sweetened_beverages', 'diet_low_in_fiber',
                 'diet_low_in_seafood_omega_3_fatty_acids', 'diet_low_in_polyunsaturated_fatty_acids',
                 'diet_high_in_trans_fatty_acids', 'diet_high_in_sodium', 'low_physical_activity',
                 'occupational_asthmagens', 'occupational_particulate_matter_gases_and_fumes', 'occupational_noise',
                 'occupational_injuries', 'occupational_ergonomic_factors', 'non_exclusive_breastfeeding',
                 'discontinued_breastfeeding', 'drug_use_dependence_and_blood_borne_viruses',
                 'suicide_due_to_drug_use_disorders', 'high_fasting_plasma_glucose_continuous',
                 'high_fasting_plasma_glucose_categorical', 'low_glomerular_filtration_rate',
                 'occupational_exposure_to_asbestos', 'occupational_exposure_to_arsenic',
                 'occupational_exposure_to_beryllium', 'occupational_exposure_to_cadmium',
                 'occupational_exposure_to_chromium', 'occupational_exposure_to_diesel_engine_exhaust',
                 'occupational_exposure_to_secondhand_smoke', 'occupational_exposure_to_formaldehyde',
                 'occupational_exposure_to_nickel', 'occupational_exposure_to_polycyclic_aromatic_hydrocarbons',
                 'occupational_exposure_to_silica', 'occupational_exposure_to_sulfuric_acid', 'smoking_sir_approach',
                 'smoking_prevalence_approach', 'intimate_partner_violence_exposure_approach',
                 'intimate_partner_violence_direct_paf_approach', 'unsafe_sex',
                 'intimate_partner_violence_hiv_paf_approach', 'occupational_exposure_to_trichloroethylene',
                 'no_handwashing_with_soap', 'childhood_wasting', 'childhood_stunting', 'lead_exposure_in_blood',
                 'lead_exposure_in_bone', 'childhood_sexual_abuse_against_females',
                 'childhood_sexual_abuse_against_males')

    def __init__(self,
                 unsafe_water_source: Risk,
                 unsafe_sanitation: Risk,
                 ambient_particulate_matter_pollution: Risk,
                 household_air_pollution_from_solid_fuels: Risk,
                 ambient_ozone_pollution: Risk,
                 residential_radon: Risk,
                 childhood_underweight: Risk,
                 iron_deficiency: Risk,
                 vitamin_a_deficiency: Risk,
                 zinc_deficiency: Risk,
                 secondhand_smoke: Risk,
                 alcohol_use: Risk,
                 high_total_cholesterol: Risk,
                 high_systolic_blood_pressure: Risk,
                 high_body_mass_index: Risk,
                 low_bone_mineral_density: Risk,
                 diet_low_in_fruits: Risk,
                 diet_low_in_vegetables: Risk,
                 diet_low_in_whole_grains: Risk,
                 diet_low_in_nuts_and_seeds: Risk,
                 diet_low_in_milk: Risk,
                 diet_high_in_red_meat: Risk,
                 diet_high_in_processed_meat: Risk,
                 diet_high_in_sugar_sweetened_beverages: Risk,
                 diet_low_in_fiber: Risk,
                 diet_low_in_seafood_omega_3_fatty_acids: Risk,
                 diet_low_in_polyunsaturated_fatty_acids: Risk,
                 diet_high_in_trans_fatty_acids: Risk,
                 diet_high_in_sodium: Risk,
                 low_physical_activity: Risk,
                 occupational_asthmagens: Risk,
                 occupational_particulate_matter_gases_and_fumes: Risk,
                 occupational_noise: Risk,
                 occupational_injuries: Risk,
                 occupational_ergonomic_factors: Risk,
                 non_exclusive_breastfeeding: Risk,
                 discontinued_breastfeeding: Risk,
                 drug_use_dependence_and_blood_borne_viruses: Risk,
                 suicide_due_to_drug_use_disorders: Risk,
                 high_fasting_plasma_glucose_continuous: Risk,
                 high_fasting_plasma_glucose_categorical: Risk,
                 low_glomerular_filtration_rate: Risk,
                 occupational_exposure_to_asbestos: Risk,
                 occupational_exposure_to_arsenic: Risk,
                 occupational_exposure_to_beryllium: Risk,
                 occupational_exposure_to_cadmium: Risk,
                 occupational_exposure_to_chromium: Risk,
                 occupational_exposure_to_diesel_engine_exhaust: Risk,
                 occupational_exposure_to_secondhand_smoke: Risk,
                 occupational_exposure_to_formaldehyde: Risk,
                 occupational_exposure_to_nickel: Risk,
                 occupational_exposure_to_polycyclic_aromatic_hydrocarbons: Risk,
                 occupational_exposure_to_silica: Risk,
                 occupational_exposure_to_sulfuric_acid: Risk,
                 smoking_sir_approach: Risk,
                 smoking_prevalence_approach: Risk,
                 intimate_partner_violence_exposure_approach: Risk,
                 intimate_partner_violence_direct_paf_approach: Risk,
                 unsafe_sex: Risk,
                 intimate_partner_violence_hiv_paf_approach: Risk,
                 occupational_exposure_to_trichloroethylene: Risk,
                 no_handwashing_with_soap: Risk,
                 childhood_wasting: Risk,
                 childhood_stunting: Risk,
                 lead_exposure_in_blood: Risk,
                 lead_exposure_in_bone: Risk,
                 childhood_sexual_abuse_against_females: Risk,
                 childhood_sexual_abuse_against_males: Risk,):

        super().__init__()
        self.unsafe_water_source = unsafe_water_source
        self.unsafe_sanitation = unsafe_sanitation
        self.ambient_particulate_matter_pollution = ambient_particulate_matter_pollution
        self.household_air_pollution_from_solid_fuels = household_air_pollution_from_solid_fuels
        self.ambient_ozone_pollution = ambient_ozone_pollution
        self.residential_radon = residential_radon
        self.childhood_underweight = childhood_underweight
        self.iron_deficiency = iron_deficiency
        self.vitamin_a_deficiency = vitamin_a_deficiency
        self.zinc_deficiency = zinc_deficiency
        self.secondhand_smoke = secondhand_smoke
        self.alcohol_use = alcohol_use
        self.high_total_cholesterol = high_total_cholesterol
        self.high_systolic_blood_pressure = high_systolic_blood_pressure
        self.high_body_mass_index = high_body_mass_index
        self.low_bone_mineral_density = low_bone_mineral_density
        self.diet_low_in_fruits = diet_low_in_fruits
        self.diet_low_in_vegetables = diet_low_in_vegetables
        self.diet_low_in_whole_grains = diet_low_in_whole_grains
        self.diet_low_in_nuts_and_seeds = diet_low_in_nuts_and_seeds
        self.diet_low_in_milk = diet_low_in_milk
        self.diet_high_in_red_meat = diet_high_in_red_meat
        self.diet_high_in_processed_meat = diet_high_in_processed_meat
        self.diet_high_in_sugar_sweetened_beverages = diet_high_in_sugar_sweetened_beverages
        self.diet_low_in_fiber = diet_low_in_fiber
        self.diet_low_in_seafood_omega_3_fatty_acids = diet_low_in_seafood_omega_3_fatty_acids
        self.diet_low_in_polyunsaturated_fatty_acids = diet_low_in_polyunsaturated_fatty_acids
        self.diet_high_in_trans_fatty_acids = diet_high_in_trans_fatty_acids
        self.diet_high_in_sodium = diet_high_in_sodium
        self.low_physical_activity = low_physical_activity
        self.occupational_asthmagens = occupational_asthmagens
        self.occupational_particulate_matter_gases_and_fumes = occupational_particulate_matter_gases_and_fumes
        self.occupational_noise = occupational_noise
        self.occupational_injuries = occupational_injuries
        self.occupational_ergonomic_factors = occupational_ergonomic_factors
        self.non_exclusive_breastfeeding = non_exclusive_breastfeeding
        self.discontinued_breastfeeding = discontinued_breastfeeding
        self.drug_use_dependence_and_blood_borne_viruses = drug_use_dependence_and_blood_borne_viruses
        self.suicide_due_to_drug_use_disorders = suicide_due_to_drug_use_disorders
        self.high_fasting_plasma_glucose_continuous = high_fasting_plasma_glucose_continuous
        self.high_fasting_plasma_glucose_categorical = high_fasting_plasma_glucose_categorical
        self.low_glomerular_filtration_rate = low_glomerular_filtration_rate
        self.occupational_exposure_to_asbestos = occupational_exposure_to_asbestos
        self.occupational_exposure_to_arsenic = occupational_exposure_to_arsenic
        self.occupational_exposure_to_beryllium = occupational_exposure_to_beryllium
        self.occupational_exposure_to_cadmium = occupational_exposure_to_cadmium
        self.occupational_exposure_to_chromium = occupational_exposure_to_chromium
        self.occupational_exposure_to_diesel_engine_exhaust = occupational_exposure_to_diesel_engine_exhaust
        self.occupational_exposure_to_secondhand_smoke = occupational_exposure_to_secondhand_smoke
        self.occupational_exposure_to_formaldehyde = occupational_exposure_to_formaldehyde
        self.occupational_exposure_to_nickel = occupational_exposure_to_nickel
        self.occupational_exposure_to_polycyclic_aromatic_hydrocarbons = \
            occupational_exposure_to_polycyclic_aromatic_hydrocarbons
        self.occupational_exposure_to_silica = occupational_exposure_to_silica
        self.occupational_exposure_to_sulfuric_acid = occupational_exposure_to_sulfuric_acid
        self.smoking_sir_approach = smoking_sir_approach
        self.smoking_prevalence_approach = smoking_prevalence_approach
        self.intimate_partner_violence_exposure_approach = intimate_partner_violence_exposure_approach
        self.intimate_partner_violence_direct_paf_approach = intimate_partner_violence_direct_paf_approach
        self.unsafe_sex = unsafe_sex
        self.intimate_partner_violence_hiv_paf_approach = intimate_partner_violence_hiv_paf_approach
        self.occupational_exposure_to_trichloroethylene = occupational_exposure_to_trichloroethylene
        self.no_handwashing_with_soap = no_handwashing_with_soap
        self.childhood_wasting = childhood_wasting
        self.childhood_stunting = childhood_stunting
        self.lead_exposure_in_blood = lead_exposure_in_blood
        self.lead_exposure_in_bone = lead_exposure_in_bone
        self.childhood_sexual_abuse_against_females = childhood_sexual_abuse_against_females
        self.childhood_sexual_abuse_against_males = childhood_sexual_abuse_against_males


class HealthcareEntity(ModelableEntity):
    """Container for healthcare system GBD ids and data."""
    __slots__ = ('name', 'gbd_id', 'proportion')

    def __init__(self,
                 name: str,
                 gbd_id: meid,
                 proportion: meid = None):
        super().__init__(name=name, gbd_id=gbd_id)
        self.proportion = proportion


class HealthcareEntities(GbdRecord):
    """Holder of healthcare modelable entities"""
    __slots__ = ('outpatient_visits', )

    def __init__(self,
                 outpatient_visits: ModelableEntity,):

        super().__init__()
        self.outpatient_visits = outpatient_visits


class TreatmentTechnology(ModelableEntity):
    """Container for treatment technology GBD ids and data."""
    __slots__ = ('name', 'unit_cost', 'coverage')

    def __init__(self,
                 name: str,
                 gbd_id: meid,
                 unit_cost: float,
                 coverage: float,):
        super().__init__(name=name, gbd_id=gbd_id)
        self.unit_cost = unit_cost
        self.coverage = coverage


class TreatmentTechnologies(GbdRecord):
    """Holder for treatment technology records."""
    __slots__ = ('ors')

    def __init__(self,
                 ors: TreatmentTechnology,):
        self.ors = ors
