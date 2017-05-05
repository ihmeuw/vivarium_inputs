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

raw_cause_mapping = {
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
            'mortality': meid(2412),
        },
        'moderate_heart_failure': {
            'gbd_cause': cid(493),
            'disability_weight': meid(1822),
            'excess_mortality': meid(2412),
            'prevalence': meid(1822),
        },
        'severe_heart_failure': {
            'gbd_cause': cid(493),
            'disability_weight': meid(1823),
            'excess_mortality': meid(2412),
            'prevalence': meid(1823),
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
            'mortality': meid(1817),
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
        },
        'severe_angina': {
            'gbd_cause': cid(493),
            'disability_weight': meid(1820),
            'excess_mortality': meid(1817),
            'prevalence': meid(1820),
        },
        'asymptomatic_ihd': {
            'gbd_cause': cid(493),
            'disability_weight': meid(3233),
            'excess_mortality': 0.0,
            'prevalence': meid(3233),
            'mortality': meid(3233),
        },
        'hemorrhagic_stroke': {
            'gbd_cause': cid(496),
            'disability_weight': 0.32,
            'excess_mortality': meid(9311),
            'prevalence': meid(9311),
            'mortality': meid(9311),
            'incidence': meid(9311),
        },
        'ischemic_stroke': {
            'gbd_cause': cid(495),
            'disability_weight': 0.32,
            'excess_mortality': meid(9310),
            'prevalence': meid(9310),
            'mortality': meid(9310),
            'incidence': meid(9310),
        },
        'chronic_stroke': {
            'gbd_cause': cid(494),
            'disability_weight': 0.32,
            'excess_mortality': meid(9312),
            'prevalence': meid(9312),
            'mortality': meid(9312),
        },
        'diarrhea': {
            'gbd_cause': cid(302),
            'disability_weight': 0.23,
            'excess_mortality': meid(1181),
            'prevalence': meid(1181),
            'incidence': meid(1181),
            'mortality': meid(1181),
        },
        {
        'death_due_to_severe_diarrhea': {
            'mortality': meid(1181),
        },
        'chronic_kidney_disease': {
            'gbd_cause': cid(591),
        }
}

for k,v in raw_cause_mapping.items():
    v['name'] = k

causes = ConfigTree(raw_cause_mapping)
causes.freeze()

raw_risk_mapping = {
        'smoking': {
            'gbd_risk': rid(166),
            'effected_causes': [causes.heart_attack, causes.hemorrhagic_stroke, causes.ischemic_stroke],
        },
        'secondhand_smoke': {
            'gbd_risk': rid(100),
            'effected_causes': [causes.heart_attack, causes.hemorrhagic_stroke, causes.ischemic_stroke],
        },
        'household_air_polution': {
            'gbd_risk': rid(87),
            'effected_causes': [causes.heart_attack, causes.hemorrhagic_stroke, causes.ischemic_stroke],
        },
        'body_mass_index': {
            'gbd_risk': rid(108),
            'effected_causes': [causes.heart_attack, causes.hemorrhagic_stroke, causes.ischemic_stroke],
            'tmrl': 21,
            'scale': 5,
        },
        'systolic_blood_pressure': {
            'gbd_risk': rid(107),
            'effected_causes': [causes.heart_attack, causes.hemorrhagic_stroke, causes.ischemic_stroke, causes.chronic_kidney_disease],
            'tmrl': 112.5,
            'scale': 10,
        },
        'cholesterol': {
            'gbd_risk': rid(106),
            'effected_causes': [causes.heart_attack, causes.ischemic_stroke],
            'tmrl': 3.08,
            'scale': 1,
        },
        'fasting_plasma_glucose': {
            'gbd_risk': rid(141),
            'effected_causes': [causes.heart_attack, causes.hemorrhagic_stroke, causes.ischemic_stroke],
            'tmrl': 5.1,
            'scale': 1,
        },
}

for k,v in raw_risk_mapping.items():
    v['name'] = k

risk_factors = ConfigTree(raw_risk_mapping)
risk_factors.freeze()
