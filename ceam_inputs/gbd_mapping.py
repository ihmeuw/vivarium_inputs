from ceam.config_tree import ConfigTree

class meid(int):
    def __repr__(self):
        return 'meid({:d})'.format(self)

raw_cause_mapping = {
        'heart_attack': {
            'excess_mortality': meid(1814),
            'prevalence': meid(1814),
            'mortality': meid(1814),
            'incidence': meid(1814),
        },
        'mild_heart_failure': {
            'disability_weight': meid(1821),
            'excess_mortality': meid(2412),
            'prevalence': meid(1821),
            'mortality': meid(2412),
        },
        'moderate_heart_failure': {
            'disability_weight': meid(1822),
            'excess_mortality': meid(2412),
            'prevalence': meid(1822),
        },
        'severe_heart_failure': {
            'disability_weight': meid(1823),
            'excess_mortality': meid(2412),
            'prevalence': meid(1823),
        },
        'angina_not_due_to_MI': {
            'incidence': meid(1817),
        },
        'asymptomatic_angina': {
            'disability_weight': meid(1823),
            'excess_mortality': meid(1817),
            'prevalence': meid(3102),
            'mortality': meid(1817),
        },
        'mild_angina': {
            'disability_weight': meid(1818),
            'excess_mortality': meid(1817),
            'prevalence': meid(1818),
        },
        'moderate_angina': {
            'disability_weight': meid(1819),
            'excess_mortality': meid(1817),
            'prevalence': meid(1819),
        },
        'severe_angina': {
            'disability_weight': meid(1820),
            'excess_mortality': meid(1817),
            'prevalence': meid(1820),
        },
        'asymptomatic_ihd': {
            'disability_weight': meid(3233),
            'excess_mortality': 0.0,
            'prevalence': meid(3233),
            'mortality': meid(3233),
        },
        'hemorrhagic_stroke': {
            'disability_weight': 0.32,
            'excess_mortality': meid(9311),
            'prevalence': meid(9311),
            'mortality': meid(9311),
            'incidence': meid(9311),
        },
        'ischemic_stroke': {
            'disability_weight': 0.32,
            'excess_mortality': meid(9310),
            'prevalence': meid(9310),
            'mortality': meid(9310),
            'incidence': meid(9310),
        },
        'chronic_stroke': {
            'disability_weight': 0.32,
            'excess_mortality': meid(9312),
            'prevalence': meid(9312),
            'mortality': meid(9312),
        },
}
for k,v in raw_cause_mapping.items():
    v['name'] = k

causes = ConfigTree(raw_cause_mapping)
causes.freeze()
