from enum import Enum, IntEnum, unique

class meid(int):
    def __repr__(self):
        return 'meid({:d})'.format(self)

raw_cause_mapping = [
        {
            'name': 'heart_attack',
            'excess_mortality': meid(1814),
            'prevalence': meid(1814),
            'mortality': meid(1814),
            'incidence': meid(1814),
        },
        {
            'name': 'mild_heart_failure',
            'disability_weight': meid(1821),
            'excess_mortality': meid(2412),
            'prevalence': meid(1821),
            'mortality': meid(2412),
        },
        {
            'name': 'moderate_heart_failure',
            'disability_weight': meid(1822),
            'excess_mortality': meid(2412),
            'prevalence': meid(1822),
        },
        {
            'name': 'severe_heart_failure',
            'disability_weight': meid(1823),
            'excess_mortality': meid(2412),
            'prevalence': meid(1823),
        },
        {
            'name': 'angina_not_due_to_MI',
            'incidence': meid(1817),
        },
        {
            'name': 'asymptomatic_angina',
            'disability_weight': meid(1823),
            'excess_mortality': meid(1817),
            'prevalence': meid(3102),
            'mortality': meid(1817),
        },
        {
            'name': 'mild_angina',
            'disability_weight': meid(1818),
            'excess_mortality': meid(1817),
            'prevalence': meid(1818),
        },
        {
            'name': 'moderate_angina',
            'disability_weight': meid(1819),
            'excess_mortality': meid(1817),
            'prevalence': meid(1819),
        },
        {
            'name': 'severe_angina',
            'disability_weight': meid(1820),
            'excess_mortality': meid(1817),
            'prevalence': meid(1820),
        },
        {
            'name': 'asymptomatic_ihd',
            'disability_weight': meid(3233),
            'excess_mortality': 0.0,
            'prevalence': meid(3233),
            'mortality': meid(3233),
        },
        {
            'name': 'hemorrhagic_stroke',
            'disability_weight': 0.32,
            'excess_mortality': meid(9311),
            'prevalence': meid(9311),
            'mortality': meid(9311),
            'incidence': meid(9311),
        },
        {
            'name': 'ischemic_stroke',
            'disability_weight': 0.32,
            'excess_mortality': meid(9310),
            'prevalence': meid(9310),
            'mortality': meid(9310),
            'incidence': meid(9310),
        },
        {
            'name': 'chronic_stroke',
            'disability_weight': 0.32,
            'excess_mortality': meid(9312),
            'prevalence': meid(9312),
            'mortality': meid(9312),
        },
        {
            'name': 'diarrhea',
            'disability_weight': 0.23,
            'excess_mortality': meid(1181),
            'prevalence': meid(1181),
            'incidence': meid(1181),
            'mortality': meid(1181),
        },
        {
            'name': 'death_due_to_severe_diarrhea',
            'mortality': meid(1181),
        },
]

class DotDict:
    def __init__(self, data):
        self.__data = data

    def __getattr__(self, name):
        if name in self.__data:
            return self.__data[name]
        raise AttributeError()

    def __getitem__(self, key):
        return self.__data[key]

    def __contains__(self, key):
        return key in self.__data

    def __repr__(self):
        return repr(self.__data)

assert len({d['name'] for d in raw_cause_mapping}) == len(raw_cause_mapping), "Duplicate name in causes"
causes = DotDict({d['name']:DotDict(d) for d in raw_cause_mapping})

