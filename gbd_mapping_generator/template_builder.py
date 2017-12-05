"""Tools for automatically generating the GBD mapping template"""
from .util import TAB, SPACING, make_record
from .data import get_cause_list, get_etiology_list, get_sequela_list, get_risk_list


gbd_rec_attrs = ()
me_attrs = (('name', 'str'),
            ('gbd_id', 'Union[rid, cid, sid, hid, meid, covid, None]'),)
hs_attrs = (('name', 'str'),
            ('gbd_id', 'hid'),)
eti_attrs = (('name', 'str'),
             ('gbd_id', 'Union[rid, None]'),)
seq_attrs = (('name', 'str'),
             ('gbd_id', 'sid'),
             ('dismod_id', 'meid'),
             ('healthstate', 'Healthstate'),)
restrictions_attrs = (('male_only', 'bool'),
                      ('female_only', 'bool'),
                      ('yll_only', 'bool'),
                      ('yld_only', 'bool'),
                      ('yll_age_start', 'scalar = None'),
                      ('yll_age_end', 'scalar = None'),
                      ('yld_age_start', 'scalar = None'),
                      ('yld_age_end', 'scalar = None'),)
cause_attrs = (('name', 'str'),
               ('gbd_id', 'cid'),
               ('dismod_id', 'Union[meid, _Unknown]'),
               ('restrictions', 'Restrictions'),
               ('sequelae', 'Tuple[Sequela, ...] = None'),
               ('etiologies', 'Tuple[Etiology, ...] = None'),)
tmred_attrs = (('distribution', 'str'),
               ('min', 'scalar'),
               ('max', 'scalar'),
               ('inverted', 'bool'),)
levels_attrs = tuple([('cat1', 'str'), ('cat2', 'str')] + [(f'cat{i}', 'str = None') for i in range(3, 60)])
exp_params_attrs = (('dismod_id', 'meid = None'),
                    ('scale', 'scalar = None'),
                    ('max_rr', 'scalar = None'),
                    ('max_val', 'scalar = None'),
                    ('min_val', 'scalar = None'),)
risk_attrs = (('name', 'str'),
              ('gbd_id', 'rid'),
              ('distribution', 'str'),
              ('affected_causes', 'Tuple[Cause, ...]'),
              ('restrictions', 'Restrictions'),
              ('levels', 'Levels = None'),
              ('tmred', 'Tmred = None'),
              ('exposure_parameters', 'ExposureParameters = None'),)
causes_attrs = tuple([(name, 'Cause') for name in get_cause_list()])
etiologies_attrs = tuple([(name, 'Etiology') for name in get_etiology_list()])
sequelae_attrs = tuple([(name, 'Sequela') for name in get_sequela_list()])
risks_attrs = tuple([(name, 'Risk') for name in get_risk_list()])

gbd_types = {'GbdRecord': {'attrs': gbd_rec_attrs, 'superclass': (None, ()),
                           'docstring': 'Base class for entities modeled in the GBD.'},
             'ModelableEntity': {'attrs': me_attrs, 'superclass': ('GbdRecord', gbd_rec_attrs),
                                 'docstring': 'Container for general GBD ids and metadata.'},
             'Healthstate': {'attrs': hs_attrs, 'superclass': ('ModelableEntity', me_attrs),
                             'docstring': 'Container for healthstate GBD ids and metadata.'},
             'Etiology': {'attrs': eti_attrs, 'superclass': ('ModelableEntity', me_attrs),
                          'docstring': 'Container for etiology GBD ids and metadata.'},
             'Sequela': {'attrs': seq_attrs, 'superclass': ('ModelableEntity', me_attrs),
                         'docstring': 'Container for sequela GBD ids and metadata.'},
             'Restrictions': {'attrs': restrictions_attrs, 'superclass': ('GbdRecord', gbd_rec_attrs),
                              'docstring': 'Container for risk restriction data.'},
             'Cause': {'attrs': cause_attrs, 'superclass': ('ModelableEntity', me_attrs),
                       'docstring': 'Container for cause GBD ids and metadata.'},
             'Tmred': {'attrs': tmred_attrs, 'superclass': ('GbdRecord', gbd_rec_attrs),
                       'docstring': 'Container for theoretical minimum risk exposure distribution data.'},
             'Levels': {'attrs': levels_attrs, 'superclass': ('GbdRecord', gbd_rec_attrs),
                        'docstring': 'Container for categorical risk exposure levels.'},
             'ExposureParameters': {'attrs': exp_params_attrs, 'superclass': ('GbdRecord', gbd_rec_attrs),
                                    'docstring': 'Container for continuous risk exposure distribution parameters'},
             'Risk': {'attrs': risk_attrs, 'superclass': ('GbdRecord', gbd_rec_attrs),
                      'docstring': 'Container for risk factor GBD ids and metadata.'},
             'Causes': {'attrs': causes_attrs, 'superclass': ('GbdRecord', gbd_rec_attrs),
                        'docstring': 'Container for GBD causes.'},
             'Etiologies': {'attrs': etiologies_attrs, 'superclass': ('GbdRecord', gbd_rec_attrs),
                            'docstring': 'Container for GBD etiologies.'},
             'Sequelae': {'attrs': sequelae_attrs, 'superclass': ('GbdRecord', gbd_rec_attrs),
                          'docstring': 'Container for GBD sequelae.'},
             'Risks': {'attrs': risks_attrs, 'superclass': ('GbdRecord', gbd_rec_attrs),
                       'docstring': 'Container for GBD risks.'}, }


def make_module_docstring():
    out = f'"""This code is automatically generated by /ceam_inputs/gbd_mapping/scripts/template_builder.py\n\n'
    out += 'Any manual changes will be lost.\n"""\n'
    return out


def make_imports():
    return '''from typing import Union, Tuple\n'''


def make_ids():
    out = ''
    id_docstring_map = (('meid', 'Modelable Entity ID'),
                        ('rid', 'Risk Factor ID'),
                        ('cid', 'Cause ID'),
                        ('sid', 'Sequela ID'),
                        ('covid', 'Covariate ID'),
                        ('hid', 'Health State ID'))
    for k, v in id_docstring_map:
        out += f'class {k}(int):\n'
        out += TAB + f'"""{v}"""\n'
        out += TAB + 'def __repr__(self):\n'
        out += 2*TAB + f'return "{k}({{:d}})".format(self)\n'
        out += SPACING

    out += 'class scalar(float):\n'
    out += TAB + '"""Raw Measure Value"""\n'
    out += TAB + 'def __repr__(self):\n'
    out += 2 * TAB + 'return "scalar({:f})".format(self)\n'
    out += SPACING

    return out


def make_unknown_flag():
    out = ''
    out += 'class _Unknown:\n'
    out += TAB + '"""Marker for unknown values."""\n'
    out += TAB + 'def __repr__(self):\n'
    out += 2*TAB + 'return "UNKNOWN"\n' + SPACING
    out += 'UNKNOWN = _Unknown()\n' + SPACING
    out += 'class UnknownEntityError(Exception):\n'
    out += TAB + '"""Exception raised when a quantity is requested from ceam_inputs with an `UNKNOWN` id."""\n'
    out += TAB + 'pass\n'
    return out


def make_gbd_record():
    out = '''class GbdRecord:
    """Base class for entities modeled in the GBD."""
    __slots__ = ()
    
    def __contains__(self, item):
        return item in self.__slots__

    def __getitem__(self, item):
        if item in self:
            return getattr(self, item)
        else:
            raise KeyError(item)

    def __iter__(self):
        for item in self.__slots__:
            yield getattr(self, item)

    def __repr__(self):
        out = f'{self.__class__.__name__}('
        for i, slot in enumerate(self.__slots__):
            attr = self[slot]
            if i != 0:
                out += ','
            out += f'\\n{slot}='
            if isinstance(attr, tuple):
                out += '['+','.join([entity.name for entity in attr]) + ']'
            else:
                out += repr(attr)
        return out + ')'
'''
    return out


def build_templates():
    templates = ''
    templates += make_module_docstring()
    templates += make_imports() + SPACING
    templates += make_ids()
    templates += make_unknown_flag() + SPACING
    templates += make_gbd_record() + SPACING
    for entity, info in gbd_types.items():
        if entity == 'GbdRecord':
            continue
        templates += make_record(entity, **info) + SPACING

    return templates
