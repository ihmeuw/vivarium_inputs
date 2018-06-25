import os

import numpy as np


TEXTWIDTH = 118  # type: int
TAB = '    '  # type: str
SPACING = '\n\n'  # type: str


def clean_entity_list(raw_entity_series):
    replace_with_underscore_chars = ['/', '(', ')', ' – ', ' - ', '-', ' ', ',', '–', '____', '___', '__']
    replace_chars = {char: '_' for char in replace_with_underscore_chars}
    replace_chars.update({"'": '', 'é': 'e', '<': 'less_than_', '>': 'greater_than_'})
    cleaned_up_entities = []
    for entity in list(raw_entity_series):
        entity = str(entity)
        # Clean up the string
        for char, rep_char in replace_chars.items():
            entity = entity.replace(char, rep_char)

        entity = entity.lower().rstrip().rstrip('_')
        cleaned_up_entities.append(entity)
    return cleaned_up_entities


def clean_risk_me(me_names):
    replace = {' ': '_', '&': 'and', '_interpolated_annual_results': ''}
    out = []
    for me_name in me_names:
        me_name = ''.join(me_name.split(',')[1:]).lower().lstrip()
        for k, v in replace.items():
            me_name = me_name.replace(k, v)
        out.append(me_name)
    return out


def to_id(number, id_type):
    if np.isnan(number):
        return 'UNKNOWN'
    else:
        return id_type + f'({int(number)})'


def make_class_sig(name, superclass=None, docstring=None):
    out = f'class {name}'
    if superclass[0]:
        out += f'({superclass[0]})'
    out += ':\n'
    if docstring:
        out += TAB + f'"""{docstring}"""\n'
    return out


def make_slots(field_list):
    declaration = TAB + '__slots__ = ('
    offset = len(declaration)

    out = declaration
    char_count = offset

    for field in field_list:
        field = f"'{field}', "
        field_width = len(field)
        if char_count == offset:
            out += field
            char_count += field_width
        elif char_count + field_width > TEXTWIDTH:
            out = out[:-1] + '\n' + ' '*offset + field
            char_count = offset + field_width
        else:
            out += field
            char_count += field_width

    out += ')\n\n'

    return out


def make_init(field_dtype_tuples):
    declaration = TAB + 'def __init__('
    offset = len(declaration)

    out = declaration + 'self,'

    if len(field_dtype_tuples) > 255:
        out += ' ' + '**kwargs):\n'
    else:
        out += '\n'
        for field_name, data_type in field_dtype_tuples:
            out += ' '*offset + field_name + ': ' + data_type + ',\n'
        out = out[:-1] + ' ):\n'
    return out


def make_super_call(superclass):
    field_names = [attr[0] for attr in superclass[1]]
    declaration = 2*TAB + 'super().__init__('
    if not field_names:
        return declaration + ')\n'

    offset = len(declaration)
    out = declaration
    for field in field_names:
        out += f'{field}={field},\n'
        out += ' '*offset

    out = out[:-offset-2] + ')\n'
    return out


def make_attribute_assignment(field_names):
    offset = 8

    out = ''
    if len(field_names) > 255:
        for field in field_names:
            out += ' '*offset + f"self.{field} = kwargs.get('{field}')\n"
    else:
        for field in field_names:
            out += ' '*offset + f'self.{field} = {field}\n'
    return out


def make_record(name, attrs=None, superclass=None, docstring=None):
    out = ''
    out += make_class_sig(name, superclass, docstring)
    out += make_slots([attr[0] for attr in attrs])
    out += make_init(attrs)
    if superclass:
        out += make_super_call(superclass)
    out += make_attribute_assignment([attr[0] for attr in attrs])
    return out


def get_default_output_directory():
    here = os.path.realpath(__file__)
    return os.path.realpath(os.path.dirname(here) + '/../ceam_inputs/gbd_mapping/')

