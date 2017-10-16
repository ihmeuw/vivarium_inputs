TEXTWIDTH = 120

def make_class_sig(name, superclass=None, docstring=None):
    out = f'class {name}'
    if superclass:
        out += '({superclass})'
    out += ':\n'
    if docstring:
        out += f'    """{docstring}"""\n'


def make_imports():
    return 'from typing import Union, Tuple\n'


def make_ids():
    out = ''
    id_docstring_map =(('meid', 'Modelable Entity ID'),
                       ('rid', 'Risk Factor ID'),
                       ('cid', 'Cause ID'),
                       ('sid', 'Sequela ID'),
                       ('hid', 'Health State ID'),
                       ('scalar', 'Raw Measure Value'))
    for k, v in id_docstring_map:
        out += f'class {k}(int):\n'
        out += f'    """{v}"""\n'
        out +=  '    def __repr__(self):\n'
        out += f'        return "{k}({{:d}}).format(self)\n'

    return out


def make_unknown_flag():
    out = ''
    out += 'class _Unknown:\n'
    out += '    """Marker for unknown values."""\n'
    out += '    def __repr__(self):\n'
    out += '        return "UNKNOWN"\n\n'
    out += 'UNKNOWN = _Unknown()\n\n'
    out += 'class UnknownEntityError(Exception):\n'
    out += '    """Exception raised when a quantity is requested from ceam_inputs with an `UNKNOWN` id."""\n'
    out += '    pass\n'
    return out


def make_base_record():
    out = ''
    out +=  'class GbdRecord:\n'
    out +=  '    """Base class for entities modeled in the GBD."""\n'
    out +=  '    __slots__ = ()\n\n'
    out +=  '    def __contains__(self, item):\n'
    out +=  '        return item in self.__slots__\n\n'
    out +=  '    def __getitem__(self, item):\n'
    out +=  '        if item in self:\n'
    out +=  '            return getattr(self.item)\n'
    out +=  '        else:\n'
    out +=  '            raise KeyError\n\n'
    out +=  '   def __iter__(self):\n'
    out +=  '       for item in self.__slots__:\n'
    out +=  '           yield getattr(self, item)\n\n'
    out +=  '   def __repr__(self):\n'
    out +=  '       return "{{}}({{}})".format(self.__class__.__name__,\n'
    out += r'                                  ",\n".join(["{{}}={{}}".format(name, self[name])' + '\n'
    out +=  '                                             for name in self.__slots__]))\n'





if __name__ == '__main__':
    spacing = '\n\n'
    template = ''
    template += make_imports() + spacing
    template += make_ids() + spacing
    template += make_unknown_flag() + spacing
    template += make_base_record() + spacing
    with open('templates.py', 'w') as f:
        f.write(template)

