import argparse
import os

from ceam_inputs.gbd_mapping.generator.template_builder import build_templates
from ceam_inputs.gbd_mapping.generator.cause_builder import build_cause_mapping
from ceam_inputs.gbd_mapping.generator.risk_builder import build_risk_mapping
from ceam_inputs.gbd_mapping.generator.util import get_default_output_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping', choices=['templates', 'causes', 'risks', 'all'], default='all', type=str)
    parser.add_argument('--output_directory', '-o', type=str, default=get_default_output_directory(),)
    args = parser.parse_args()

    func_mapping = {'templates': build_templates, 'causes': build_cause_mapping, 'risks': build_risk_mapping}

    for mapping_type in func_mapping:
        if args.mapping in [mapping_type, 'all']:
            data = func_mapping[mapping_type]()
            with open(os.path.join(args.output_directory, f'{mapping_type}.py'), 'w') as f:
                f.write(data)


