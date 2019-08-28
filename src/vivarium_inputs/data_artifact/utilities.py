import getpass
import logging
import pathlib
import pkg_resources
import subprocess
import tempfile
from typing import Callable

import pandas as pd
import tables

import gbd_mapping
from vivarium_inputs import mapping_extension


def get_versions():
    libraries = ['vivarium', 'vivarium_inputs', 'vivarium_public_health', 'gbd_mapping']
    return {k: pkg_resources.get_distribution(k).version for k in libraries}


def get_output_base(output_root_arg: str) -> pathlib.Path:
    """Resolve the correct output directory

    Defaults to /ihme/scratch/users/{user}/vivarium_artifacts/
    if no user passed output directory. Makes output directory
    if doesn't already exist.

    Parameters
    ----------
    output_root_arg
        The output_root argument passed to the click executable

    Returns
    -------
        A PathLike object containing the path to the output directory
    """

    if output_root_arg:
        output_base = pathlib.Path(output_root_arg).resolve()
    else:
        output_base = (pathlib.Path('/share') / 'scratch' / 'users' /
                       getpass.getuser() / 'vivarium_artifacts')

    if not output_base.is_dir():
        output_base.mkdir(parents=True)

    return output_base


def make_log_dir(output_root):
    output_log_dir = get_output_base(output_root) / 'logs'

    if not output_log_dir.is_dir():
        output_log_dir.mkdir()

    return output_log_dir


def setup_logging(output_root, verbose, tag, model_specification, append):
    """ Setup logging to write to a file in the output directory

    Log file named as {model_specification}_{location}_build_artifact.log
    to match naming format of qsubbed jobs. File saved in output directory
    (either passed by user or default)/logs. Raises error if that output
    directory is not found.

    """

    output_log_dir = make_log_dir(output_root)

    log_level = logging.DEBUG if verbose else logging.ERROR
    log_tag = f'_{tag}' if tag is not None else ''
    spec_name = pathlib.Path(model_specification).resolve().stem
    log_name = f'{output_log_dir}/{spec_name}{log_tag}_build_artifact.log'

    logging.basicConfig(level=log_level,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt="%m-%d-%y %H:%M",
                        filename=log_name,
                        filemode='a' if append else 'w')


def split_interval(data, interval_column, split_column_prefix):
    if isinstance(data, pd.DataFrame) and interval_column in data.index.names:
        data[f'{split_column_prefix}_end'] = [x.right for x in data.index.get_level_values(interval_column)]
        if not isinstance(data.index, pd.MultiIndex):
            data[f'{split_column_prefix}_start'] = [x.left for x in data.index.get_level_values(interval_column)]
            data = data.set_index([f'{split_column_prefix}_start', f'{split_column_prefix}_end'])
        else:
            interval_starts = [x.left for x in data.index.levels[data.index.names.index(interval_column)]]
            data.index = (data.index.rename(f'{split_column_prefix}_start', interval_column)
                          .set_levels(interval_starts, f'{split_column_prefix}_start'))
            data = data.set_index(f'{split_column_prefix}_end', append=True)
    return data


# TODO: test with random stuff out of the mapping
def parse_entity(entity_type: str, entity_name: str) -> gbd_mapping.ModelableEntity:
    """Parse an entity type and name into a gbd_mapping object,.

    Parameters
    ----------
    entity_type
        an entity_type from the gbd_mapping.
    entity_name
        an entity name.

    Returns
    -------
        The ModelableEntity representation of the entity.
    """
    plurality_mapping = {'sequela': 'sequelae',
                         'etiology': 'etiologies',
                         'health_technology': 'health_technologies',
                         'healthcare_entity': 'healthcare_entities'}
    plural_entity_type = entity_type + 's' if entity_type not in plurality_mapping else plurality_mapping[entity_type]
    if entity_type in ['alternative_risk_factor', 'health_technology', 'healthcare_entity']:
        return getattr(mapping_extension, plural_entity_type)[entity_name]
    else:
        return getattr(gbd_mapping, plural_entity_type)[entity_name]


def handle_tables_versions(get_measure: Callable) -> Callable:
    """Wraps get_measure to handle tables versioning issues.
    The only acceptable argument is the get_measure function.

    The wrapped function handles the tables version issue by catching the
    decompression error that arises and calling out to a get_measure cli
    endpoint under an environment with updated tables.

    Returns
    -------
        A wrapped version of get_measure that handles tables version issues.
    """
    # TODO: Place the environment somewhere central
    new_tables_get_measure = '/home/cody/miniconda3/envs/sam/bin/get_measure'

    # TODO: check version. Raise if not the same.

    def wrapped(entity, measure, location):

        try:
            return get_measure(entity, measure, location)
        except tables.exceptions.HDF5ExtError as e:
            if 'Blosc decompression error' in e:
                with tempfile.NamedTemporaryFile() as tmpf:
                    subprocess.run([new_tables_get_measure, entity.type, entity.name, measure, location,
                                   '--fname', tmpf.name], check=True)
                    return pd.read_pickle(tmpf, compression='gzip')
            else:
                raise e

    return wrapped
