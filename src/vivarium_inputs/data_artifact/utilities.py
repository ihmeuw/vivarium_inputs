import getpass
import logging
import pathlib
import pkg_resources

import pandas as pd


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
