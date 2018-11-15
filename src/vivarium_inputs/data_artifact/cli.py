from bdb import BdbQuit
import os
import shutil
import logging
import getpass
import pathlib
import argparse
import subprocess
from typing import Union

import click

from vivarium.framework.configuration import build_model_specification
from vivarium.framework.plugins import PluginManager
from vivarium.interface.interactive import InteractiveContext
from vivarium.config_tree import ConfigTree


@click.command()
@click.argument('model_specification', type=click.Path(dir_okay=False,
                readable=True))
@click.option('--output-root', '-o', type=click.Path(file_okay=False, writable=True),
              help="Directory to save artifact to. "
                   "Overwrites model specification file")
@click.option('--append', '-a', is_flag=True,
              help="Preserve existing artifact and append to it")
@click.option('--verbose', '-v', is_flag=True,
              help="Turn on debug mode for logging")
@click.option('--pdb', 'debugger', is_flag=True, help='Drop the debugger if an error occurs')
def build_artifact(model_specification, output_root, append, verbose, debugger):
    """
    build_artifact is a program for building data artifacts locally
    from a MODEL_SPECIFICATION file.  It requires access to the J drive and,
    depending on where the output is sent, /ihme as well.

    Any artifact.path specified in the configuration file is guaranteed to
    be overwritten either by the optional output_root option or a predetermined
    path based on username: /ihme/scratch/users/{user}/vivarium_artifacts

    If you are running this job from a qlogin on the new cluster, you must
    specifically request J drive access when you qlogin by adding "-l archive=TRUE"
    to your qsub command."""
    _build_artifact()


@click.command()
@click.argument('model_specification', type=click.Path(dir_okay=False,
                readable=True))
@click.argument('locations', nargs=-1)
@click.option('--project', '-P', default='proj_cost_effect',
              help='Cluster project under which the job will '
                   'be submitted. Defaults to proj_cost_effect')
@click.option('--output-root', '-o', type=click.Path(file_okay=False, writable=True),
              help="Directory to save artifact to. "
                   "Overwrites model specification file")
@click.option('--append', '-a', is_flag=True,
              help="Preserve existing artifact and append to it")
@click.option('--verbose', '-v', is_flag=True,
              help="Turn on debug mode for logging")
@click.option('--error_logs', '-e', is_flag=True,
              help="Write SGE error logs to output location")
@click.option('--memory', '-m', default=10, help="Specifies the amount of memory in G that will be requested for a "
                                                 "job. Defaults to 10. Only applies to the new cluster.")
def multi_build_artifact(model_specification, locations, project,
                   output_root, append, verbose, error_logs, memory):
    """
    multi_build_artifact is a program for building data artifacts on the cluster
    from a MODEL_SPECIFICATION file.

    This script necessarily offloads work to the cluster, and so requires being
    run in the cluster environment.  It will qsub jobs for building artifacts
    under the "proj_cost_effect" project unless a different project is
    specified. Multiple, optional LOCATIONS can be provided to overwrite
    the configuration file. For locations containing spaces, replace the
    space with an underscore and surround any locations containing 
    apostrophes with double quotes, e.g.:

    multi_build_artifact example.yaml Virginia Pennsylvania New_York "Cote_d'Ivoire"

    Any artifact.path specified in the configuration file is guaranteed to
    be overwritten either by the optional output_root or a predetermined path
    based on user: /ihme/scratch/users/{user}/vivarium_artifacts

    If you are running this on the new cluster and find your jobs failing with no
    messages in the log files, consider the memory usage of the job by typing
    "qacct -j <job_id>" and the default memory usage used by this script of 10G.
    The new cluster will kill jobs that go over memory without giving a useful message.
    """

    config_path = pathlib.Path(model_specification).resolve()
    python_context_path = pathlib.Path(shutil.which("python")).resolve()
    script_path = pathlib.Path(__file__).resolve()

    script_args = f"{script_path} {config_path} "
    if output_root:
        script_args += f"--output-root {output_root} "
    if append:
        script_args += f"--append "
    if verbose:
        script_args += f"--verbose "

    error_log_dir = _make_log_dir(output_root) if error_logs else None

    num_locations = len(locations)
    if num_locations > 0:
        script_args += "--location {} "
        for i, location in enumerate(locations):
            location = location.replace("'", "-")
            job_name = f"{config_path.stem}_{location}_build_artifact"
            command = build_submit_command(python_context_path, job_name, project, error_log_dir,
                                           script_args.format(location), memory, archive=True, queue='all.q')
            click.echo(f"submitting job {i+1} of {num_locations} ({job_name})")
            submit_job(command, job_name)
    else:
        job_name = f"{config_path.stem}_build_artifact"
        command = build_submit_command(python_context_path, job_name, project, error_log_dir, script_args,
                                       memory, archive=True, queue='all.q')
        click.echo(f"submitting job {job_name}")
        submit_job(command, job_name)


def submit_job(command: str, name: str):
    """Submit a qsub command to the shell and report the result.

    Parameters
    ----------
    command
        A string containing a qsub job command
    name
        The name of the job described by `command`

    Returns
    -------
        None
    """

    exitcode, response = subprocess.getstatusoutput(command)
    if exitcode:
        click.secho(f"submission of {name} failed with exit code {exitcode}: {response}",
                    fg='red')
    else:
        click.secho(f"submission of {name} succeeded: {response}", fg='green')


def build_submit_command(python_context_path: str, job_name: str, project: str,
                         error_log_dir: Union[str, pathlib.Path], script_args: str, memory: int,
                         slots: int = 2, archive: bool = False, queue: str = 'all.q') -> str:
    """Construct a valid qsub job command string.

    Parameters
    ----------
    python_context_path
        The full path to a python executable under which the job will be
        executed
    job_name
        The name of the job
    project
        The cluster project to run the job under
    error_log_dir
        The directory to save error logs from the build process to
    script_args
        A string comprised of the full path to the script to be executed
        followed by its arguments
    slots
        The number of slots with which to execute the job
    archive
        toggles the archive flag. When true, J drive access will be provided.

    Returns
    -------
    str
        A valid qsub command string
    """

    logs = f"-e {str(error_log_dir)}" if error_log_dir else ""
    command = f"qsub -N {job_name} {logs} "

    if os.environ['SGE_CLUSTER_NAME'] == 'cluster':
        command += f"-l fthread={slots} "
        command += f"-l m_mem_free={memory}G "
        command += f"-P {project} "
        command += f"-q {queue} "
        if archive:
            command += '-l archive=TRUE '
        else:
            command += '-l archive=FALSE '
    elif os.environ['SGE_CLUSTER_NAME'] == 'prod':
        command += f"-pe multi_slot {slots} "
        command += f"-P {project} "
    else:  # dev
        command += f"-pe multi_slot {slots} "

    command += f"-b y {python_context_path} "

    return command + script_args


def _build_artifact():
    """Build a data artifact from a configuration file and optional arguments.

    This function parses command line input for arguments and builds a data
    artifact accordingly. It is meant for use by the click executable
    build_artifact, not as an outward-facing interface. Command line arguments
    are documented and can be viewed by calling this file with the -h flag.

    Returns
    -------
        None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_specification', type=str,
                        help="path to a model_specification file")
    parser.add_argument('--output-root', '-o', type=str, required=False,
                        help="directory to save artifact to. "
                             "Overwrites model_specification file")
    parser.add_argument('--location', type=str, required=False,
                        help="location to get data for. "
                             "Overwrites model_specification file")
    parser.add_argument('--append', '-a', action="store_true",
                        help="Preserve existing artifact and append to it")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Turn on debug mode for logging")
    parser.add_argument('--pdb', action='store_true')
    args = parser.parse_args()

    _setup_logging(args.output_root, args.verbose, args.location,
                   args.model_specification, args.append)

    try:
        main(args.model_specification, args.output_root, args.location, args.append)
    except (BdbQuit, KeyboardInterrupt):
        raise
    except Exception as e:
        logging.exception("Uncaught exception: %s", e)
        if args.pdb:
            import pdb
            import traceback
            traceback.print_exc()
            pdb.post_mortem()
        else:
            raise


def main(model_specification_file, output_root, location, append):
    model_specification = build_model_specification(model_specification_file)
    model_specification.plugins.optional.update({
        "data": {
            "controller": "vivarium_inputs.data_artifact.ArtifactBuilder",
            "builder_interface": "vivarium_public_health.dataset_manager.ArtifactManagerInterface",
        }})

    logging.debug("Configuring simulation")
    plugin_config = model_specification.plugins
    component_config = model_specification.components
    simulation_config = model_specification.configuration

    simulation_config.input_data.location = get_location(location, simulation_config)
    simulation_config.input_data.artifact_path = get_output_path(model_specification_file,
                                                                 output_root, location)
    simulation_config.input_data.append_to_artifact = append

    plugin_manager = PluginManager(plugin_config)
    component_config_parser = plugin_manager.get_plugin('component_configuration_parser')
    components = component_config_parser.get_components(component_config)
    
    logging.debug("Setting up simulation")
    simulation = InteractiveContext(simulation_config, components, plugin_manager)
    simulation.setup()


def get_location(location_arg: str, configuration: ConfigTree) -> str:
    """Resolve the model location

    This function takes in to account the model configuration and the passed
    location argument. User-passed arguments supercede the configuration.

    Parameters
    ----------
    location_arg
        the location argument passed to the click executable
    configuration
        A model configuration object

    Returns
    -------
    str
        The resolved location name
    """

    if location_arg:
        return location_arg.replace('_', ' ').replace("-", "'")
    elif contains_location(configuration):
        return configuration.input_data.location
    else:
        raise argparse.ArgumentError("specify a location or include "
                                     "configuration.input_data.location "
                                     "in model specification")


def contains_location(configuration: ConfigTree) -> bool:
    """Check if location is specified in the configuration

    Parameters
    ----------
    configuration
        A model configuration

    Returns
    -------
    bool
        True if a location is specified else False
    """

    return ('input_data' in configuration and
            'location' in configuration.input_data and
            configuration.input_data.location)


def get_output_path(configuration_arg: str, output_root_arg: str,
                    location_arg: str) -> str:
    """Resolve the correct model output path

    Takes in to account the model specification and passed arguments.
    User-passed arguments supercede the configuration. Guaranteed to
    clobber the configuration output path with either what the user
    provided or a default. The file name is taken from the model
    configuration file name. Location info is added if a location argument
    was provided.

    Parameters
    ----------
    configuration_arg
        Path to the model configuration file
    output_root_arg
        The output_root argument passed to the click executable
    location_arg
        The location argument passed to the click executable

    Returns
    -------
        A PathLike object containing the path to the output hdf file
    """

    configuration_path = pathlib.Path(configuration_arg)

    output_base = get_output_base(output_root_arg)

    if location_arg:
        output_path = output_base / (configuration_path.stem + f'_{location_arg}.hdf')
    else:
        output_path = output_base / (configuration_path.stem + '.hdf')
    return str(output_path)


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


def _make_log_dir(output_root):
    output_log_dir = get_output_base(output_root) / 'logs'

    if not output_log_dir.is_dir():
        output_log_dir.mkdir()

    return output_log_dir


def _setup_logging(output_root, verbose, location,
                   model_specification, append):
    """ Setup logging to write to a file in the output directory

    Log file named as {model_specification}_{location}_build_artifact.log
    to match naming format of qsubbed jobs. File saved in output directory
    (either passed by user or default)/logs. Raises error if that output
    directory is not found.

    """

    output_log_dir = _make_log_dir(output_root)

    log_level = logging.DEBUG if verbose else logging.ERROR
    log_tag = f'_{location}' if location is not None else ''
    spec_name = pathlib.Path(model_specification).resolve().stem
    log_name = f'{output_log_dir}/{spec_name}{log_tag}_build_artifact.log'

    logging.basicConfig(level=log_level,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt="%m-%d-%y %H:%M",
                        filename=log_name,
                        filemode='a' if append else 'w')


if __name__ == "__main__":
    _build_artifact()
