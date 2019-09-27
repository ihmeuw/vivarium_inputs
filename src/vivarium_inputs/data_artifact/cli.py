from bdb import BdbQuit
import shutil
import logging
import pathlib
import argparse
import subprocess
from typing import Union
import click
import yaml

from vivarium.framework.engine import SimulationContext

from vivarium_inputs.data_artifact import utilities


@click.command()
@click.argument('model_specification', type=click.Path(dir_okay=False))
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
    to your qsub command.

    Please have at least 50GB of memory on your qlogin."""
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
                                                 "job. Defaults to 10.")
def multi_build_artifact(model_specification, locations, project, output_root,
                    append, verbose, error_logs, memory):
    """
    multi_build_artifact is a program for building data artifacts on the cluster
    from a MODEL_SPECIFICATION file. It will generate one artifact per given
    location.

    This script necessarily offloads work to the cluster, and so requires being
    run in the cluster environment.  It will qsub jobs for building artifacts
    under the "proj_cost_effect" project unless a different project is
    specified. Multiple, optional LOCATIONS can be provided to overwrite
    the configuration file. For locations containing spaces, replace the
    space with an underscore and surround any locations containing
    apostrophes with double quotes, e.g.:

    multi_build_artifact example.yaml India South_Korea "Cote_d'Ivoire"

    Any artifact.path specified in the configuration file is guaranteed to
    be overwritten either by the optional output_root or a predetermined path
    based on user: /ihme/scratch/users/{user}/vivarium_artifacts

    If you find your jobs failing with no messages in the log files, consider
    the memory usage of the job by typing "qacct -j <job_id>" and the default
    memory usage used by this script of 10G. The cluster will kill jobs that
    go over memory without giving a useful message.
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

    error_log_dir = utilities.make_log_dir(output_root) if error_logs else None

    num_locations = len(locations)
    if num_locations > 0:
        script_args += "--location {} "
        for i, location in enumerate(locations):
            location = location.replace("'", "-")
            job_name = f"{config_path.stem}_{location}_build_artifact"
            command = build_submit_command(python_context_path, job_name, project, error_log_dir,
                                           script_args.format(location), memory, archive=True, queue='all.q')
            click.echo(f"Submitting job {i+1} of {num_locations} ({job_name}).")
            submit_job(command, job_name)
    else:
        job_name = f"{config_path.stem}_build_artifact"
        command = build_submit_command(python_context_path, job_name, project, error_log_dir, script_args,
                                       memory, archive=True, queue='all.q')
        click.echo(f"Submitting job {job_name}.")
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
        jobid
    """

    exitcode, response = subprocess.getstatusoutput(command)
    if exitcode:
        click.secho(f"submission of {name} failed with exit code {exitcode}: {response}",
                    fg='red')
        jid = None
    else:
        click.secho(f"submission of {name} succeeded: {response}", fg='green')
        jid = response.split(' ')[2]
    return jid


def build_submit_command(python_context_path: str, job_name: str, project: str,
                         error_log_dir: Union[str, pathlib.Path], script_args: str,
                         memory: int, archive: bool = False,
                         queue: str = 'all.q') -> str:
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
    memory
        The amount of memory to request, in GB.
    archive
        toggles the archive flag. When true, J drive access will be provided.
    queue
        name of the queue to use. currently 'all.q' by default

    Returns
    -------
    str
        A valid qsub command string
    """

    logs = f"-e {str(error_log_dir)} " if error_log_dir else ""
    command = f"qsub -N {job_name} {logs} "

    command += "-l fthread=1 "
    command += f"-l m_mem_free={memory}G "
    command += f"-P {project} "
    command += f"-q {queue} "
    if archive:
        command += '-l archive '

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

    utilities.setup_logging(args.output_root, args.verbose, args.location, args.model_specification, args.append)

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
    logging.debug("Setting up simulation")

    config = {
        'input_data': {
            'location': get_location(location, model_specification_file),
            'artifact_path': get_output_path(model_specification_file, output_root, location),
            'append_to_artifact': append
        }
    }
    plugin_config = {
        "required": {
            "data": {
                "controller": "vivarium_inputs.data_artifact.ArtifactBuilder",
                "builder_interface": "vivarium.framework.artifact.ArtifactInterface",
            }
        }
    }
    simulation = SimulationContext(model_specification_file, configuration=config, plugin_configuration=plugin_config)
    simulation.setup()


def get_location(location_arg: str, model_spec: str) -> str:
    """Resolve the model location

    This function takes in to account the model configuration and the passed
    location argument. User-passed arguments supercede the configuration.

    Parameters
    ----------
    location_arg
        the location argument passed to the click executable
    model_spec
        File path to the model specification

    Returns
    -------
    str
        The resolved location name
    """
    configuration = yaml.full_load(pathlib.Path(model_spec).open())['configuration']
    if location_arg:
        return location_arg.replace('_', ' ').replace("-", "'")
    elif contains_location(configuration):
        return configuration['input_data']['location']
    else:
        raise argparse.ArgumentError("specify a location or include "
                                     "configuration.input_data.location "
                                     "in model specification")


def contains_location(configuration: dict) -> bool:
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
            'location' in configuration['input_data'] and
            configuration['input_data']['location'])


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

    output_base = utilities.get_output_base(output_root_arg)

    if location_arg:
        output_path = output_base / (configuration_path.stem + f'_{location_arg}.hdf')
    else:
        output_path = output_base / (configuration_path.stem + '.hdf')
    return str(output_path)


if __name__ == "__main__":
    _build_artifact()
