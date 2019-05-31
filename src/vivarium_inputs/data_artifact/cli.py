from bdb import BdbQuit
import os
import shutil
import logging
import pathlib
import argparse
import subprocess
from typing import Union, List, Dict
import click

from vivarium.framework.configuration import build_model_specification
from vivarium.framework.plugins import PluginManager
from vivarium.interface.interactive import InteractiveContext
from vivarium.config_tree import ConfigTree

from vivarium_inputs.data_artifact.aggregation import disaggregate
from vivarium_inputs.data_artifact import utilities


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
@click.option('--memory', '-m', default=50, help="Specifies the amount of memory in G that will be requested for a "
                                                 "job. Defaults to 50. Only applies to the new cluster.")
def multi_build_artifact(model_specification, locations, project, output_root, append, verbose, error_logs, memory):
    """
    multi_build_artifact is a program for building data artifacts on the cluster
    from a MODEL_SPECIFICATION file. It will generate a single artifact containing
    the data for multiple locations (up to 5 locations only).

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
    "qacct -j <job_id>" and the default memory usage used by this script of 50G.
    The new cluster will kill jobs that go over memory without giving a useful message.
    """

    if len(set(locations)) > 5:
        raise ValueError(f'We can make an artifact for up to 5 locations. '
                         f'You provided {len(set(locations))} locations.')

    if len(set(locations)) < 1:
        raise ValueError('You must provide a list of locations for this artifact. You did not provide any.')

    config_path = pathlib.Path(model_specification).resolve()
    python_context_path = pathlib.Path(shutil.which("python")).resolve()
    script_path = pathlib.Path(__file__).resolve()
    output_root = utilities.get_output_base(output_root)

    if append:
        click.secho('Pre-processing the existing artifact for appending. Please wait for the job submission messages. '
                    'It may take long, especially if your artifact already includes many locations. '
                    'Please DO NOT QUIT during the process.', fg='blue')
        existing_locations = disaggregate(config_path.stem, output_root)
    else:
        existing_locations = {}

    new_locations = set(locations).difference(set(existing_locations)) if set(locations) > set(existing_locations) else {}
    new_locations = [l.replace("'", "-") for l in new_locations]
    existing_locations = [l.replace("'", "-") for l in existing_locations]

    script_args = f"{script_path} {config_path} --output-root {output_root}"
    if verbose:
        script_args += f" --verbose "

    error_log_dir = utilities.make_log_dir(output_root) if error_logs else None
    jids = list()

    existing_locations_jobs = {loc: f"{config_path.stem}_{loc}_build_artifact" for loc in existing_locations}
    new_locations_jobs = {loc: f"{config_path.stem}_{loc}_build_artifact" for loc in new_locations}

    existing_loc_commands = {loc: build_submit_command(python_context_path, job, project, error_log_dir, script_args,
                                                       memory, archive=True, queue='all.q') + f'--location {loc} --append'
                             for loc, job in existing_locations_jobs.items()}

    new_loc_commands = {loc: build_submit_command(python_context_path, job, project, error_log_dir, script_args, memory,
                                                  archive=True, queue='all.q') + f'--location {loc}'
                        for loc, job in new_locations_jobs.items()}

    jids.extend(submit_jobs_multi_locations(existing_locations, existing_loc_commands, existing_locations_jobs))
    jids.extend(submit_jobs_multi_locations(new_locations, new_loc_commands, new_locations_jobs))
    jids = ",".join(jids)

    locations = [l.replace("'", "-") for l in locations]
    aggregate_script = pathlib.Path(__file__).parent / 'aggregation.py'
    aggregate_args = f'--locations {" ".join(locations)} --output_root {output_root} ' \
        f'--config_path {config_path} {"--verbose" if verbose else ""}'
    aggregate_job_name = f"{config_path.stem}_aggregate_artifacts"
    aggregate_command = build_submit_command(python_context_path, aggregate_job_name, 
                                             project, error_log_dir, f'{aggregate_script} {aggregate_args}', memory=35,
                                             archive=True, queue='all.q', hold=True, jids=jids, slots=15)
    submit_job(aggregate_command, aggregate_job_name)


def submit_jobs_multi_locations(locations: List, commands: Dict, job_names: Dict) -> List:
    """ submit a qsub command for the multiple location jobs and collect
        the jobids which were successfully submitted and give click messages
        to the user
    Parameters
    ----------
    locations
        set of name of locations for the jobs to be submitted
    commands
        dictionary for location(key) and matching qsub command for
        that location(value)
    job_names
        dictionary for location(key) and matching job_name for
        that location(value)

    Returns
    -------
        job_ids

    """
    job_ids = []
    num_locations = len(locations)
    if len(locations) > 0:
        for i, location in enumerate(locations):
            click.echo(f"submitting job {i + 1} of {num_locations} ({job_names[location]})")
            jobid = submit_job(commands[location], job_names[location])
            job_ids.append(jobid)
    return job_ids


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
                         error_log_dir: Union[str, pathlib.Path], script_args: str, memory: int,
                         slots: int = 8, archive: bool = False, queue: str = 'all.q', hold=False, jids=None) -> str:
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
    queue
        name of the queue to use. currently 'all.q' by default
    hold
        whether the job needs to wait until the given job_ids are completed
    jids
        when the hold is True, jids are the jobs that this new job is held for completion.

    Returns
    -------
    str
        A valid qsub command string
    """

    logs = f"-e {str(error_log_dir)} " if error_log_dir else ""
    command = f"qsub -N {job_name} {logs} "
    if hold:
        command = f"qsub -hold_jid {jids} -N {job_name} {logs}"

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

    output_base = utilities.get_output_base(output_root_arg)

    if location_arg:
        output_path = output_base / (configuration_path.stem + f'_{location_arg}.hdf')
    else:
        output_path = output_base / (configuration_path.stem + '.hdf')
    return str(output_path)


if __name__ == "__main__":
    _build_artifact()
