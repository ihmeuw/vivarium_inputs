import os
from bdb import BdbQuit
import logging
import getpass
import pathlib
import argparse
import subprocess

import click

from vivarium.framework.configuration import build_model_specification
from vivarium.framework.plugins import PluginManager
from vivarium.interface.interactive import InteractiveContext
from vivarium.config_tree import ConfigTree


@click.command()
@click.argument('model_specification', type=click.Path(dir_okay=False,
                readable=True))
@click.argument('project')
@click.argument('locations', nargs=-1)
@click.option('--output_root', type=click.Path(exists=True, file_okay=False, writable=True),
              help="Directory to save artifact result in")
@click.option('--from_scratch', type=click.BOOL, default=True,
              help="Do not reuse any data in the artifact, if any exists")
def build_artifact(model_specification, project, locations,
                   output_root, from_scratch):
    """
    build_artifact is a program for building data artifacts from a
    SIMULATION_CONFIGURATION file. The work is offloaded to the cluster under
    the provided PROJECT. Multiple, optional LOCATIONS can be provided to
    overwrite the configuration file.

    Any artifact.path specified in the configuration file is guaranteed to
    be overwritten either by the optional output_root or a predetermined path
    based on user: /ihme/scratch/{user}/vivarium_artifacts
    """

    config_path = pathlib.Path(model_specification).resolve()
    python_context_path = pathlib.Path(subprocess.getoutput("which python")).resolve()
    script_path = pathlib.Path(__file__).resolve()

    script_args = f"{script_path} {config_path} "
    if output_root:
        script_args += f"--output_root {output_root} "
    if from_scratch:
        script_args += f"--from_scratch "

    num_locations = len(locations)
    if num_locations > 0:
        script_args += "--location {}"
        for i in range(num_locations):
            location = locations[i]
            job_name = f"{config_path.stem}_{location}_build_artifact"
            command = build_submit_command(python_context_path, job_name,
                                           project,
                                           script_args.format(location))
            click.echo(f"submitting job {i} of {num_locations} ({job_name})")
            submit_job(command, job_name)
    else:
        job_name = f"{config_path.stem}_build_artifact"
        command = build_submit_command(python_context_path, job_name,
                                       project, script_args)
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
                         script_args: str, slots: int = 2) -> str:
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
    script_args
        A string comprised of the full path to the script to be executed
        followed by its arguments
    slots
        The number of slots with which to execute the job

    Returns
    -------
    str
        A valid qsub command string
    """

    return (f"qsub -N {job_name} -P {project} -pe multi_slot {slots} " +
            f"-b y {python_context_path} " + script_args)


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
    parser.add_argument('--output_root', type=str, required=False,
                        help="directory to save artifact to. "
                             "Overwrites model_specification file")
    parser.add_argument('--location', type=str, required=False,
                        help="location to get data for. "
                             "Overwrites model_specification file")
    parser.add_argument('--from_scratch', '-s', action="store_true",
                        help="Do not reuse any data in the artifact, "
                             "if any exists")
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--pdb', action='store_true')
    args = parser.parse_args()

    _setup_logging(args.output_root, args.verbose, args.location,
                    args.model_specification, args.from_scratch)

    try:
        main(args.model_specification, args.output_root, args.location, args.from_scratch)
    except (BdbQuit, KeyboardInterrupt):
        raise
    except Exception as e:
        logging.exception("Uncaught exception: {}".format(e))
        if args.pdb:
            import pdb
            import traceback
            traceback.print_exc()
            pdb.post_mortem()
        else:
            raise


def main(model_specification_file, output_root, location, from_scratch):
    logging.debug("Building model specification from %s", model_specification_file)
    model_specification = build_model_specification(model_specification_file)
    model_specification.plugins.optional.update({
        "data": {
            "controller": "vivarium_inputs.data_artifact.ArtifactBuilder",
            "builder_interface": "vivarium_public_health.dataset_manager.ArtifactManagerInterface",
        }})

    plugin_config = model_specification.plugins
    component_config = model_specification.components
    simulation_config = model_specification.configuration

    simulation_config.input_data.location = get_location(location, simulation_config)
    simulation_config.input_data.artifact_path = get_output_path(model_specification_file,
                                                                 output_root, location)
    simulation_config.input_data.append_to_artifact = not from_scratch

    plugin_manager = PluginManager(plugin_config)
    component_config_parser = plugin_manager.get_plugin('component_configuration_parser')
    components = component_config_parser.get_components(component_config)
    
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
        return location_arg
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


def get_output_base(output_root_arg: str) -> str:
    """Resolve the correct output directory

    Defaults to /ihme/scratch/users/{user}/vivarium_artifacts/
    if no user passed output directory. Makes default directory
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
        if not os.isdir(output_base):
            raise FileNotFoundError("The passed output directory %s does not exist", output_base)
    else:
        output_base = (pathlib.Path('/share') / 'scratch' / 'users' /
                       getpass.getuser() / 'vivarium_artifacts')

        if not os.isdir(output_base):
            os.makedirs(output_base)

    return str(output_base)


def _setup_logging(output_root_arg, verbose_arg, location_arg,
                   model_specification_arg, from_scratch_arg):
    """ Setup logging to write to a file in the output directory

    Log file named as {model_specification}_{location}_build_artifact.log
    to match naming format of qsubbed jobs. File saved in output directory
    (either passed by user or default)/logs. Raises error if that output
    directory is not found.

    """

    # this will throw an error if the passed output path doesn't exist but the log hasn't been set up yet
    output_log_dir = pathlib.Path(get_output_base(output_root_arg)) / 'logs'
    if not os.isdir(output_log_dir):
        os.makedirs(output_log_dir)

    log_level = logging.DEBUG if verbose_arg else logging.ERROR
    log_tag = "_{}".format(location_arg) if location_arg is not None else ""
    log_name = "{}/{}{}_build_artifact.log".format(str(output_log_dir),
                                                   pathlib.Path(model_specification_arg).resolve().stem,
                                                   log_tag)
    logging.basicConfig(level=log_level,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt="%m-%d-%y %H:%M",
                        filename=log_name,
                        filemode='w' if from_scratch_arg else 'a')


if __name__ == "__main__":
    _build_artifact()
