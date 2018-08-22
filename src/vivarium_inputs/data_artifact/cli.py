import os
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
@click.option('--output_root', type=click.Path(file_okay=False, writable=True),
              help="Directory to save artifact result in")
@click.option('--from_scratch', type=click.BOOL, default=True,
              help="Do not reuse any data in the artifact, if any exists")
def build_artifact(simulation_configuration, project, locations,
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

    config_path = pathlib.Path(simulation_configuration).resolve()
    python_context_path = pathlib.Path(subprocess.getoutput("which python")).resolve()
    script_path = pathlib.Path(__file__).resolve()

    script_args = f"{script_path} {config_path} "
    if output_root:
        script_args += f"--output_root {output_root} "
    if from_scratch:
        script_args += f"--from_scratch "

    if len(locations) > 0:
        script_args += "--location {}"
        for location in locations:
            job_name = f"{config_path.stem}_{location}_build_artifact"
            command = build_submit_command(python_context_path, job_name,
                                           project,
                                           script_args.format(location))
            submit_job(command, job_name)
    else:
        job_name = f"{config_path.stem}_build_artifact"
        command = build_submit_command(python_context_path, job_name,
                                       project, script_args)
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
        click.secho(f"{name} failed with exit code {exitcode}: {response}",
                    fg='red')
    else:
        click.secho(f"{name} succeeded: {response}", fg='green')


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
    args = parser.parse_args()

    model_specification = build_model_specification(args.simulation_configuration)
    model_specification.plugins.optional.update({
        "data": {
            "controller": "vivarium_inputs.data_artifact.ArtifactBuilder",
            "builder_interface": "vivarium_public_health.dataset_manager.ArtifactManagerInterface",
        }})

    plugin_config = model_specification.plugins
    component_config = model_specification.components
    simulation_config = model_specification.configuration

    simulation_config.input_data.location = get_location(args.location, simulation_config)
    simulation_config.input_data.artifact_path = get_output_path(args.simulation_configuration,
                                                                 args.output_root, args.location)
    simulation_config.input_data.append_to_artifact = not args.from_scratch

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
                    location_arg: str) -> os.PathLike:
    """Resolve the correct model output path

    Takes in to account the model specification and passed arguments.
    User-passed arguments supercede the configuration. Gauranteed to
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

    if output_root_arg:
        output_base = pathlib.Path(output_root_arg).resolve()
    else:
        output_base = (pathlib.Path('/share') / 'scratch' / 'users' /
                       getpass.getuser() / 'vivarium_artifacts')

    if location_arg:
        return output_base / (configuration_path.stem + f'_{location_arg}.hdf')
    else:
        return output_base / (configuration_path.stem + '.hdf')


if __name__ == "__main__":
    _build_artifact()
