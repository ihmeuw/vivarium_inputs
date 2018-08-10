import getpass
import pathlib
import argparse
import subprocess
import time

import click

from vivarium.framework.configuration import build_model_specification
from vivarium.framework.plugins import PluginManager
from vivarium.interface.interactive import InteractiveContext


@click.command()
@click.argument('simulation_configuration', type=click.Path(dir_okay=False, readable=True))
@click.argument('project')
@click.argument('locations', nargs=-1)
@click.option('--output_root', type=click.Path(file_okay=False, writable=True),
        help="Directory to save artifact result in")
@click.option('--from_scratch', type=click.BOOL,  default=True,
        help="Do not reuse any data in the artifact, if any exists")
def build_artifact(simulation_configuration, project,locations, output_root, from_scratch):
    """
    build_artifact is a program for building data artifacts from a SIMULATION_CONFIGURATION file. The
    work is offloaded to the cluster under the provided PROJECT. Multiple, optional LOCATIONS can be
    provided to overwrite the configuration file.
 
    Any artifact.path specified in the configuration file is guaranteed to be overwritten either by the
    optional output_root or a predetermined path based on user: /ihme/scratch/{user}/vivarium_artifacts
    """ 
    config_path = pathlib.Path(simulation_configuration).resolve()
    python_context_path = pathlib.Path(subprocess.getoutput("which python")).resolve()
    script_path = pathlib.Path(__file__).resolve()

    script_arg = f"{script_path} {simulation_configuration} "
    if output_root:
        script_arg += f"--output_root {output_root} "
    if from_scratch:
        script_arg += f"--from_scratch {from_scratch} "

    if len(locations) > 0:
        for location in locations:
            job_name = f"{config_path.stem}_{location}_build_artifact"
            slots = 2
            submit_command = (f"qsub -N {job_name} -P {project} " +
                              f"-pe multi_slot {slots} " +
                              f"-b y {python_context_path} " +
                              script_arg + f"--location {location}")
            exitcode, response = subprocess.getstatusoutput(submit_command)
            if exitcode:
                click.secho(f"{location} qsub failed with exit code {exitcode}: {response}", fg='red')
            else:
                click.secho(f"{location} qsub succeeded: {response}", fg='green')
    else:
        job_name = f"{config_name}_build_artifact"
        slots = 2
        submit_command = (f"qsub -N {job_name} -pe multi_slot {slots} " +
                          script_arg)
        exitcode, response = subprocess.getstatusoutput(submit_command)
        if exitcode:
            click.secho(f"qsub failed with exit code {exitcode}: {response}", fg='red')
        else:
            click.secho(f"qsub succeeded: {response}", fg='green')


def parse_qsub(response):
    """Parse stdout from a call to qsub and return the job id"""

    split_response = response.split()
    # qsub response can be funky but JID should directly follow "Your job".
    # job arrays say "job-array", regular jobs say "job"
    job_ind = 0
    if 'job' in split_response:
        job_ind = split_response.index('job')
    elif 'job-array' in split_response:
        job_ind = split_response.index('job-array')
    if not job_ind:
        raise OSError("Unexpected response from qsub: " + response)

    jid = split_response[job_ind + 1]

    # If this is an array job, we want the parent jid not the array indicators.
    period_ind = jid.find(".")
    jid = jid[:period_ind]

    return jid


def _build_artifact():
    """Inward-facing interface for building a data artifact from a simulation
    configuration file"""

    parser = argparse.ArgumentParser()
    parser.add_argument('simulation_configuration', type=str,
            help="path to a simulation configuration file")
    parser.add_argument('--output_root', type=str, required=False,
            help="directory to save artifact to. Overwrites configuration file")
    parser.add_argument('--location', type=str, required=False,
            help="location to get data for. Overwrites configuration file")
    parser.add_argument('--from_scratch', '-s', action="store_true",
            help="Do not reuse any data in the artifact, if any exists")
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
    simulation_config.artifact.path = get_output_path(args.simulation_configuration, args.output_root,
                                                      args.location)

    output_path = simulation_config.artifact.path

    plugin_manager = PluginManager(plugin_config)
    component_config_parser = plugin_manager.get_plugin('component_configuration_parser')
    components = component_config_parser.get_components(component_config)

    simulation = InteractiveContext(simulation_config, components, plugin_manager)
    simulation.data.start_processing(simulation.component_manager, output_path,
                                     [simulation.configuration.input_data.location],
                                     incremental=not args.from_scratch)
    simulation.setup()
    simulation.data.end_processing()


def get_location(location_arg, configuration):
    """Return the correct location given the location argument and a configuration file"""

    if location_arg:
        return location_arg
    elif contains_location(configuration):
        return configuration.input_data.location
    else:
        raise argparse.ArgumentError(
            "specify a location or include configuration.input_data.location in model specification")    


def contains_location(configuration):
    """Check if location is specified in the configuration"""

    return ('input_data' in configuration and 'location' in configuration.input_data and
            configuration.input_data.location)


def get_output_path(configuration_arg, output_root_arg, location_arg):
    """Return the correct output path for the data artifact given the output_root and location arguments
    and a configuration file"""

    configuration_path = pathlib.Path(configuration_arg)

    if output_root_arg:
        output_base = pathlib.Path(output_root_arg).resolve()
    else:
        output_base = pathlib.Path('/share') / 'scratch' / 'users' / getpass.getuser() / 'vivarium_artifacts'

    if location_arg:
        return output_base / (configuration_path.stem + f'_{location_arg}.hdf')
    else:
        return output_base /= (configuration_path.stem + '.hdf')


if __name__ == "__main__":
    _build_artifact()
