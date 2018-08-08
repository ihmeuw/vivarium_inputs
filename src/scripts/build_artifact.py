import logging
import getpass
import os
import argparse
import subprocess
import time

import click

from vivarium.framework.configuration import build_model_specification
from vivarium.framework.plugins import PluginManager
from vivarium.interface.interactive import InteractiveContext


_log = logging.getLogger(__name__)


@click.command()
@click.argument('simulation_configuration', nargs=1)
@click.argument('project', nargs=1)
@click.argument('locations', nargs=-1)
@click.option('--output_root')
@click.option('--from_scratch')
def build_artifact(simulation_configuration, project,locations, output_root, from_scratch):
    """Outward-facing interface for building a data artifact from a simulation
    configuration file

    simulation_configuration - path to a simulation config file
    locations - optional location specification. Can be multiple
    output_root - optional directory specification to save results in

    location and output_root specifications overwrite parameteres from the
    configuration file. Any artifact.path in the configuration file is 
    guaranteed to be overwritten either by the passed output_root or a
    predetermined path at /ihme/scratch/{user}/vivarium_artifacts/
    """ 
    config_file = os.path.basename(simulation_configuration)
    config_name = os.path.splitext(config_file)[0]

    python_context = os.path.realpath(subprocess.getoutput("which python"))

    #TODO: get full path. Will __file__ work with click?
    script_path = os.path.realpath(__file__)
    script_arg = f"{script_path} {simulation_configuration} "
    if output_root:
        script_arg += f"--output_root {output_root} "
    if from_scratch:
        script_arg += f"--from_scratch {from_scratch} "

    jids = []
    if len(locations) > 0:
        for location in locations:
            job_name = f"{config_name}_{location}_build_artifact"
            slots = 2
            submit_command = (f"qsub -N {job_name} -P {project} " +
                              f"-pe multi_slot {slots} " +
                              f"-b y {python_context} " +
                              script_arg + f"--location {location}")
            exitcode, response = subprocess.getstatusoutput(submit_command)
            if exitcode:
                raise OSError(exitcode, response)
            _log.info(response)
            jid = parse_qsub(response)
            jids.append(jid)
    else:
        job_name = f"{config_name}_build_artifact"
        slots = 2
        submit_command = (f"qsub -N {job_name} -pe multi_slot {slots} " +
                          script_arg)
        exitcode, response = subprocess.getstatusoutput(submit_command)
        if exitcode:
            raise OSError(exitcode, response)
        _log.info(response)
        jid = parse_qsub(response)
        jids.append(jid)

    monitor_job_ids(jids)


def monitor_job_ids(jids):
    """Monitor a list of job ids and print the percentage finished at
    15 second intervals"""

    total = len(jids)
    running = total
    while running:
        running = 0
        result = subprocess.getoutput('qstat')
        for jid in jids:
            if jid in result:
                running += 1
        percent_done = int(float(total - running) / total * 100)
        print(' ' * 15, end='\r')
        time.sleep(1)
        print(f' {percent_done}% finished.', end='\r')
        time.sleep(2)
    print("\n")


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
    parser.add_argument('simulation_configuration', type=str)
    parser.add_argument('--output_root', type=str, required=False)
    parser.add_argument('--location', type=str, required=False)
    parser.add_argument('--from_scratch', '-s', action="store_true",
                        help="Do not reuse any data in the artifact, if any exists")
    args = parser.parse_args()

    model_specification = build_model_specification(args.simulation_configuration)
    model_specification.plugins.optional.update({
        "data": {
            "controller": "vivarium_inputs.data_artifact.ArtifactBuilder",
            "builder_interface": "vivarium_public_health.dataset_manager.ArtifactManagerInterface",
        }})

    update_configuration(args.simulation_configuration, args.location, args.output_root,
                         model_specification.configuration)

    plugin_config = model_specification.plugins
    component_config = model_specification.components
    simulation_config = model_specification.configuration

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


def update_configuration(specification_arg, location_arg, output_root_arg, configuration):
    """Update the simulation configuration artifact output path and location with 
    command line inputs."""

    specification_file = os.path.basename(specification_arg)
    specification_name = os.path.splitext(specification_file)[0]

    if not location_arg and not output_root_arg:
        if ('input_data' in configuration and 'location' in configuration.input_data and 
            configuration.input_data.location):
            configuration.artifact.path = os.path.join('/', 'share', 'scratch', 'users',
                    getpass.getuser(), 'vivarium_artifacts', specification_name + '.hdf')
        else:
            raise argparse.ArgumentError(
                "specify a location or include configuration.input_data.location in model specification")
    elif not location_arg and output_root_arg:
        if ('input_data' in configuration and 'location' in configuration.input_data and 
            configuration.input_data.location):
            configuration.artifact.path = os.path.join(output_root_arg, 'specification_name' + '.hdf')
        else:
            raise argparse.ArgumentError(
                "specify a location or include configuration.input_data.location in model specification")
    elif location_arg and not output_root_arg:
        configuration.input_data.location = location_arg
        configuration.artifact.path = os.path.join('/', 'share', 'scratch', 'users',
                getpass.getuser(), 'vivarium_artifacts', specification_name + f'_{location_arg}.hdf')
    else:
        configuration.input_data.location = location_arg
        configuration.artifact.path = os.path.join(output_root_arg,
                                                   'specification_name' + f'_{location_arg}.hdf')


if __name__ == "__main__":
    _build_artifact()
