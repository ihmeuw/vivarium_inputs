import logging
import getpass
import os
import argparse

from vivarium.framework.configuration import build_model_specification
from vivarium.framework.plugins import PluginManager
from vivarium.interface.interactive import InteractiveContext


def _build_artifact():
    parser = argparse.ArgumentParser()
    parser.add_argument('simulation_configuration', type=str)
    parser.add_argument('--output_root', type=str, optional=True)
    parser.add_argument('--location', type=str, optional=True)
    parser.add_argument('--from_scratch', '-s', action="store_true",
                        help="Do not reuse any data in the artifact, if any exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

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
            configuration.artifact.path = os.path.join('\\', 'share', 'scratch', getpass.getuser(),
                'vivarium_artifacts', specification_name + '.hdf')
        else
            raise argparse.ArgumentError(
                "specify a location or include configuration.input_data.location in model specification")
    elif not location_arg and output_root_arg 
        if ('input_data' in configuration and 'location' in configuration.input_data and 
            configuration.input_data.location):
            configuration.artifact.path = os.path.join(output_root_arg, 'specification_name' + '.hdf')
        else:
            raise argparse.ArgumentError(
                "specify a location or include configuration.input_data.location in model specification")
    elif location_arg and not output_root_arg
        configuration.input_data.location = location_arg
        configuration.artifact.path = os.path.join('\\', 'share', 'scratch', getpass.getuser(),
            'vivarium_artifacts', specification_name + f'_{location_arg}.hdf')
    else:
        configuration.input_data.location = location_arg
        configuration.artifact.path = os.path.join(output_root_arg,
                                                   'specification_name' + f'_{location_arg}.hdf')


if __name__ == "__main__":
    _build_artifact()
