import logging
import argparse

from vivarium.framework.configuration import build_model_specification
from vivarium.framework.plugins import PluginManager
from vivarium.interface.interactive import InteractiveContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('simulation_configuration', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--from_scratch', '-s', action="store_true", help="Do not reuse any data in the artifact, if any exists")
    args = parser.parse_args()


    logging.basicConfig(level=logging.INFO)
    model_specification = build_model_specification(args.simulation_configuration)
    model_specification.plugins.optional.update({
        "data": {
            "controller": "ceam_inputs.data_artifact.ArtifactBuilder",
            "builder_interface": "ceam_public_health.dataset_manager.ArtifactManagerInterface",
        }})

    plugin_config = model_specification.plugins
    component_config = model_specification.components
    simulation_config = model_specification.configuration

    plugin_manager = PluginManager(plugin_config)
    component_config_parser = plugin_manager.get_plugin('component_configuration_parser')
    components = component_config_parser.get_components(component_config)

    simulation = InteractiveContext(simulation_config, components, plugin_manager)
    simulation.data.start_processing(simulation.component_manager, args.output_path,
                                     [simulation.configuration.input_data.location],
                                     incremental=not args.from_scratch)
    simulation.setup()
    simulation.data.end_processing()


if __name__ == "__main__":
    main()
