import argparse

from vivarium.framework.engine import build_simulation_configuration, load_component_manager, setup_simulation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('simulation_configuration', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    config = build_simulation_configuration(vars(args))
    config.vivarium.dataset_manager = "ceam_inputs.data_artifact.ArtifactBuilder"
    config.run_configuration.input_draw_number = 0
    component_manager = load_component_manager(config)
    component_manager.dataset_manager.start_processing(component_manager, args.output_path, [component_manager.config.input_data.location])
    simulation = setup_simulation(component_manager, config)
    component_manager.dataset_manager.end_processing()

if __name__ == "__main__":
    main()
