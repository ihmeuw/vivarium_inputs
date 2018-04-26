import argparse

from vivarium.framework.engine import build_simulation_configuration, load_component_manager, setup_simulation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('simulation_configuration', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('-p', '--parallelism', type=int, default=None)
    args = parser.parse_args()

    config = build_simulation_configuration(vars(args))
    config.vivarium.dataset_manager = "ceam_inputs.data_artifact.ArtifactBuilder"
    component_manager = load_component_manager(config)
    simulation = setup_simulation(component_manager, config)
    simulation.data.process(args.output_path, [simulation.configuration.input_data.location_id], parallelism=args.parallelism)

if __name__ == "__main__":
    main()
