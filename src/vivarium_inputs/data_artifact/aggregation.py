import logging
import warnings
import argparse
import click
import pandas as pd
import numpy as np
import pkg_resources
from pathlib import Path
from typing import Set

from vivarium_public_health.dataset_manager import Artifact, ArtifactException, hdf, EntityKey, get_location_term

_log = logging.getLogger(__name__)


def aggregate():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--locations', nargs='+')
    args = parser.parse_args()

    config_path = Path(args.config_path)
    locations = args.locations
    locations = [l.replace('_', ' ') for l in locations]
    output = args.output_root
    individual_artifacts = Path(output).glob('*.hdf')
    artifacts = [Artifact(a.as_posix()) for a in individual_artifacts]
    artifact_path = f'{output}/{config_path.stem}.hdf'
    metadata = ['metadata.keyspace','metadata.locations', 'metadata.versions']

    if len(locations)==1:
        Path(output)/f'{config_path.stem}_{locations[0]}.hdf'.rename(Path(artifact_path))

    else:
        keyspace_set = set().union(*[a.load('metadata.keyspace') for a in artifacts])

        valid_artifacts, valid_locations = [], []
        for a in artifacts:
            if set(a.load('metadata.keyspace')) == keyspace_set:
                valid_artifacts.append(a)
                valid_locations.extend(a.load('metadata.locations'))
            warnings.warn(f'missing_keys: {keyspace_set.difference(set(a.load("metadata.keyspace")))} for location:{a.load("metadata.locations")}')
        
        if set(valid_locations) < set(locations):
            warnings.warn(f'Individual artifacts failed for {set(locations).difference(set(valid_locations))} '
                          f'and only rest of locations will be aggregated')

        
        hdf.touch(artifact_path, False)
        hdf.write(artifact_path, EntityKey("metadata.locations"), valid_locations)
        current_versions = {k: pkg_resources.get_distribution(k).version for k in ['vivarium', 'vivarium_public_health', 'gbd_mapping', 'vivarium_inputs']}
        hdf.write(artifact_path, EntityKey("metadata.versions"), current_versions)
        hdf.write(artifact_path, EntityKey('metadata.keyspace'), metadata)
        artifact = Artifact(artifact_path)

        for k in keyspace_set-set(metadata):
            data = [a.load(k) for a in valid_artifacts]
            if isinstance(data[0], pd.DataFrame):
                if 'location' in data[0].columns:
                    data = pd.concat(data)
                else:
                    assert np.all([d.equals(data[0]) for d in data])
                    data = data[0]
                artifact.write(k, data)
            else:
                assert np.all([d == data[0] for d in data])
                artifact.write(k, data[0])

        # clean-up
        for f in Path(output).glob(f'{config_path.stem}_*.hdf'):
            f.unlink()


def disaggregate(config_name: str, output_root: str) -> Set:
    metadata = ['metadata.keyspace','metadata.locations', 'metadata.versions']

    initial_artifact_path = Path(output_root) / f'{config_name}.hdf'

    if not initial_artifact_path.is_file():
        raise ArtifactException(f'To append it, you should provide the existing artifact. {output_root}/{config_name}.hdf does not exist')

    existing_artifact = Artifact(initial_artifact_path.as_posix())
    current_versions = {k: pkg_resources.get_distribution(k).version for k in
                        ['vivarium', 'vivarium_public_health', 'gbd_mapping', 'vivarium_inputs']}
    if existing_artifact.load('metadata.versions') != current_versions:
        warnings.warn('Your artifact was built under the different versions. We will build it from scratch.')
        initial_artifact_path.unlink()
        existing_locations = {}
    else:
        existing_keys = existing_artifact.load('metadata.keyspace')
        existing_locations = existing_artifact.load('metadata.locations')

        for loc in existing_locations:
            location = get_location_term(loc)
            temp_path = f'{initial_artifact_path.parent.as_posix()}/{config_name}_{loc.replace(" ", "_")}.hdf'

            hdf.touch(temp_path, False)
            hdf.write(temp_path, EntityKey("metadata.versions"),current_versions)
            hdf.write(temp_path, EntityKey("metadata.locations"), [loc])
            hdf.write(temp_path, EntityKey('metadata.keyspace'), metadata)

            existing_artifact = Artifact(initial_artifact_path.as_posix(), filter_terms=[location])
            new_artifact = Artifact(temp_path)
            for e_key in set(existing_keys)-set(metadata):
                data = existing_artifact.load(e_key)
                new_artifact.write(e_key, data)

        initial_artifact_path.unlink()
    return [l.replace(' ', '_') for l in existing_locations]


if __name__ == "__main__":
    aggregate()
