import logging
import warnings
import argparse
import pandas as pd
import numpy as np
import pkg_resources
from pathlib import Path
from typing import Set

from vivarium_public_health.dataset_manager import Artifact, ArtifactException, hdf, EntityKey

_log = logging.getLogger(__name__)


def aggregate():
    """ Aggregate multiple artifacts to a single artifact.
        We only take the union of the each keyspace of single artifacts and
        do not aggregate any single artifact does not have all the keys
        in the union of keyspaces.

        Aggregation is held until single artifact building jobs are completed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--locations', nargs='+')
    args = parser.parse_args()

    config_path = Path(args.config_path)
    locations = args.locations
    locations = [k.replace('_', ' ') for k in locations]
    output = args.output_root
    individual_artifacts = Path(output).glob('*.hdf')
    artifacts = [Artifact(a.as_posix()) for a in individual_artifacts]

    keyspace_set = set().union(*[a.load('metadata.keyspace') for a in artifacts])

    valid_artifacts, valid_locations = [], []
    for a in artifacts:
        if set(a.load('metadata.keyspace')) == keyspace_set:
            valid_artifacts.append(a)
            valid_locations.extend(a.load('metadata.locations'))

    if set(valid_locations) < set(locations):
        warnings.warn(f'Individual artifacts failed for {set(locations).difference(set(valid_locations))} '
                      f'and only rest of locations will be aggregated')

    metadata = {'keyspace': 'metadata.keyspace', 'location': 'metadata.locations', 'versions': 'metadata.versions'}
    artifact_path = f'{output}/{config_path.stem}.hdf'
    hdf.touch(artifact_path, False)
    hdf.write(artifact_path, EntityKey("metadata.locations"), list(valid_locations))
    current_versions = {k: pkg_resources.get_distribution(k).version for k in
                        ['vivarium', 'vivarium_public_health', 'gbd_mapping', 'vivarium_inputs']}
    hdf.write(artifact_path, EntityKey("metadata.versions"), current_versions)
    hdf.write(artifact_path, EntityKey('metadata.keyspace'), [EntityKey(k) for k in metadata.values()])
    artifact = Artifact(artifact_path)

    for k in keyspace_set-set(metadata.values()):
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
    for file in Path(output).glob(f'{config_path.stem}_*.hdf'):
        file.unlink()


def disaggregate(config_name: str, output_root: str) -> Set:
    """Disaggreagte the existing multi-location artifacts into the single
    location artifacts for appending. It is only called when the append flag
    is true. For now we only warn when the current versions of our libraries
    are different from the versions when the existing artifacts were built.

    :param config_name: string of the configuration file stem i.e., existing
                        artifact name
    :param output_root: where artifacts will be stored
    :return: set of existing locations
    """
    metadata = {'keyspace': 'metadata.keyspace', 'location': 'metadata.locations', 'versions' : 'metadata.versions'}

    initial_artifact_path = Path(output_root) / f'{config_name}.hdf'

    if not initial_artifact_path.is_file():
        raise ArtifactException('To append it, you should provide the existing artifacts')

    existing_artifact = Artifact(initial_artifact_path.as_posix())
    current_versions = {k: pkg_resources.get_distribution(k).version for k in
                        ['vivarium', 'vivarium_public_health', 'gbd_mapping', 'vivarium_inputs']}

    #  FIXME: For now we only warn and build from scratch. We need a smarter way to handle ths.
    if existing_artifact.load(metadata['versions']) != current_versions:
        warnings.warn('Your artifact was built under the different versions. We will build it from scratch.')
        initial_artifact_path.unlink()
        existing_locations = {}
    else:
        existing_keys = existing_artifact.load('metadata.keyspace')
        existing_locations = existing_artifact.load('metadata.locations')

        for loc in existing_locations:
            temp_path = f'{initial_artifact_path.parent.as_posix()}/{config_name}_{loc.replace(" ", "_")}.hdf'

            hdf.touch(temp_path, False)
            hdf.write(temp_path, EntityKey("metadata.versions"), current_versions)
            hdf.write(temp_path, EntityKey("metadata.locations"), [loc])
            hdf.write(temp_path, EntityKey('metadata.keyspace'), [EntityKey(k) for k in metadata.values()])
            
            new_artifact = Artifact(temp_path)
            for e_key in set(existing_keys)-set(metadata.values()):
                data = existing_artifact.load(e_key)
                if isinstance(data, pd.DataFrame) and 'location' in data.columns:
                    data = data[data.location == loc]
                new_artifact.write(e_key, data)

        initial_artifact_path.unlink()
    return existing_locations


if __name__ == "__main__":
    aggregate()
