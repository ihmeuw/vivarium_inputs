import logging
import warnings
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from vivarium_public_health.dataset_manager import Artifact, ArtifactException, get_location_term
from vivarium_inputs.data_artifact.utilities import get_versions


_log = logging.getLogger(__name__)
METADATA = ['metadata.keyspace', 'metadata.locations', 'metadata.versions']


class ArtifactAggregationWarning(Warning):
    pass


def unpack_arguments(parser, rawargs=None):
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--locations', nargs='+')
    args = parser.parse_args(rawargs)

    config_path = Path(args.config_path)
    locations = args.locations
    locations = [l.replace('_', ' ') for l in locations]
    output_root = args.output_root

    return config_path, locations, output_root


def aggregate(artifacts: [Artifact], artifact_path: str) -> Artifact:
    """
    aggregates the list of artifacts with individual location. It takes a union
    of all the keys in individual artifacts. If there is any artifact which
    does not have all the keys, it will drop that location with a warning and
    aggregate only the locations with the same keys.

    :param artifacts: list of Artifact objects
    :param artifact_path: path to the final aggregated artifact to be stored
    :return: aggregated artifact
    """

    artifact_path = Path(artifact_path).resolve()
   
    if artifact_path.is_file():
        artifact_path.unlink()

    keyspace_set = set().union(*[a.load('metadata.keyspace') for a in artifacts])
    valid_artifacts, valid_locations = [], []
    for a in artifacts:
        if set(a.load('metadata.keyspace')) == keyspace_set:
            valid_artifacts.append(a)
            valid_locations.extend(a.load('metadata.locations'))
        else:
            warnings.warn(f'missing_keys: {keyspace_set.difference(set(a.load("metadata.keyspace")))} '
                          f'for location:{a.load("metadata.locations")}', ArtifactAggregationWarning)

    artifact = Artifact(artifact_path.as_posix())
    artifact.write("metadata.locations", valid_locations)
    current_versions = get_versions()

    artifact.write("metadata.versions", current_versions)

    for k in keyspace_set - set(METADATA):
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
    for loc in valid_locations:
        f = artifact_path.parent/f'{artifact_path.stem}_{loc.replace(" ", "_")}.hdf'
        f.unlink()
    return artifact


def main(rawargs=None):
    """ Aggregate multiple artifacts to a single artifact.
        We only take the union of the each keyspace of single artifacts and
        do not aggregate any single artifact does not have all the keys
        in the union of keyspaces.

        Aggregation is held until single artifact building jobs are completed.
    """
    parser = argparse.ArgumentParser()
    config_path, locations, output = unpack_arguments(parser, rawargs)

    individual_artifacts = []
    for loc in locations:
        individual_path = Path(output)/f'{config_path.stem}_{loc.replace(" ", "_")}.hdf'
        individual_artifacts.append(individual_path)
    
    artifacts = [Artifact(a.as_posix()) for a in individual_artifacts]
    artifact_path = f'{output}/{config_path.stem}.hdf'
    _log.debug(f'{locations}')
    if len(locations) == 1:
        location = locations[0].replace(' ', '_')
        current_artifact = Path(output) / f'{config_path.stem}_{location}.hdf'
        current_artifact.rename(Path(artifact_path))

    else:
        aggregate(artifacts, artifact_path)


def disaggregate(config_name: str, output_root: str) -> List:
    """Disaggreagte the existing multi-location artifacts into the single
    location artifacts for appending. It is only called when the append flag
    is true. If the existing artifact was built under the different versions of
    relevant libraries from the versions of those at the moment when appending
    is called, we will wipe it out and build it from scratch.

    :param config_name: string of the configuration file stem i.e., existing
                        artifact name
    :param output_root: where artifacts will be stored
    :return: set of existing locations
    """

    initial_artifact_path = Path(output_root) / f'{config_name}.hdf'

    if not initial_artifact_path.is_file():
        raise ArtifactException(
            f'To append it, you should provide the existing artifact. {output_root}/{config_name}.hdf does not exist')

    existing_artifact = Artifact(initial_artifact_path.as_posix())

    if existing_artifact.load('metadata.versions') != get_versions():
        #  FIXME: For now we only warn and build from scratch. We need a smarter way to handle ths.
        warnings.warn('Your artifact was built under the different versions and cannot append it',
                      ArtifactAggregationWarning)

        initial_artifact_path.unlink()
        existing_locations = {}
    else:
        existing_keys = existing_artifact.load('metadata.keyspace')
        existing_locations = existing_artifact.load('metadata.locations')

        for loc in existing_locations:
            location = get_location_term(loc)
            temp_path = f'{initial_artifact_path.parent.as_posix()}/{config_name}_{loc.replace(" ", "_")}.hdf'
            existing_artifact = Artifact(initial_artifact_path.as_posix(), filter_terms=[location])

            new_artifact = Artifact(temp_path)
            new_artifact.write('metadata.locations', [loc])
            new_artifact.write('metadata.versions', get_versions())

            for e_key in set(existing_keys) - set(METADATA):
                data = existing_artifact.load(e_key)
                new_artifact.write(e_key, data)

        initial_artifact_path.unlink()
    return [l.replace(' ', '_') for l in existing_locations]


if __name__ == "__main__":
    main()
