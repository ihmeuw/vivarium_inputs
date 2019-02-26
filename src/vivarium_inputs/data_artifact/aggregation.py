import logging
import warnings
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Set

from vivarium_public_health.dataset_manager import Artifact, ArtifactException, get_location_term
from vivarium_inputs.data_artifact.utilities import get_versions, setup_logging


_log = logging.getLogger(__name__)
METADATA = ['metadata.keyspace', 'metadata.locations', 'metadata.versions']


class ArtifactAggregationWarning(Warning):
    pass


def unpack_arguments(parser, rawargs=None):
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--locations', nargs='+')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(rawargs)

    config_path = Path(args.config_path)
    locations = args.locations
    locations = [l.replace('_', ' ') for l in locations]
    output_root = args.output_root
    verbose = args.verbose

    return config_path, locations, output_root, verbose


def aggregate(artifacts: [Artifact], artifact_path: str) -> Artifact:
    """
    aggregates the list of artifacts with individual location. It takes a union
    of all the keys in individual artifacts. If there is any artifact which
    does not have all the keys, it will drop that location with a warning and
    aggregate only the locations with the same keys.


    Parameters
    ----------
    artifacts:
        list of Artifact objects
    artifact_path:
        path to the final aggregated artifact to be stored
    Returns
    -------
        aggregated artifact
    """

    artifact_path = Path(artifact_path).resolve()
   
    if artifact_path.is_file():
        artifact_path.unlink()

    logging.debug(f'Validating individual artifacts for the common keyspace.')

    locations = set().union(*[a.load('metadata.locations') for a in artifacts])
    keyspace_set = set().union(*[a.load('metadata.keyspace') for a in artifacts])
    valid_artifacts, valid_locations = [], []
    for a in artifacts:
        if set(a.load('metadata.keyspace')) == keyspace_set:
            valid_artifacts.append(a)
            valid_locations.extend(a.load('metadata.locations'))
        else:
            warnings.warn(f'Missing_keys: {keyspace_set.difference(set(a.load("metadata.keyspace")))} '
                          f'for location:{a.load("metadata.locations")} All artifacts not aggregated'
                          f'will be stored in {artifact_path}/broken_artifacts.', ArtifactAggregationWarning)

    artifact = Artifact(artifact_path.as_posix())
    artifact.write("metadata.locations", valid_locations)
    current_versions = get_versions()

    artifact.write("metadata.versions", current_versions)
    keys_to_process = keyspace_set - set(METADATA)
    for i, k in enumerate(keys_to_process):
        logging.debug(f'Processing key {i+1} of {len(keys_to_process)}: {k}.')
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

    logging.debug('Finished aggregating keys. Commencing cleanup of location-specific artifacts.')

    # clean-up
    for loc in valid_locations:
        l = loc.replace(' ', '_').replace("'", "-")
        f = artifact_path.parent/f'{artifact_path.stem}_{l}.hdf'
        f.unlink()

    invalid_locations = set(locations).difference(set(valid_locations))
    if len(invalid_locations) > 0:
        new_dir = artifact_path.parent/'broken_artifacts'
        new_dir.mkdir()

        for loc in invalid_locations:
            l = loc.replace(' ', '_').replace("'", "-")
            f = artifact_path.parent/f'{artifact_path.stem}_{l}.hdf'
            f.rename(new_dir/f.name)

    logging.debug('Aggregation cleanup finished.')

    return artifact


def main(rawargs=None):
    """ Aggregate multiple artifacts to a single artifact.
        We only take the union of the each keyspace of single artifacts and
        do not aggregate any single artifact does not have all the keys
        in the union of keyspaces.
    """
    parser = argparse.ArgumentParser()
    config_path, locations, output, verbose = unpack_arguments(parser, rawargs)

    setup_logging(output, verbose, 'aggregation', config_path, False)

    individual_artifacts = []
    for loc in locations:
        l = loc.replace(' ', '_').replace("'", "-")
        individual_path = Path(output)/f'{config_path.stem}_{l}.hdf'
        individual_artifacts.append(Artifact(individual_path.as_posix()))

    artifact_path = f'{output}/{config_path.stem}.hdf'
    logging.debug(f'Beginning aggregation for: {locations}.')
    if len(locations) == 1:
        logging.debug(f'Processing location 1 of 1: {locations[0]}.')
        location = locations[0].replace(' ', '_').replace("'", "-")
        current_artifact = Path(output) / f'{config_path.stem}_{location}.hdf'
        current_artifact.rename(Path(artifact_path))

    else:
        aggregate(individual_artifacts, artifact_path)


def disaggregate(config_name: str, output_root: str) -> Set:
    """Disaggregate the existing multi-location artifacts into the single
    location artifacts for appending. It is only called when the append flag
    is true. If the existing artifact was built under the different versions of
    relevant libraries from the versions of those at the moment when appending
    is called, we will wipe it out and build it from scratch.

    Parameters
    ----------
    config_name:
        string of the configuration file stem i.e., existing artifact name
    output_root:
        where artifacts will be stored

    Returns
    -------
        set of existing locations
    """

    initial_artifact_path = Path(output_root) / f'{config_name}.hdf'

    if not initial_artifact_path.is_file():
        raise ArtifactException(
            f'To append it, you should provide the existing artifact. {output_root}/{config_name}.hdf does not exist')

    existing_artifact = Artifact(initial_artifact_path.as_posix())

    if existing_artifact.load('metadata.versions') != get_versions():
        #  FIXME: For now we only warn and build from scratch. We need a smarter way to handle ths.
        warnings.warn('Your artifact was built under different versions from those currently installed, '
                      'so we cannot append to it. It will be built from scratch.', ArtifactAggregationWarning)

        initial_artifact_path.unlink()
        existing_locations = {}
    else:
        existing_locations = _disaggregate(existing_artifact, initial_artifact_path)
    return existing_locations


def _disaggregate(existing_artifact: Artifact, initial_artifact_path: Path) -> Set:
    existing_keys = existing_artifact.load('metadata.keyspace')
    existing_locations = existing_artifact.load('metadata.locations')

    for loc in existing_locations:
        location = get_location_term(loc)
        l = loc.replace(' ', '_').replace("'", "-")
        temp_path = f'{initial_artifact_path.parent.as_posix()}/{initial_artifact_path.stem}_{l}.hdf'
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
