import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import click

import gbd_mapping
from vivarium_public_health.dataset_manager import Artifact, ArtifactException, hdf, EntityKey

_log = logging.getLogger(__name__)


def aggregate():

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
        _log.debug(f'Individual artifacts failed for {set(locations).difference(set(valid_locations))} '
                      f'and only rest of locations will be aggregated', fg='red')

    artifact_path = f'{output}/{config_path.stem}.hdf'
    hdf.touch(artifact_path, False)
    hdf.write(artifact_path, EntityKey("metadata.locations"), list(valid_locations))
    hdf.write(artifact_path, EntityKey('metadata.keyspace'), [EntityKey('metadata.keyspace'), EntityKey('metadata.locations')])


    artifact = Artifact(artifact_path)

    for k in keyspace_set-{'metadata.keyspace', 'metadata.locations'}:
        data = [a.load(k) for a in valid_artifacts]
        if isinstance(data[0], pd.DataFrame):
            if 'location' in data[0].columns:
                data = pd.concat(data)
            else:
                assert np.all([d.equals(data[0]) for d in data])
                data = data[0]
            artifact.write(k, data)
        else:
            assert np.all([d == data[0] for d in data])  #for none dataframe, expect to have all same data
            artifact.write(k, data[0])

    # clean-up
    for file in Path(output).glob(f'{config_path.stem}_*.hdf'):
        file.unlink()


def disaggregate(config_name, output_root):
    metadata = {'keyspace': 'metadata.keyspace', 'location': 'metadata.locations', 'versions' : 'metadata.versions'}

    initial_artifact_path = pathlib.Path(output_root) / f'{config_name}.hdf'

    if not initial_artifact_path.is_file():
        raise ArtifactException('To append it, you should provide the existing artifacts')

    existing_artifact = Artifact(initial_artifact_path.as_posix())
    current_versions = {k.__title__: k.__version__ for k in [vivarium, vivarium_inputs, gbd_mapping]}
    if existing_artifact.load(metadata['versions']) != current_versions:
        click.secho('Your artifact was built under the different versions. We will build it from scratch.', fg='red')
        initial_artifact_path.unlink()
        existing_locations={}
    else:
        existing_keys = existing_artifact.load('metadata.keyspace')
        existing_locations = existing_artifact.load('metadata.locations')


        click.secho('Starting the disaggregation', fg='blue')
        for loc in existing_locations:
            temp_path = f'{initial_artifact_path.parent.as_posix()}/{config_name}_{loc.replace(" ", "_")}.hdf'

            hdf.touch(temp_path, False)
            hdf.write(temp_path, EntityKey("metadata.versions"),current_versions)
            hdf.write(temp_path, EntityKey("metadata.locations"), [loc])
            hdf.write(temp_path, EntityKey('metadata.keyspace'), [EntityKey(k) for k in metadata.values()])
            new_artifact = Artifact(temp_path)
            for e_key in set(existing_keys)-set(metadata.values()):
                data = existing_artifact.load(e_key)
                if isinstance(data, pd.DataFrame) and 'location' in data.columns:
                    data = data[data.location == loc]
                new_artifact.write(e_key, data)
        click.echo('Finished the disaggregation', fg='blue')
    return existing_locations




if __name__ == "__main__":
    aggregate()
