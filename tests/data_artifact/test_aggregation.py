import pytest

from pathlib import Path
import pandas as pd
import argparse

from vivarium_inputs.data_artifact import (aggregate, disaggregate, unpack_arguments, main, ArtifactAggregationWarning,
                                           get_versions)

from vivarium_public_health.dataset_manager import Artifact


METADATA = ['metadata.keyspace','metadata.locations', 'metadata.versions']


@pytest.fixture()
def single_artifact(tmpdir):
    def test_artifact(location):
        artifact_path = f'{str(tmpdir)}/test_{location}.hdf'
        artifact = Artifact(artifact_path)
        artifact.write('metadata.locations', [location.replace('_', ' ')])
        artifact.write('metadata.versions', get_versions())
        artifact.write('cause.all_causes.cause_specific_mortality',
                       pd.DataFrame({'location': location.replace('_', ' '), 'value': 0.1}, index=range(10)))
        artifact.write('risk_factor.child_stunting.exposure',
                       pd.DataFrame({'location': location.replace('_', ' '), 'value': 0.2}, index=range(10)))
        artifact.write('risk_factor.child_stunting.distribution', 'polytomous')
        return artifact
    return test_artifact


def test_unpack_arguments():
    config_path = 'path/to/test.yaml'
    output_root = 'path/to/artifact'
    locations = ['United States', 'Canada']
    args = f'--config_path {config_path} --output_root {output_root} --locations United_States Canada '.split()
    parser = argparse.ArgumentParser()
    assert unpack_arguments(parser, args) == (Path(config_path), locations, output_root)


def test_aggregate_one_location(single_artifact):
    test_artifact = single_artifact('United_States')
    artifact_path = Path(test_artifact.path)
    config_path = 'path/to/test.yaml'
    output_root = artifact_path.parent.as_posix()
    locations = 'United_States'

    main(['--config_path', config_path, '--output_root', output_root, '--locations', locations])

    renamed_artifact = artifact_path.parent / 'test.hdf'
    assert not artifact_path.is_file()
    assert renamed_artifact.is_file()


def test_aggregate_with_missing_keys(single_artifact):

    usa_artifact = single_artifact('United_States')
    canada_artifact = single_artifact('Canada')
    kenya_artifact = single_artifact('Kenya')

    # adding a new key to only two locations
    new_key = 'new.key'
    usa_artifact.write(new_key, 'data')
    canada_artifact.write(new_key, 'data')
    artifact_path = Path(usa_artifact.path).parent.as_posix()

    with pytest.warns(ArtifactAggregationWarning) as warn:
        new_artifact = aggregate([usa_artifact, canada_artifact, kenya_artifact], f'{artifact_path}/test.hdf')

        assert len(warn) == 1
        assert new_key in str(warn[-1].message)
        assert 'Kenya' in str(warn[-1].message)
        assert Path(new_artifact.path).is_file()
        assert set(new_artifact.keys) == set(usa_artifact.keys) == set(canada_artifact.keys)
        assert set(new_artifact.load('metadata.locations')) == {'Canada', 'United States'}


def test_aggregate(single_artifact):
    usa_artifact = single_artifact('United_States')
    canada_artifact = single_artifact('Canada')
    artifact_path = Path(usa_artifact.path).parent.as_posix()
    new_artifact = aggregate([usa_artifact, canada_artifact], f'{artifact_path}/test.hdf')

    assert set(new_artifact.keys) == set(usa_artifact.keys) == set(canada_artifact.keys)
    assert set(new_artifact.load('metadata.locations')) == {'Canada', 'United States'}


def test_disaggregate(single_artifact):
    usa_artifact = single_artifact('United_States')
    canada_artifact = single_artifact('Canada')
    kenya_artifact = single_artifact('Kenya')
    individual_artifacts = [usa_artifact, canada_artifact, kenya_artifact]
    expected_keys = usa_artifact.keys

    artifact_path = Path(usa_artifact.path).parent / 'test.hdf'
    aggregated_artifact = aggregate([usa_artifact, canada_artifact, kenya_artifact], artifact_path.as_posix())
    config_name = 'test'
    output_root = Path(aggregated_artifact.path).parent

    #  delete the individual artifacts
    for art in individual_artifacts:
        Path(art.path).unlink()

    disaggregated_artifacts_locations = disaggregate(config_name, output_root)

    # after disaggregation, initial aggregated artifact should be deleted
    assert not Path(aggregated_artifact.path).is_file()

    for loc in disaggregated_artifacts_locations:
        new_individual_artifact_path = output_root/f'{config_name}_{loc}.hdf'
        assert new_individual_artifact_path.is_file()
        new_artifact = Artifact(new_individual_artifact_path.as_posix())
        assert set(new_artifact.keys) == set(expected_keys)
        assert new_artifact.load('metadata.locations') == [loc.replace('_', ' ')]


def test_disaggregate_with_different_versions(single_artifact):
    usa_artifact = single_artifact('United_States')
    canada_artifact = single_artifact('Canada')
    kenya_artifact = single_artifact('Kenya')
    current_versions = get_versions()
    different_versions = {k: '0.0.1' for k in current_versions}

    artifact_path = Path(usa_artifact.path).parent / 'test.hdf'
    aggregated_artifact = aggregate([usa_artifact, canada_artifact, kenya_artifact], artifact_path.as_posix())

    # change the versions in aggregated_artifact
    aggregated_artifact.replace('metadata.versions', different_versions)
    assert aggregated_artifact.load('metadata.versions') == different_versions
    config_name = 'test'
    output_root = Path(aggregated_artifact.path).parent




    individual_artifacts = [usa_artifact, canada_artifact, kenya_artifact]
    expected_keys = usa_artifact.keys





