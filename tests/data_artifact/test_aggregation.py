import pytest

from pathlib import Path
import pandas as pd
import argparse

from vivarium_inputs.data_artifact import aggregation, ArtifactAggregationWarning, get_versions

from vivarium_public_health.dataset_manager import Artifact


METADATA = ['metadata.keyspace', 'metadata.locations', 'metadata.versions']


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
    args = f'--config_path {config_path} --output_root {output_root} --locations United_States Canada'
    parser = argparse.ArgumentParser()
    assert aggregation.unpack_arguments(parser, args.split()) == (Path(config_path), locations, output_root, False)
    parser = argparse.ArgumentParser()
    assert aggregation.unpack_arguments(parser, (args + ' --verbose').split()) == (Path(config_path), locations,
                                                                                   output_root, True)


def test_aggregate_one_location(single_artifact, mocker):
    aggregate_mock = mocker.patch('vivarium_inputs.data_artifact.aggregation.aggregate')
    test_artifact = single_artifact('United_States')
    artifact_path = Path(test_artifact.path)
    config_path = 'path/to/test.yaml'
    output_root = artifact_path.parent.as_posix()
    locations = 'United_States'

    aggregation.main(['--config_path', config_path, '--output_root', output_root, '--locations', locations])

    renamed_artifact = artifact_path.parent / 'test.hdf'
    assert not artifact_path.is_file()
    assert renamed_artifact.is_file()
    aggregate_mock.assert_not_called()


def test_aggregate_with_missing_keys(single_artifact):

    usa_artifact = single_artifact('United_States')
    canada_artifact = single_artifact('Canada')
    kenya_artifact = single_artifact('Kenya')

    # adding a new key to only two locations
    new_key = 'new.key'
    usa_artifact.write(new_key, 'data')
    canada_artifact.write(new_key, 'data')
    artifact_path = Path(usa_artifact.path).parent.as_posix()

    with pytest.warns(ArtifactAggregationWarning) as record:
        new_artifact = aggregation.aggregate([usa_artifact, canada_artifact, kenya_artifact], f'{artifact_path}/test.hdf')

        assert len(record) == 1
        assert new_key in str(record[-1].message)
        assert 'Kenya' in str(record[-1].message)
        assert Path(new_artifact.path).is_file()
        assert set(new_artifact.keys) == set(usa_artifact.keys) == set(canada_artifact.keys)
        assert set(new_artifact.load('metadata.locations')) == {'Canada', 'United States'}

        # clean up the individual artifact used for aggregation
        assert not Path(usa_artifact.path).is_file()
        assert not Path(canada_artifact.path).is_file()
        assert not Path(kenya_artifact.path).is_file()
        # Kenya artifact should be moved to broken artifacts folder
        new_location = Path(artifact_path)/'broken_artifacts'/f'{Path(kenya_artifact.path).name}'
        assert new_location.is_file()


def test_aggregate(single_artifact):
    usa_artifact = single_artifact('United_States')
    canada_artifact = single_artifact('Canada')
    artifact_path = Path(usa_artifact.path).parent.as_posix()
    new_artifact = aggregation.aggregate([usa_artifact, canada_artifact], f'{artifact_path}/test.hdf')

    assert set(new_artifact.keys) == set(usa_artifact.keys) == set(canada_artifact.keys)
    assert set(new_artifact.load('metadata.locations')) == {'Canada', 'United States'}

    assert Path(new_artifact.path).is_file()
    assert not Path(usa_artifact.path).is_file()
    assert not Path(canada_artifact.path).is_file()


def test_disaggregate(single_artifact):
    usa_artifact = single_artifact('United_States')
    canada_artifact = single_artifact('Canada')
    kenya_artifact = single_artifact('Kenya')
    individual_artifacts = [usa_artifact, canada_artifact, kenya_artifact]
    expected_keys = usa_artifact.keys

    artifact_path = Path(usa_artifact.path).parent / 'test.hdf'
    aggregated_artifact = aggregation.aggregate(individual_artifacts, artifact_path.as_posix())
    config_name = 'test'
    output_root = Path(aggregated_artifact.path).parent

    disaggregated_artifacts_locations = aggregation.disaggregate(config_name, output_root)

    # after disaggregation, initial aggregated artifact should be deleted
    assert not Path(aggregated_artifact.path).is_file()

    for loc in disaggregated_artifacts_locations:
        l = loc.replace(' ', '_').replace("'", "-")
        new_individual_artifact_path = output_root/f'{config_name}_{l}.hdf'
        assert new_individual_artifact_path.is_file()
        new_artifact = Artifact(new_individual_artifact_path.as_posix())
        assert set(new_artifact.keys) == set(expected_keys)
        assert new_artifact.load('metadata.locations') == [loc.replace('_', ' ')]


def test_disaggregate_with_different_versions(single_artifact):
    usa_artifact = single_artifact('United_States')
    canada_artifact = single_artifact('Canada')
    kenya_artifact = single_artifact('Kenya')
    individual_artifacts = [usa_artifact, canada_artifact, kenya_artifact]

    current_versions = get_versions()
    different_versions = {k: '0.0.1' for k in current_versions}

    artifact_path = Path(usa_artifact.path).parent / 'test.hdf'
    aggregated_artifact = aggregation.aggregate(individual_artifacts, artifact_path.as_posix())

    # change the versions in aggregated_artifact
    aggregated_artifact.replace('metadata.versions', different_versions)
    assert aggregated_artifact.load('metadata.versions') == different_versions
    config_name = 'test'
    output_root = Path(aggregated_artifact.path).parent

    with pytest.warns(ArtifactAggregationWarning, match = 'different versions'):
        disaggregated_artifacts_locations = aggregation.disaggregate(config_name, output_root)

    for loc in disaggregated_artifacts_locations:
        new_individual_artifact_path = output_root / f'{config_name}_{loc}.hdf'
        assert new_individual_artifact_path.is_file()
        new_artifact = Artifact(new_individual_artifact_path.as_posix())
        assert new_artifact.load('metadata.versions') == current_versions

