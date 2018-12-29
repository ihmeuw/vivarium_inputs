import pytest

from pathlib import Path
from vivarium_inputs.data_artifact import ArtifactBuilder, OutdatedArtifactWarning
from vivarium_public_health.dataset_manager import EntityKey


@pytest.fixture()
def test_artifact(tmpdir):
    path = Path(tmpdir)/'test.hdf'
    location = 'United States'
    draw = 0
    artifact = ArtifactBuilder.initialize_artifact(path.as_posix(), False, draw, location)
    return artifact


def test_initialize_artifact_append_not_a_file(tmpdir):
    path = Path(tmpdir)/'test.hdf'
    location = 'United States'
    draw = 0

    with pytest.raises(ValueError):
        ArtifactBuilder.initialize_artifact(path.as_posix(), True, draw, location)


def test_initialize_artifact_append_keyspace(test_artifact):
    draw = 0
    location = 'United States'

    # artifact without a keyspace
    test_artifact.remove('metadata.keyspace')
    assert EntityKey('metadata.keyspace') not in test_artifact

    with pytest.warns(OutdatedArtifactWarning):
        artifact = ArtifactBuilder.initialize_artifact(test_artifact.path, True, draw, location)
        assert EntityKey('metadata.keyspace') in artifact


def test_initialize_artifact_append_locations(test_artifact):
    draw = 0
    location = 'United States'

    # artifact without a location
    test_artifact.remove('metadata.locations')
    assert EntityKey('metadata.locations') not in test_artifact

    with pytest.warns(OutdatedArtifactWarning):
        artifact = ArtifactBuilder.initialize_artifact(test_artifact.path, True, draw, location)
        assert EntityKey('metadata.locations') in artifact
        assert artifact.load('metadata.locations') == [location]

    # artifact with a different location
    new_location = 'Canada'
    assert artifact.load('metadata.locations') == [location]

    with pytest.raises(ValueError):
        ArtifactBuilder.initialize_artifact(test_artifact.path, True, draw, new_location)


def test_initialize_artifact_append_versions(test_artifact):
    draw = 0
    location = 'United States'

    # artifact without versions
    test_artifact.remove('metadata.versions')
    assert EntityKey('metadata.versions') not in test_artifact

    with pytest.warns(OutdatedArtifactWarning):
        artifact = ArtifactBuilder.initialize_artifact(test_artifact.path, True, draw, location)
        assert EntityKey('metadata.versions') in artifact

    current_versions = artifact.load('metadata.versions')
    new_versions = {k: '0.1' for k in current_versions}
    artifact.replace('metadata.versions', new_versions)

    assert artifact.load('metadata.versions') == new_versions

    with pytest.warns(OutdatedArtifactWarning):
        artifact = ArtifactBuilder.initialize_artifact(test_artifact.path, True, draw, location)
        assert artifact.load('metadata.versions') == current_versions


def test_initialize_artifact_no_append_file(test_artifact):
    draw = 0
    location = 'United States'

    test_artifact.write('test.key', 'data')
    assert EntityKey('test.key') in test_artifact

    # check whether the existing file wiped out
    artifact = ArtifactBuilder.initialize_artifact(test_artifact.path, False, draw, location)
    assert EntityKey('test.key') not in artifact
