from pathlib import Path

import pytest
from tables import NoSuchNodeError
from vivarium.framework.artifact import ArtifactException

from vivarium_inputs.data_artifact import ArtifactBuilder, OutdatedArtifactError


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
    assert 'metadata.keyspace' not in test_artifact

    with pytest.raises(NoSuchNodeError):
        artifact = ArtifactBuilder.initialize_artifact(test_artifact.path, True, draw, location)


def test_initialize_artifact_append_locations(test_artifact):
    draw = 0
    location = 'United States'

    # artifact without a location
    test_artifact.remove('metadata.locations')
    assert 'metadata.locations' not in test_artifact

    with pytest.raises(ArtifactException):
        artifact = ArtifactBuilder.initialize_artifact(test_artifact.path, True, draw, location)


def test_initialize_artifact_append_versions(test_artifact):
    draw = 0
    location = 'United States'

    # artifact without versions
    test_artifact.remove('metadata.versions')
    assert 'metadata.versions' not in test_artifact

    with pytest.raises(ArtifactException):
        artifact = ArtifactBuilder.initialize_artifact(test_artifact.path, True, draw, location)


def test_initialize_artifact_no_append_file(test_artifact):
    draw = 0
    location = 'United States'

    test_artifact.write('test.key', 'data')
    assert 'test.key' in test_artifact

    # check whether the existing file wiped out
    artifact = ArtifactBuilder.initialize_artifact(test_artifact.path, False, draw, location)
    assert 'test.key' not in artifact
