from datetime import datetime
import logging
import warnings
from tables.exceptions import NoSuchNodeError
from typing import Collection, Any
import pandas as pd
from pathlib import Path

from vivarium.framework.artifact import Artifact, get_location_term, filter_data, EntityKey

from vivarium_public_health.disease import DiseaseModel

from vivarium_inputs.data_artifact.loaders import loader
from vivarium_inputs.data_artifact.utilities import get_versions, split_interval, comparable_versions


_log = logging.getLogger(__name__)


class OutdatedArtifactWarning(Warning):
    pass


class ArtifactBuilder:

    configuration_defaults = {
        'input_data': {
            'artifact_path': None,
            'artifact_filter_term': None,
        }
    }

    def setup(self, builder):
        path = builder.configuration.input_data.artifact_path
        append = builder.configuration.input_data.append_to_artifact
        draw = builder.configuration.input_data.input_draw_number
        self.location = builder.configuration.input_data.location
        self.artifact = self.initialize_artifact(path, append, draw, self.location)

        self.modeled_causes = {c.cause for c in builder.components.get_components_by_type(DiseaseModel)}
        self.processed_entities = set()
        self.start_time = datetime.now()

        # always include demographic dimensions so adding data free components doesn't require rebuilding artifacts
        self.load('population.demographic_dimensions')

        builder.event.register_listener('post_setup', self.end_processing)

    @property
    def name(self):
        return "artifact_builder"

    @staticmethod
    def initialize_artifact(path: str, append: bool, draw: int, location: str) -> Artifact:
        """
        For the given arguments, it checks all the basic conditions and
        initializes the artifact. For now, any artifact which is missing the
        required metadata or have inconsistent metadata will be rebuilt
        instead of appended.
        """

        if append:
            if not Path(path).is_file():
                raise ValueError(f'{path} is not a file. You should provide the existing artifact path to append')

            try:
                artifact = Artifact(path, filter_terms=[f'draw == {draw}', get_location_term(location)])

            except NoSuchNodeError:
                #  it means that path was a file but does not have metadata.keyspace inside
                warnings.warn('Keyspace information not found in existing artifact. Artifact will be rebuilt.',
                              OutdatedArtifactWarning)
                artifact = create_new_artifact(path, draw, location)

            if 'metadata.locations' not in artifact:
                warnings.warn('Location information not found in existing artifact. Artifact will be rebuilt.',
                              OutdatedArtifactWarning)
                artifact = create_new_artifact(path, draw, location)

            elif artifact.load('metadata.locations') != [location]:
                raise ValueError(f"Artifact was build for {artifact.load('metadata.locations')}. "
                                 f"We cannot append {location}.")

            if 'metadata.versions' not in artifact:
                warnings.warn('Version information not found in existing artifact. Artifact will be rebuilt.',
                              OutdatedArtifactWarning)
                artifact = create_new_artifact(path, draw, location)

            elif not comparable_versions(artifact.load('metadata.versions'), get_versions()):
                warnings.warn('Existing artifact was built under different major/minor versions. '
                              'Artifact will be rebuilt.', OutdatedArtifactWarning)
                artifact = create_new_artifact(path, draw, location)
        else:
            artifact = create_new_artifact(path, draw, location)

        return artifact

    def load(self, entity_key: str, **__) -> Any:
        if entity_key not in self.artifact:
            self.process(entity_key)
        data = self.artifact.load(entity_key)
        if isinstance(data, pd.DataFrame):  # could be metadata dict
            data = data.reset_index()
            draw_col = [c for c in data if 'draw' in c]
            if draw_col:
                data = data.rename(columns={draw_col[0]: 'value'})
        return filter_data(data, **__) if isinstance(data, pd.DataFrame) else data

    def end_processing(self, event) -> None:
        _log.debug(f"Data loading took at most {datetime.now() - self.start_time} seconds")

    def process(self, entity_key: str) -> None:
        if entity_key not in self.processed_entities:
            _worker(entity_key, self.location, self.modeled_causes, self.artifact)
            self.processed_entities.add(entity_key)


def _worker(entity_key: str, location: str, modeled_causes: Collection[str],
            artifact: Artifact) -> None:
    data = loader(EntityKey(entity_key), location, modeled_causes, all_measures=False)
    # FIXME: This is a hack since hdf files can't handle pandas.Interval objects
    if data is not None:
        data = split_interval(data, interval_column='age', split_column_prefix='age')
        data = split_interval(data, interval_column='year', split_column_prefix='year')
        artifact.write(entity_key, data)
    else:
        _log.warning(f"None received when loading data for {entity_key}.")


def create_new_artifact(path: str, draw: int, location: str) -> Artifact:
    if Path(path).is_file():
        Path(path).unlink()
    art = Artifact(path, filter_terms=[f'draw == {draw}', get_location_term(location)])
    art.write('metadata.versions', get_versions())
    art.write('metadata.locations', [location])
    return art
