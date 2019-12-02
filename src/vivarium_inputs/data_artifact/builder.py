from datetime import datetime
import logging
from typing import Collection, Any
import pandas as pd
from pathlib import Path

from vivarium.framework.artifact import Artifact, get_location_term, filter_data, EntityKey

from vivarium_public_health.disease import DiseaseModel

from vivarium_inputs.data_artifact.loaders import loader
from vivarium_inputs.data_artifact.utilities import get_versions, comparable_versions


_log = logging.getLogger(__name__)


class OutdatedArtifactError(Exception):
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
        If append, the existing artifact is validated and on passing, the
        existing artifact is returned. If not append, a new artifact is created
        and returned. In either case, the artifact returned has filter terms
        specified for the given draw and location.
        """
        if append:
            validate_artifact_for_appending(path, location)
            artifact = Artifact(path, filter_terms=[f'draw == {draw}', get_location_term(location)])
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
    if data is not None:
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


def validate_artifact_for_appending(path, location):
    if not Path(path).is_file():
        raise ValueError(f'{path} is not a file. You must provide an existing path to an artifact to append.')
    art = Artifact(path)
    if art.load('metadata.locations') != [location]:
        raise ValueError(f"Artifact was built for {art.load('metadata.locations')}. "
                         f"We cannot append {location}.")
    if not comparable_versions(art.load('metadata.versions'), get_versions()):
        raise OutdatedArtifactError('Existing artifact was built under different '
                                    'major/minor versions and cannot be appended to. '
                                    'You must build this artifact from scratch without append specified.')
