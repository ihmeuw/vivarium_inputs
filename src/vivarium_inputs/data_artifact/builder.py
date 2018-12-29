from datetime import datetime
import logging
import warnings
from tables.exceptions import NoSuchNodeError
from typing import Collection, Any
import pandas as pd
from pathlib import Path

from vivarium_public_health.dataset_manager import (EntityKey, Artifact, get_location_term, filter_data)

from vivarium_public_health.disease import DiseaseModel

from vivarium_inputs.data_artifact.loaders import loader
from vivarium_inputs.forecasting import load_forecast
from vivarium_inputs.data_artifact.utilities import get_versions


_log = logging.getLogger(__name__)


class OutdatedArtifactWarning(Warning):
    pass


class ArtifactBuilder:
    configuration_defaults = {
        'input_data': {
            'forecast': False
        }
    }

    def setup(self, builder):
        path = builder.configuration.input_data.artifact_path
        append = builder.configuration.input_data.append_to_artifact
        draw = builder.configuration.input_data.input_draw_number
        self.location = builder.configuration.input_data.location
        self.artifact = self.initialize_artifact(path, append, draw, self.location)

        self.modeled_causes = {c.cause for c in builder.components.get_components(DiseaseModel)}
        self.processed_entities = set()
        self.start_time = datetime.now()

        self.load("dimensions.full_space")

        builder.event.register_listener('post_setup', self.end_processing)

    @staticmethod
    def initialize_artifact(path: str, append: bool, draw: int, location: str) -> Artifact:
        """
        For the given arguments, it checks all the basic conditions and
        initialize the artifact. For now, all the outdated artifacts which
        do not have proper metadata or not consistent metadata will not be
        appended. we will wipe it out and build a new artifact.
        """

        if append:
            if not Path(path).is_file():
                raise ValueError(f'{path} is not a file. You should provide the existing artifact path to append')

            try:
                artifact = Artifact(path, filter_terms=[f'draw == {draw}', get_location_term(location)])

            except NoSuchNodeError:
                #  it means that path was a file but does not have metadata.keyspace inside
                warnings.warn('We will wipe it out and build from scratch', OutdatedArtifactWarning)
                artifact = create_new_artifact(path, draw, location)

            if EntityKey('metadata.locations') not in artifact:
                warnings.warn('We will build from scratch', OutdatedArtifactWarning)
                artifact = create_new_artifact(path, draw, location)

            elif artifact.load('metadata.locations') != [location]:
                raise ValueError(f"Artifact has {artifact.load('metadata.locations')} and we cannot append {location}")

            if EntityKey('metadata.versions') not in artifact:
                warnings.warn('We will build from scratch', OutdatedArtifactWarning)
                artifact = create_new_artifact(path, draw, location)

            elif artifact.load('metadata.versions') != get_versions():
                warnings.warn('Your artifact was made under different versions. We will wipe it out',
                              OutdatedArtifactWarning)
                artifact = create_new_artifact(path, draw, location)
        else:
            artifact = create_new_artifact(path, draw, location)

        return artifact

    def load(self, entity_key: str, future=False, **__) -> Any:
        entity_key = EntityKey(entity_key)
        if entity_key not in self.artifact:
            self.process(entity_key, future)
        data = self.artifact.load(entity_key, future)
        return filter_data(data, **__) if isinstance(data, pd.DataFrame) else data

    def end_processing(self, event) -> None:
        _log.debug(f"Data loading took at most {datetime.now() - self.start_time} seconds")

    def process(self, entity_key: EntityKey, future=False) -> None:
        if entity_key not in self.processed_entities:
            _worker(entity_key, self.location, self.modeled_causes, self.artifact, future)
            self.processed_entities.add(entity_key)


def _worker(entity_key: EntityKey, location: str, modeled_causes: Collection[str],
            artifact: Artifact, future: bool) -> None:
    if future:
        data = load_forecast(entity_key, location)
    else:
        data = loader(entity_key, location, modeled_causes, all_measures=False)
    artifact.write(entity_key, data)


def create_new_artifact(path: str, draw: int, location: str) -> Artifact:
    if Path(path).is_file():
        Path(path).unlink()
    art = Artifact(path, filter_terms=[f'draw == {draw}', get_location_term(location)])
    art.write('metadata.versions', get_versions())
    art.write('metadata.locations', [location])
    return art