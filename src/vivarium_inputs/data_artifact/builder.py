from datetime import datetime
import logging
from typing import NamedTuple, Collection, Any

from vivarium_public_health.dataset_manager import EntityKey, Artifact, hdf

from vivarium_inputs.data_artifact.loaders import loader


_log = logging.getLogger(__name__)


class LoadContext(NamedTuple):
    location: str
    modeled_causes: Collection[str]


class ArtifactBuilder:

    def start_processing(self, path: str, append: bool, location: str, modeled_causes: Collection[str]) -> None:
        hdf.touch(path, append)

        self.artifact = Artifact(path)
        self.load_context = LoadContext(location, modeled_causes)
        self.processed_entities = set()
        self.start_time = datetime.now()

    def load(self, entity_key: str, *_, **__) -> Any:
        entity_key = EntityKey(entity_key)
        if entity_key not in self.artifact:
            self.process(entity_key)
        return self.artifact.load(entity_key)

    def end_processing(self) -> None:
        _log.debug(f"Data loading took at most {datetime.now() - self.start_time} seconds")

    def process(self, entity_key: EntityKey) -> None:
        if (entity_key.type, entity_key.name) not in self.processed_entities:
            _worker(entity_key, self.load_context, self.artifact)
            self.processed_entities.add((entity_key.type, entity_key.name))


def _worker(entity_key: EntityKey, load_context: LoadContext, artifact: Artifact) -> None:
    for measure, data in loader(entity_key, load_context):
        key = entity_key.with_measure(measure)
        artifact.write(key, data)
