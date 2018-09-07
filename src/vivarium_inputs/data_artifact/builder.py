from datetime import datetime
import logging
from typing import Collection, Any
import pandas as pd

from vivarium_public_health.dataset_manager import EntityKey, Artifact, hdf, get_location_term, filter_data
from vivarium_public_health.disease import DiseaseModel

from vivarium_inputs.data_artifact.loaders import loader


_log = logging.getLogger(__name__)


class ArtifactBuilder:

    def setup(self, builder):
        path = builder.configuration.input_data.artifact_path
        append = builder.configuration.input_data.append_to_artifact
        hdf.touch(path, append)
        draw = builder.configuration.input_data.input_draw_number

        self.location = builder.configuration.input_data.location
        self.artifact = Artifact(path, filter_terms=[f'draw == {draw}', get_location_term(self.location)])
        self.modeled_causes = builder.components.get_components(DiseaseModel)
        self.processed_entities = set()
        self.start_time = datetime.now()

        self.load("dimensions.full_space")

        builder.event.register_listener('post_setup', self.end_processing)

    def load(self, entity_key: str, keep_age_group_edges=False, **__) -> Any:
        entity_key = EntityKey(entity_key)
        if entity_key not in self.artifact:
            self.process(entity_key)
        data = self.artifact.load(entity_key)
        return filter_data(data, keep_age_group_edges, **__) if isinstance(data, pd.DataFrame) else data

    def end_processing(self, event) -> None:
        _log.debug(f"Data loading took at most {datetime.now() - self.start_time} seconds")

    def process(self, entity_key: EntityKey) -> None:
        if (entity_key.type, entity_key.name) not in self.processed_entities:
            _worker(entity_key, self.location, self.modeled_causes, self.artifact)
            self.processed_entities.add((entity_key.type, entity_key.name))


def _worker(entity_key: EntityKey, location: str, modeled_causes: Collection[str], artifact: Artifact) -> None:
    for measure, data in loader(entity_key, location, modeled_causes, all_measures=True):
        key = entity_key.with_measure(measure)
        artifact.write(key, data)
