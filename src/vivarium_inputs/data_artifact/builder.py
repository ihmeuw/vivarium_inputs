from datetime import datetime
import logging
from typing import Collection, Any
import pandas as pd
import pkg_resources

from vivarium_public_health.dataset_manager import (EntityKey, Artifact, get_location_term, ArtifactException,
                                                    filter_data, create_hdf_with_keyspace)

from vivarium_public_health.disease import DiseaseModel

from vivarium_inputs.data_artifact.loaders import loader
from vivarium_inputs.forecasting import load_forecast


_log = logging.getLogger(__name__)


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
        if not append:
            create_hdf_with_keyspace(path)

        self.artifact = Artifact(path, filter_terms=[f'draw == {draw}', get_location_term(self.location)])
        self.write_metadata(append)
        self.modeled_causes = {c.cause for c in builder.components.get_components(DiseaseModel)}
        self.processed_entities = set()
        self.start_time = datetime.now()

        self.load("dimensions.full_space")

        builder.event.register_listener('post_setup', self.end_processing)

    def write_metadata(self, append):
        if append:
            try:
                self.artifact.remove('metadata.locations')
                self.artifact.remove('metadata.versions')
            except ArtifactException:
                # FIXME: We do not have a good plan to deal with appending an old artifact
                _log.debug('You provided an outdated artifact. We will build from scratch')
                create_hdf_with_keyspace(self.artifact.path)

        current_versions = {pkg_resources.get_distribution(k).version for k in
                            ['vivarium', 'vivarium_inputs', 'vivarium_public_health', 'gbd_mapping']}
        self.artifact.write('metadata.versions', current_versions)
        self.artifact.write('metadata.locations', [self.location])

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
