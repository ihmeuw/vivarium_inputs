from datetime import datetime
import logging
from typing import Callable, Mapping, NamedTuple, Optional, Collection

import pandas as pd
from vivarium.framework.components import ComponentManager
from vivarium_public_health.dataset_manager import EntityKey, Artifact
from vivarium_public_health.disease import DiseaseModel

from vivarium_inputs import core
from vivarium_inputs.data_artifact import LOADERS

_log = logging.getLogger(__name__)


class EntityConfig(NamedTuple):
    """A representation of an entity and the context in which to load it's data."""
    entity_key: EntityKey
    locations: Collection[str]
    year_start: int
    year_end: int
    modeled_causes: Collection[str]
    entity: Optional[str] = None


class ArtifactBuilder:
    """Builds a data artifact by first accumulating requests, then loading and writing the data in parallel.

    The data is output into an HDF file suitable for parsing by the ``vivarium`` simulation framework.
    """

    def __init__(self):
        import vivarium_gbd_access.gbd as gbd
        self.incremental = True
        estimation_years = gbd.get_estimation_years(gbd.GBD_ROUND_ID)
        self.year_start = min(estimation_years)
        self.year_end = max(estimation_years)
        self.processed_entities = set()
        self.modeled_causes = set()

        self.artifact = None

    def load(self, entity_k: str, keep_age_group_edges=False, **column_filters) -> None:
        """Records a request for entity data for future processing."""
        entity_key = EntityKey(entity_k)
        needs_load = self.incremental and entity_key not in self.artifact or not self.incremental

        if needs_load:
            self.process(entity_key)
        else:
            _log.info(f"Loading '{entity_key}' from artifact")

        return self.artifact.load(entity_key, keep_age_group_edges, **column_filters)

    def start_processing(self, component_manager: ComponentManager, path: str,
                         locations: Collection[str], loaders: Mapping[str, Callable]=None,
                         incremental: bool=True) -> None:
        self.incremental = incremental
        self.modeled_causes = {c.cause for c in component_manager._components if isinstance(c, DiseaseModel)}
        self.locations = locations
        if loaders is None:
            loaders = LOADERS
        self.loaders = loaders
        self.path = path
        self.artifact = Artifact(self.path)
        self.start_time = datetime.now()
        age_bins = core.get_age_bins()
        dimensions = [range(self.year_start, self.year_end+1), ["Male", "Female"], age_bins.age_group_id, locations]
        dimensions = pd.MultiIndex.from_product(dimensions, names=["year", "sex", "age_group_id", "location"])
        dimensions = dimensions.to_frame().reset_index(drop=True)
        self.artifact.write(EntityKey("dimensions.full_space"), dimensions)

    def end_processing(self) -> None:
        _log.debug(f"Data loading took at most {datetime.now() - self.start_time} seconds")

    def process(self, entity_key: EntityKey) -> None:
        """Loads all requested data and writes it out to a HDF file.

        Note
        ----
        The data loading process can be memory intensive. To reduce peak consumption, reduce parallelism.
        """

        if (entity_key.type, entity_key.name) not in self.processed_entities:
            entity_config = EntityConfig(entity_key=entity_key,
                                         year_start=self.year_start,
                                         year_end=self.year_end,
                                         locations=self.locations,
                                         modeled_causes=self.modeled_causes)
            _worker(entity_config, self.artifact, self.loaders[entity_key.type])
            self.processed_entities.add((entity_key.type, entity_key.name))


def _worker(entity_config: EntityConfig, artifact: Artifact, loader: Callable) -> None:
    """Loads and writes the data for a single entity into a shared output file.

    Parameters
    ----------
    entity_config :
        Container for contextual information used in the loading process.
    artifact :
        The data artifact to write to.
    loader :
        The function to load the entity's data. The loader must take an ``_EntityConfig`` object and
        the writer Callable defined within as arguments.
    """
    _log.info(f"Loading data for {entity_config.entity_key}")

    for measure, data in loader(entity_config):
        if isinstance(data, pd.DataFrame) and "year" in data:
            data = data.loc[(data.year >= entity_config.year_start) & (data.year <= entity_config.year_end)]
        key = entity_config.entity_key.with_measure(measure)
        artifact.write(key, data)
