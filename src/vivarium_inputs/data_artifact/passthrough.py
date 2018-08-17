from typing import Any

from vivarium_public_health.dataset_manager import EntityKey

class ArtifactPassthrough:

    def load(self, entity_key: str, keep_age_group_edges: bool=False, **column_filters: str) -> Any:
        entity_key = EntityKey(entity_key)
