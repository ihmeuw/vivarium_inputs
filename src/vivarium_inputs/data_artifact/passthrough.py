from typing import Any

from vivarium_public_health.dataset_manager import EntityKey, filter_data
from vivarium_public_health.disease import DiseaseModel

from vivarium_inputs.data_artifact.loaders import loader


class ArtifactPassthrough:

    def setup(self, builder):
        self.modeled_causes = builder.components.get_components(DiseaseModel)
        self.location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number

        self.base_filter = {'draw': draw,
                            'location': [self.location, 'Global']}

    def load(self, entity_key: str, keep_age_group_edges: bool=False, **column_filters: str) -> Any:
        entity_key = EntityKey(entity_key)
        data = loader(entity_key, self.location, self.modeled_causes)
        for key, val in self.base_filter.items():
            if key in data.columns:
                column_filters[key] = val
        return filter_data(data, keep_age_group_edges, **column_filters)
