from typing import Any

import pandas as pd
from vivarium_public_health.dataset_manager import EntityKey, filter_data, validate_filter_term
from vivarium_public_health.disease import DiseaseModel

from vivarium_inputs.data_artifact.loaders import loader
from vivarium_inputs.data_artifact.utilities import split_interval


class ArtifactPassthrough:

    def setup(self, builder):
        self.modeled_causes = builder.components.get_components_by_type(DiseaseModel)
        self.location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number

        self.base_filter = {'draw': draw,
                            'location': [self.location, 'Global']}
        self.config_filter_term = validate_filter_term(builder.configuration.input_data.artifact_filter_term)

    @property
    def name(self):
        return "artifact_passthrough"

    def load(self, entity_key: str, **column_filters: str) -> Any:
        entity_key = EntityKey(entity_key)
        data = loader(entity_key, self.location, self.modeled_causes)

        if isinstance(data, pd.DataFrame):  # could be a metadata dict
            data = split_interval(data, interval_column='age', split_column_prefix='age_group')
            data = split_interval(data, interval_column='year', split_column_prefix='year')
            data = data.reset_index()

            for key, val in self.base_filter.items():
                if key in data.columns:
                    column_filters[key] = val

            data = filter_data(data, self.config_filter_term, **column_filters)
        return data
