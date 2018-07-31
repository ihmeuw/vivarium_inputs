import os
import warnings
from collections import defaultdict
from typing import Optional, NamedTuple, Sequence, Mapping, Iterable, Callable, Tuple

import pandas as pd
import tables
from tables.nodes import filenode
import json

from vivarium.framework.components import ComponentManager
from vivarium_inputs import core
from vivarium_inputs.utilities import normalize_for_simulation, get_age_group_midpoint_from_age_group_id
from vivarium_public_health.dataset_manager import Artifact
from vivarium_public_health.disease.model import DiseaseModel
from gbd_mapping import causes, risks, sequelae, coverage_gaps, etiologies, covariates

from vivarium_inputs.mapping_extension import healthcare_entities

import logging
_log = logging.getLogger(__name__)

CAUSE_BY_ID = {c.gbd_id: c for c in causes if c is not None}
RISK_BY_ID = {r.gbd_id: r for r in risks}


class DataArtifactError(Exception):
    """Base error raised for issues in data artifact construction."""
    pass


class EntityError(DataArtifactError):
    """Error raised when modeled entities are improperly specified."""
    pass


class _EntityConfig(NamedTuple):
    """A representation of an entity and the context in which to load it's data."""
    type: str
    name: str
    locations: Iterable[str]
    year_start: int
    year_end: int
    modeled_causes: Iterable[str]
    entity: Optional[str] = None


def _normalize(data: pd.DataFrame) -> pd.DataFrame:
    """Remove GBD specific column names and concepts and make the dataframe long over draws."""
    assert not data.empty
    data = normalize_for_simulation(data)
    if "age_group_id" in data:
        data = get_age_group_midpoint_from_age_group_id(data)
    draw_columns = [c for c in data.columns if "draw_" in c]
    index_columns = [c for c in data.columns if "draw_" not in c]
    data = pd.melt(data, id_vars=index_columns, value_vars=draw_columns, var_name="draw")
    data["draw"] = data.draw.str.partition("_")[2].astype(int)
    return data


def _parse_entity_path(entity_path: str) -> Tuple[str, Optional[str], str]:
    # FIXME We need a better specification of what the paths can be. This is confusing.
    entity_type, *tail = entity_path.split('.')
    if len(tail) == 2:
        entity_name = tail[0]
        measure = tail[1]
    elif len(tail) == 1:
        entity_name = None
        measure = tail[0]
    else:
        raise ValueError(f"Invalid entity path: {entity_path}")

    return entity_type, entity_name, measure


def _entities_by_type(entities: Iterable[str]) -> Mapping[str, set]:
    """Split a list of entity paths and group the entity names by entity type."""
    entity_by_type = defaultdict(set)
    for entity_path in entities:
        entity_type, entity_name, _ = _parse_entity_path(entity_path)
        entity_by_type[entity_type].add(entity_name)
    return entity_by_type


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

    def load(self, entity_path: str, keep_age_group_edges=False, **column_filters) -> None:
        """Records a request for entity data for future processing."""

        needs_load = True
        if self.incremental:
            self.artifact.open()
            try:
                if ("/"+entity_path.replace(".", "/")) in self.artifact._hdf:
                    needs_load = False
            finally:
                self.artifact.close()

        if needs_load:
            self.process(entity_path)
        else:
            _log.info(f"Loading '{entity_path}' from artifact")

        self.artifact.open()
        try:
            result = self.artifact.load(entity_path, keep_age_group_edges, **column_filters)
        finally:
            self.artifact.close()

        return result

    def start_processing(self, component_manager: ComponentManager, path: str,
                         locations: Sequence[str], loaders: Mapping[str, Callable]=None,
                         incremental: bool=True) -> None:
        self.incremental = incremental
        self.modeled_causes = {c.cause for c in component_manager._components if isinstance(c, DiseaseModel)}
        self.locations = locations
        if loaders is None:
            loaders = LOADERS
        self.loaders = loaders
        self.path = path
        self.artifact = Artifact(self.path, self.year_start, self.year_end, 0, self.locations[0])

        age_bins = core.get_age_bins()
        dimensions = [range(self.year_start, self.year_end+1), ["Male", "Female"], age_bins.age_group_id, locations]
        dimensions = pd.MultiIndex.from_product(dimensions, names=["year", "sex", "age_group_id", "location"])
        dimensions = dimensions.to_frame().reset_index(drop=True)
        _dump(dimensions, "dimensions", None, "full_space", path)

    def end_processing(self) -> None:
        pass

    def process(self, entity_path: str) -> None:
        """Loads all requested data and writes it out to a HDF file.

        Parameters
        ----------
        entity_path :
            The absolute path to the output HDF file to write.

        Note
        ----
        The data loading process can be memory intensive. To reduce peak consumption, reduce parallelism.
        """

        entity_type, entity_name, _ = _parse_entity_path(entity_path)

        if (entity_type, entity_name) not in self.processed_entities:
            entity_config = _EntityConfig(type=entity_type,
                                          name=entity_name,
                                          year_start=self.year_start,
                                          year_end=self.year_end,
                                          locations=self.locations,
                                          modeled_causes=self.modeled_causes)
            _worker(entity_config, self.path, self.loaders[entity_type])
            self.processed_entities.add((entity_type, entity_name))


def _worker(entity_config: _EntityConfig, path: str, loader: Callable) -> None:
    """Loads and writes the data for a single entity into a shared output file.

    Parameters
    ----------
    entity_config :
        Container for contextual information used in the loading process.
    path :
        The path to the output file to write to.
    loader :
        The function to load the entity's data. The loader must take an ``_EntityConfig`` object and
        the writer Callable defined within as arguments.
    """
    _log.info(f"Loading data for {entity_config.type}.{entity_config.name}")

    for measure, data in loader(entity_config):
        if isinstance(data, pd.DataFrame) and "year" in data:
            data = data.loc[(data.year >= entity_config.year_start) & (data.year <= entity_config.year_end)]

        _dump(data, entity_config.type, entity_config.name, measure, path)


def _prepare_key(entity_type: str, entity_name: Optional[str], measure: str) -> Sequence[str]:
    key_components = ["/", entity_type]
    if entity_name:
        key_components.append(entity_name)

    return key_components + [measure]

def _dump(data, entity_type: str, entity_name: Optional[str], measure: str, path: str) -> None:
    if data is None:
        return

    key_components = _prepare_key(entity_type, entity_name, measure)

    """Write a dataset out to the target HDF file keyed by the entity the data corresponds to."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        _dump_dataframe(data, key_components, path)
    else:
        _dump_json_blob(data, key_components, path)

def _dump_dataframe(data, key_components: Sequence[str], path: str) -> None:
    if data.empty:
        raise ValueError("Cannot persist empty dataset")
    data_columns = {"year", "location", "draw", "cause", "risk"}.intersection(data.columns)
    inner_path = os.path.join(*key_components)
    with pd.HDFStore(path, complevel=9, format="table") as store:
        store.put(inner_path, data, format="table", data_columns=data_columns)

def _dump_json_blob(data, key_components: Sequence[str], path:str) -> None:
    inner_path = os.path.join(*key_components)
    prefix = os.path.join(*key_components[:-2])
    store = tables.open_file(path, "a")
    if inner_path in store:
        store.remove_node(inner_path)
    try:
        store.create_group(prefix, key_components[-2], createparents=True)
    except tables.exceptions.NodeError as e:
        if "already has a child node" in str(e):
            # The parent group already exists, which is fine
            pass
        else:
            raise

    fnode = filenode.new_node(store, where=os.path.join(*key_components[:-1]), name=key_components[-1])
    fnode.write(bytes(json.dumps(data), "utf-8"))
    fnode.close()
    store.close()


def _load_cause(entity_config: _EntityConfig) -> None:
    measures = ["death", "prevalence", "incidence", "cause_specific_mortality", "excess_mortality"]
    cause = causes[entity_config.name]

    yield "restrictions", {
            "male_only": cause.restrictions.male_only,
            "female_only": cause.restrictions.female_only,
            "yll_only": cause.restrictions.yll_only,
            "yld_only": cause.restrictions.yld_only,
            "yll_age_start": cause.restrictions.yll_age_start,
            "yll_age_end": cause.restrictions.yll_age_end,
            "yld_age_start": cause.restrictions.yld_age_start,
            "yld_age_end": cause.restrictions.yld_age_end,
    }

    if cause.sequelae is not None:
        yield "sequelae", [s.name for s in cause.sequelae]
    if cause.etiologies is not None:
        yield "etiologies", [e.name for e in cause.etiologies]

    result = core.get_draws([cause], measures, entity_config.locations)
    result = _normalize(result)
    for key, group in result.groupby("measure"):
        yield key, group[["year", "location", "sex", "age", "draw", "value"]]
    del result

    if cause.name != "all_causes":
        pafs = core.get_draws([cause], ["population_attributable_fraction"], entity_config.locations)
        if pafs.empty:
            warnings.warn(f"No Population Attributable Fraction data found for cause '{cause.name}'")
        else:
            normalized = []
            for key, group in pafs.groupby(["risk_id"]):
                group = group.drop(["risk_id"], axis=1)
                group = _normalize(group)
                if key in RISK_BY_ID:
                    group["risk"] = RISK_BY_ID[key].name
                    dims = ["year", "sex", "measure", "age", "age_group_start",
                            "age_group_end", "location", "draw", "risk"]
                    normalized.append(group.set_index(dims))
                else:
                    warnings.warn(f"Found risk_id {key} in population attributable fraction data for cause "
                                  f"'{cause.name}' but that risk is missing from the gbd mapping")
            result = pd.concat(normalized).reset_index()
            result = result[["year", "location", "sex", "age", "draw", "value", "risk"]]
            yield "population_attributable_fraction", result
            del normalized
            del result

    try:
        measures = ["remission"]
        result = core.get_draws([causes[entity_config.name]], measures, entity_config.locations)
        if not result.empty:
            result = _normalize(result)[["year", "location", "sex", "age", "draw", "value"]]
            yield "remission", result
        else:
            yield "remission", None
    except core.InvalidQueryError:
        yield "remission", None


def _load_risk_factor(entity_config: _EntityConfig) -> None:
    risk = risks[entity_config.name]

    yield "affected_causes", [c.name for c in risk.affected_causes if c.name in entity_config.modeled_causes]
    yield "restrictions", {
            "male_only": risk.restrictions.male_only,
            "female_only": risk.restrictions.female_only,
            "yll_only": risk.restrictions.yll_only,
            "yld_only": risk.restrictions.yld_only,
            "yll_age_start": risk.restrictions.yll_age_start,
            "yll_age_end": risk.restrictions.yll_age_end,
            "yld_age_start": risk.restrictions.yld_age_start,
            "yld_age_end": risk.restrictions.yld_age_end,
    }
    yield "distribution", risk.distribution
    if risk.exposure_parameters is not None:
        yield "exposure_parameters", {
            "scale": risk.exposure_parameters.scale,
            "max_rr": risk.exposure_parameters.max_rr,
            "max_val": risk.exposure_parameters.max_val,
            "min_val": risk.exposure_parameters.min_val,
        }

    if risk.distribution == "ensemble":
        weights = core.get_ensemble_weights([risk])
        weights = weights.drop(["location_id", "risk_id"], axis=1)
        weights = normalize_for_simulation(weights)
        weights = get_age_group_midpoint_from_age_group_id(weights)
        yield "ensemble_weights", weights


    if risk.levels is not None:
        yield "levels" , [(f"cat{cat_number}", level_name) for level_name, cat_number in zip(risk.levels, range(60))]

    if risk.tmred is not None:
        yield "tmred", {
                "distribution": risk.tmred.distribution,
                "min": risk.tmred.min,
                "max": risk.tmred.max,
                "inverted": risk.tmred.inverted,
        }
    if risk.exposure_parameters is not None:
        yield "exposure_parameters", {
                "scale": risk.exposure_parameters.scale,
                "max_rr": risk.exposure_parameters.max_rr,
                "max_val": risk.exposure_parameters.max_val,
                "min_val": risk.exposure_parameters.min_val,
        }

    rrs = core.get_draws([risk], ["relative_risk"], entity_config.locations)
    normalized = []
    for key, group in rrs.groupby(["parameter", "cause_id"]):
        group = group.drop(["cause_id", "parameter"], axis=1)
        group = _normalize(group)
        group["parameter"] = key[0]
        group["cause"] = CAUSE_BY_ID[key[1]].name
        dims = ["year", "sex", "measure", "age", "age_group_start",
                "age_group_end", "location", "draw", "cause", "parameter"]
        normalized.append(group.set_index(dims))
    result = pd.concat(normalized).reset_index()
    result = result[["year", "location", "sex", "age", "draw", "value", "parameter", "cause"]]
    yield "relative_risk", result
    del normalized
    del result

    exposures = core.get_draws([risk], ["exposure"], entity_config.locations)
    normalized = []
    for key, group in exposures.groupby(["parameter"]):
        group = group.drop(["parameter"], axis=1)
        group = _normalize(group)
        group["parameter"] = key
        dims = ["year", "sex", "measure", "age", "age_group_start", "age_group_end", "location", "draw", "parameter"]
        normalized.append(group.set_index(dims))
    result = pd.concat(normalized).reset_index()
    result = result[["year", "location", "sex", "age", "draw", "value", "parameter"]]
    yield "exposure", result
    del normalized
    del result

    if risk.exposure_parameters is not None:
        exposure_sds = core.get_draws([risk], ["exposure_standard_deviation"], entity_config.locations)
        exposure_sds = _normalize(exposure_sds)
        exposure_sds = exposure_sds[["year", "location", "sex", "age", "draw", "value"]]
        yield "exposure_standard_deviation", exposure_sds
    else:
        yield "exposure_standard_deviation", None


def _load_sequela(entity_config: _EntityConfig) -> None:
    sequela = sequelae[entity_config.name]

    yield "healthstate", sequela.healthstate.name

    measures = ["prevalence", "incidence"]
    result = core.get_draws([sequela], measures, entity_config.locations).drop("sequela_id", axis=1)
    result = _normalize(result)
    result["sequela_id"] = sequela.gbd_id
    for key, group in result.groupby("measure"):
        yield key, group[["year", "location", "sex", "age", "draw", "value"]]
    del result

    weights = core.get_draws([sequela], ["disability_weight"], entity_config.locations)
    index_columns = [c for c in weights.columns if "draw_" not in c]
    draw_columns = [c for c in weights.columns if "draw_" in c]
    weights = pd.melt(weights, id_vars=index_columns, value_vars=draw_columns, var_name="draw")
    weights["draw"] = weights.draw.str.partition("_")[2].astype(int)
    yield "disability_weight", weights[['draw', 'value']]


def _load_healthcare_entity(entity_config: _EntityConfig) -> None:
    healthcare_entity = healthcare_entities[entity_config.name]

    cost = core.get_draws([healthcare_entity], ["cost"], entity_config.locations)
    cost = _normalize(cost)
    cost = cost[["year", "location", "draw", "value"]]
    yield "cost", cost

    annual_visits = core.get_draws([healthcare_entity], ["annual_visits"], entity_config.locations)
    annual_visits = _normalize(annual_visits)
    annual_visits = annual_visits[["year", "sex", "age", "value", "draw"]]
    yield "annual_visits", annual_visits


def _load_coverage_gap(entity_config: _EntityConfig) -> None:
    entity = coverage_gaps[entity_config.name]

    yield "affected_causes", [c.name for c in entity.affected_causes if c.name in entity_config.modeled_causes]
    yield "restrictions", {
            "male_only": entity.restrictions.male_only,
            "female_only": entity.restrictions.female_only,
            "yll_only": entity.restrictions.yll_only,
            "yld_only": entity.restrictions.yld_only,
            "yll_age_start": entity.restrictions.yll_age_start,
            "yll_age_end": entity.restrictions.yll_age_end,
            "yld_age_start": entity.restrictions.yld_age_start,
            "yld_age_end": entity.restrictions.yld_age_end,
    }
    yield "distribution", entity.distribution

    yield "levels" , [(f"cat{cat_number}", level_name) for level_name, cat_number in zip(entity.levels, range(60))]

    try:
        exposure = core.get_draws([entity], ["exposure"], entity_config.locations)
        exposure = _normalize(exposure)
        yield "exposure", exposure
    except core.InvalidQueryError:
        yield "exposure", None

    relative_risk = core.get_draws([entity], ["relative_risk"], entity_config.locations)
    relative_risk = _normalize(relative_risk)
    relative_risk["cause"] = relative_risk.cause_id.apply(lambda cause_id: CAUSE_BY_ID[cause_id].name)
    relative_risk = relative_risk.drop('cause_id', axis=1)
    yield "relative_risk", relative_risk

    paf = core.get_draws([entity], ["population_attributable_fraction"], entity_config.locations)
    if not paf.empty:
        paf = _normalize(paf)
        paf["cause"] = paf.cause_id.apply(lambda cause_id: CAUSE_BY_ID[cause_id].name)
        paf = paf.drop('cause_id', axis=1)
        yield "population_attributable_fraction", paf


def _load_etiology(entity_config: _EntityConfig) -> None:
    entity = etiologies[entity_config.name]

    paf = core.get_draws([entity], ["population_attributable_fraction"], entity_config.locations)
    paf = _normalize(paf)
    paf["cause"] = paf.cause_id.apply(lambda cause_id: CAUSE_BY_ID[cause_id].name)
    paf = paf[["year", "location", "cause", "sex", "age", "draw", "value"]]
    yield "population_attributable_fraction", paf


def _load_population(entity_config: _EntityConfig) -> None:
    pop = core.get_populations(entity_config.locations)
    pop = normalize_for_simulation(pop)
    pop = get_age_group_midpoint_from_age_group_id(pop)
    yield "structure", pop

    bins = core.get_age_bins()[["age_group_years_start", "age_group_years_end", "age_group_name"]]
    bins = bins.rename(columns={"age_group_years_start": "age_group_start", "age_group_years_end": "age_group_end"})
    yield "age_bins", bins

    yield "theoretical_minimum_risk_life_expectancy", core.get_theoretical_minimum_risk_life_expectancy()


def _load_covariate(entity_config: _EntityConfig) -> None:
    entity = covariates[entity_config.name]
    location_ids = [core.get_location_ids_by_name()[l] for l in entity_config.locations]
    estimate = gbd.get_covariate_estimates([entity.gbd_id], location_ids)
    estimate['location'] = estimate.location_id.apply(core.get_location_names_by_id().get)
    estimate = estimate.drop('location_id', 'columns')

    if entity is covariates.age_specific_fertility_rate:
        columns = ["location", "mean_value", "lower_value", "upper_value", "age_group_id", "sex_id", "year_id"]
        estimate = estimate[columns]
        estimate = get_age_group_midpoint_from_age_group_id(estimate)
        estimate = normalize_for_simulation(estimate)
    elif entity in (covariates.live_births_by_sex, covariates.dtp3_coverage_proportion):
        columns = ["location", "mean_value", "lower_value", "upper_value", "sex_id", "year_id"]
        estimate = estimate[columns]
        estimate = normalize_for_simulation(estimate)
    yield "estimate", estimate


def _load_subregions(entity_config: _EntityConfig) -> None:
    df = pd.DataFrame(core.get_subregions(entity_config.locations))
    df = df.melt(var_name="location", value_name="subregion_id")
    yield "sub_region_ids", df


LOADERS = {
    "cause": _load_cause,
    "risk_factor": _load_risk_factor,
    "sequela": _load_sequela,
    "population": _load_population,
    "healthcare_entity": _load_healthcare_entity,
    "coverage_gap": _load_coverage_gap,
    "etiology": _load_etiology,
    "covariate": _load_covariate,
    "subregions": _load_subregions,
}
