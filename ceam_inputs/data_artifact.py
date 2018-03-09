import os
import multiprocessing
from collections import defaultdict
from random import shuffle
from typing import Tuple, Optional, NamedTuple, Sequence, Mapping, Iterable, Callable

import pandas as pd

from ceam_inputs import core
from ceam_inputs.utilities import normalize_for_simulation, get_age_group_midpoint_from_age_group_id

from .gbd import get_estimation_years, get_covariate_estimates, GBD_ROUND_ID
from .gbd_mapping import (causes, risk_factors, sequelae, healthcare_entities,
                          treatment_technologies, coverage_gaps, etiologies, covariates)

import logging
_log = logging.getLogger(__name__)


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
    locations: Sequence[int]
    year_start: int
    year_end: int
    modeled_causes: Iterable
    entity: Optional[str] = None


def _normalize(data: pd.DataFrame) -> pd.DataFrame:
    """Remove GBD specific column names and concepts and make the dataframe long over draws."""
    data = normalize_for_simulation(data)
    if "age_group_id" in data:
        data = get_age_group_midpoint_from_age_group_id(data)
    draw_columns = [c for c in data.columns if "draw_" in c]
    index_columns = [c for c in data.columns if "draw_" not in c]
    data = pd.melt(data, id_vars=index_columns, value_vars=draw_columns, var_name="draw")
    data["draw"] = data.draw.str.partition("_")[2].astype(int)
    return data


def _entities_by_type(entities: Iterable[str]) -> Mapping[str, set]:
    """Split a list of entity paths and group the entity names by entity type."""
    entity_by_type = defaultdict(set)
    for entity_path in entities:
        entity_type, *tail = entity_path.split('.')
        entity_name = tail[0] if tail else None
        entity_by_type[entity_type].add(entity_name)
    return entity_by_type


class ArtifactBuilder:
    """Builds a data artifact by first accumulating requests, then loading and writing the data in parallel.

    The data is output into an HDF file suitable for parsing by the ``vivarium`` simulation framework.

    Attributes
    ----------
    entities :
        The entities (e.g. causes, risk factors, treatment technologies, etc.) whose data will
        be included in the artifact.
    """

    def __init__(self):
        self.entities = set()

    def data_container(self, entity_path: str) -> None:
        """Records a request for entity data for future processing."""
        self.entities.add(entity_path)
        _log.info(f"Adding {entity_path} to list of data sets to load")

    def process(self, path: str, locations: Sequence[int], parallelism: int=None,
                loaders: Mapping[str, Callable]=None) -> None:
        """Loads all requested data and writes it out to a HDF file.

        Parameters
        ----------
        path :
            The absolute path to the output HDF file to write.
        locations :
            A set of locations to load data for.
        parallelism :
            The number of processes to use when loading the data. Defaults to the number of available CPUs.
        loaders :
            A mapping between entity types and the functions that load their data.

        Note
        ----
        The data loading process can be memory intensive. To reduce peak consumption, reduce parallelism.
        """

        if loaders is None:
            loaders = LOADERS

        locations = locations

        estimation_years = get_estimation_years(GBD_ROUND_ID)
        year_start = min(estimation_years)
        year_end = max(estimation_years)

        entity_by_type = _entities_by_type(self.entities)

        age_bins = core.get_age_bins()
        dimensions = [range(year_start, year_end+1), ["Male", "Female"], age_bins.age_group_id, locations]
        dimensions = pd.MultiIndex.from_product(dimensions, names=["year", "sex", "age_group_id", "location_id"])
        dimensions = dimensions.to_frame().reset_index(drop=True)
        _dump(dimensions, "dimensions", None, "full_space", path)

        if parallelism is None:
            parallelism = multiprocessing.cpu_count()

        lock_manager = multiprocessing.Manager()
        lock = lock_manager.Lock()

        pool = multiprocessing.Pool(parallelism)
        by_type = list(entity_by_type.items())
        shuffle(by_type)
        jobs = []
        for entity_type, entities in by_type:
            for entity_name in entities:
                entity_config = _EntityConfig(type=entity_type,
                                              name=entity_name,
                                              year_start=year_start,
                                              year_end=year_end,
                                              locations=locations,
                                              modeled_causes=entity_by_type["cause"])
                if parallelism > 1:
                    jobs.append(pool.apply_async(_worker, (entity_config, path, loaders[entity_type], lock)))
                else:
                    _worker(entity_config, path, loaders[entity_type], lock)
        pool.close()
        # NOTE: This loop is necessary because without the calls to get exceptions raised within the
        # workers would not be reraised here and the program could exit before all the workers finish.
        for j in jobs:
            j.get()


def _worker(entity_config: _EntityConfig, path: str, loader: Callable, lock: multiprocessing.Lock) -> None:
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
    lock :
        A mechanism to prevent multiple processes from writing to the output file at the
        same time to prevent data corruption.
    """
    _log.info(f"Loading data for {entity_config.type}.{entity_config.name}")

    for measure, data in loader(entity_config):
        if isinstance(data, pd.DataFrame) and "year" in data:
            data = data.loc[(data.year >= entity_config.year_start) & (data.year <= entity_config.year_end)]

        lock.acquire()
        try:
            _dump(data, entity_config.type, entity_config.name, measure, path)
        finally:
            lock.release()



def _dump(data: pd.DataFrame, entity_type: str, entity_name: Optional[str], measure: str, path: str) -> None:
    """Write a dataset out to the target HDF file keyed by the entity the data corresponds to."""
    key_components = ["/", entity_type]
    if entity_name:
        key_components.append(entity_name)

    key = os.path.join(*(key_components + [measure]))
    with pd.HDFStore(path, complevel=9, format="table") as store:
        store.put(key, data, format="table")


def _load_cause(entity_config: _EntityConfig) -> None:
    measures = ["death", "prevalence", "incidence", "cause_specific_mortality", "excess_mortality"]
    result = core.get_draws([causes[entity_config.name]], measures, entity_config.locations)
    result = _normalize(result)
    for key, group in result.groupby("measure"):
        yield key, group
    del result

    try:
        measures = ["remission"]
        result = core.get_draws([causes[entity_config.name]], measures, entity_config.locations)
        result["cause_id"] = causes[entity_config.name].gbd_id
        yield "remission", result
    except core.InvalidQueryError:
        pass


def _load_risk_factor(entity_config: _EntityConfig) -> None:
    if entity_config.name == "correlations":
        # TODO: weird special case but this groups it with the other risk data which  I think makes sense
        correlations = core.get_risk_correlation_matrix(entity_config.locations)
        yield "correlations", correlations
        return

    risk = risk_factors[entity_config.name]

    rrs = core.get_draws([risk], ["relative_risk"], entity_config.locations)
    normalized = []
    for key, group in rrs.groupby(["parameter", "cause_id"]):
        group = group.drop(["cause_id", "parameter"], axis=1)
        group = _normalize(group)
        group["parameter"] = key[0]
        group["cause_id"] = key[1]
        dims = ["year", "sex", "measure", "age", "age_group_start",
                "age_group_end", "location_id", "draw", "cause_id", "parameter"]
        normalized.append(group.set_index(dims))
    yield "relative_risk", pd.concat(normalized)
    del normalized

    mfs = core.get_draws([risk], ["mediation_factor"], entity_config.locations)
    if not mfs.empty:
        # Not all risks have mediation factors
        index_columns = [c for c in mfs.columns if "draw_" not in c]
        draw_columns = [c for c in mfs.columns if "draw_" in c]
        mfs = pd.melt(mfs, id_vars=index_columns, value_vars=draw_columns, var_name="draw")
        mfs["draw"] = mfs.draw.str.partition("_")[2].astype(int)
        yield "mediation_factor", mfs
        del mfs

    pafs = core.get_draws([risk], ["population_attributable_fraction"], entity_config.locations)
    normalized = []
    for key, group in pafs.groupby(["cause_id"]):
        group = group.drop(["cause_id"], axis=1)
        group = _normalize(group)
        group["cause_id"] = key
        dims = ["year", "sex", "measure", "age", "age_group_start", "age_group_end", "location_id", "draw", "cause_id"]
        normalized.append(group.set_index(dims))
    yield "population_attributable_fraction", pd.concat(normalized)
    del normalized

    exposures = core.get_draws([risk], ["exposure"], entity_config.locations)
    normalized = []
    for key, group in exposures.groupby(["parameter"]):
        group = group.drop(["parameter"], axis=1)
        group = _normalize(group)
        group["parameter"] = key
        dims = ["year", "sex", "measure", "age", "age_group_start", "age_group_end", "location_id", "draw", "parameter"]
        normalized.append(group.set_index(dims))
    yield "exposure", pd.concat(normalized)
    del normalized

    if risk.exposure_parameters is not None:
        exposure_stds = core.get_draws([risk], ["exposure_standard_deviation"], entity_config.locations)
        exposure_stds = _normalize(exposure_stds)
        yield "exposure_standard_deviation", exposure_stds


def _load_sequela(entity_config: _EntityConfig) -> None:
    sequela = sequelae[entity_config.name]
    measures = ["prevalence", "incidence"]
    result = core.get_draws([sequela], measures, entity_config.locations).drop("sequela_id", axis=1)
    result = _normalize(result)
    result["sequela_id"] = sequela.gbd_id
    for key, group in result.groupby("measure"):
        yield key, group
    del result

    weights = core.get_draws([sequela], ["disability_weight"], entity_config.locations)
    index_columns = [c for c in weights.columns if "draw_" not in c]
    draw_columns = [c for c in weights.columns if "draw_" in c]
    weights = pd.melt(weights, id_vars=index_columns, value_vars=draw_columns, var_name="draw")
    weights["draw"] = weights.draw.str.partition("_")[2].astype(int)
    yield "disability_weight", weights


def _load_healthcare_entity(entity_config: _EntityConfig) -> None:
    healthcare_entity = healthcare_entities[entity_config.name]

    cost = core.get_draws([healthcare_entity], ["cost"], entity_config.locations)
    cost = _normalize(cost)
    yield "cost", cost

    annual_visits = core.get_draws([healthcare_entity], ["annual_visits"], entity_config.locations)
    annual_visits = _normalize(annual_visits)
    yield "annual_visits", annual_visits


def _load_treatment_technology(entity_config: _EntityConfig) -> None:
    treatment_technology = treatment_technologies[entity_config.name]

    if treatment_technology.protection:
        try:
            protection = core.get_draws([treatment_technology], ["protection"], entity_config.locations)
            protection = _normalize(protection)
            yield "protection", protection
        except core.DataMissingError:
            pass

    if treatment_technology.relative_risk:
        relative_risk = core.get_draws([treatment_technology], ["relative_risk"], entity_config.locations)
        relative_risk = _normalize(relative_risk)
        yield "relative_risk", relative_risk

    if treatment_technology.exposure:
        try:
            exposure = core.get_draws([treatment_technology], ["exposure"], entity_config.locations)
            exposure = _normalize(exposure)
            yield "exposure", exposure
        except core.DataMissingError:
            pass

    if treatment_technology.cost:
        cost = core.get_draws([treatment_technology], ["cost"], entity_config.locations)
        cost = _normalize(cost)
        yield "cost", cost


def _load_coverage_gap(entity_config: _EntityConfig) -> None:
    entity = coverage_gaps[entity_config.name]

    try:
        exposure = core.get_draws([entity], ["exposure"], entity_config.locations)
        exposure = _normalize(exposure)
        yield "exposure", exposure
    except core.InvalidQueryError:
        pass

    mediation_factor = core.get_draws([entity], ["mediation_factor"], entity_config.locations)
    if not mediation_factor.empty:
        # TODO: This should probably be an exception. It looks like James was in the middle of doing better
        # error handling in ceam_inputs.core but hasn't finished yet
        mediation_factor = _normalize(mediation_factor)
        yield "mediation_factor", mediation_factor

    relative_risk = core.get_draws([entity], ["relative_risk"], entity_config.locations)
    relative_risk = _normalize(relative_risk)
    yield "relative_risk", relative_risk

    paf = core.get_draws([entity], ["population_attributable_fraction"], entity_config.locations)
    paf = _normalize(paf)
    yield "population_attributable_fraction", paf


def _load_etiology(entity_config: _EntityConfig) -> None:
    entity = etiologies[entity_config.name]

    paf = core.get_draws([entity], ["population_attributable_fraction"], entity_config.locations)
    paf = _normalize(paf)
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
    estimate = get_covariate_estimates([entity.gbd_id], entity_config.locations)

    if entity is covariates.age_specific_fertility_rate:
        columns = ["location_id", "mean_value", "lower_value", "upper_value", "age_group_id", "sex_id", "year_id"]
        estimate = estimate[columns]
        estimate = get_age_group_midpoint_from_age_group_id(estimate)
        estimate = normalize_for_simulation(estimate)
    elif entity in (covariates.live_births_by_sex, covariates.dtp3_coverage_proportion):
        columns = ["location_id", "mean_value", "lower_value", "upper_value", "sex_id", "year_id"]
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
    "treatment_technology": _load_treatment_technology,
    "coverage_gap": _load_coverage_gap,
    "etiology": _load_etiology,
    "covariate": _load_covariate,
    "subregions": _load_subregions,
}
