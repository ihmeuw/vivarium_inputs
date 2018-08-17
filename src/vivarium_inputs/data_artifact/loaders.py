import typing
import warnings

from gbd_mapping import causes, risks, sequelae, coverage_gaps, covariates, etiologies
import pandas as pd

from vivarium_inputs import core
from vivarium_inputs.data_artifact.utilities import normalize
from vivarium_inputs.utilities import normalize_for_simulation, get_age_group_midpoint_from_age_group_id
from vivarium_inputs.mapping_extension import healthcare_entities

if typing.TYPE_CHECKING:
    from vivarium_inputs.data_artifact.builder import EntityConfig


class DataArtifactError(Exception):
    """Base error raised for issues in data artifact construction."""
    pass


class EntityError(DataArtifactError):
    """Error raised when modeled entities are improperly specified."""
    pass


CAUSE_BY_ID = {c.gbd_id: c for c in causes if c is not None}
RISK_BY_ID = {r.gbd_id: r for r in risks}


def _load_cause(entity_config: EntityConfig) -> None:
    measures = ["death", "prevalence", "incidence", "cause_specific_mortality", "excess_mortality"]
    cause = causes[entity_config.entity_key.name]

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
    result = normalize(result)
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
                group = normalize(group)
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
        result = core.get_draws([causes[entity_config.entity_key.name]], measures, entity_config.locations)
        if not result.empty:
            result = normalize(result)[["year", "location", "sex", "age", "draw", "value"]]
            yield "remission", result
        else:
            yield "remission", None
    except core.InvalidQueryError:
        yield "remission", None


def _load_risk_factor(entity_config: EntityConfig) -> None:
    risk = risks[entity_config.entity_key.name]

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
        group = normalize(group)
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
        group = normalize(group)
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
        exposure_sds = normalize(exposure_sds)
        exposure_sds = exposure_sds[["year", "location", "sex", "age", "draw", "value"]]
        yield "exposure_standard_deviation", exposure_sds
    else:
        yield "exposure_standard_deviation", None


def _load_sequela(entity_config: EntityConfig) -> None:
    sequela = sequelae[entity_config.entity_key.name]

    yield "healthstate", sequela.healthstate.name

    measures = ["prevalence", "incidence"]
    result = core.get_draws([sequela], measures, entity_config.locations).drop("sequela_id", axis=1)
    result = normalize(result)
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


def _load_healthcare_entity(entity_config: EntityConfig) -> None:
    healthcare_entity = healthcare_entities[entity_config.entity_key.name]

    cost = core.get_draws([healthcare_entity], ["cost"], entity_config.locations)
    cost = normalize(cost)
    cost = cost[["year", "location", "draw", "value"]]
    yield "cost", cost

    annual_visits = core.get_draws([healthcare_entity], ["annual_visits"], entity_config.locations)
    annual_visits = normalize(annual_visits)
    annual_visits = annual_visits[["year", "sex", "age", "value", "draw"]]
    yield "annual_visits", annual_visits


def _load_coverage_gap(entity_config: EntityConfig) -> None:
    entity = coverage_gaps[entity_config.entity_key.name]

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
        exposure = normalize(exposure)
        yield "exposure", exposure
    except core.InvalidQueryError:
        yield "exposure", None

    relative_risk = core.get_draws([entity], ["relative_risk"], entity_config.locations)
    relative_risk = normalize(relative_risk)
    relative_risk["cause"] = relative_risk.cause_id.apply(lambda cause_id: CAUSE_BY_ID[cause_id].name)
    relative_risk = relative_risk.drop('cause_id', axis=1)
    yield "relative_risk", relative_risk

    paf = core.get_draws([entity], ["population_attributable_fraction"], entity_config.locations)
    if not paf.empty:
        paf = normalize(paf)
        paf["cause"] = paf.cause_id.apply(lambda cause_id: CAUSE_BY_ID[cause_id].name)
        paf = paf.drop('cause_id', axis=1)
        yield "population_attributable_fraction", paf


def _load_etiology(entity_config: EntityConfig) -> None:
    entity = etiologies[entity_config.entity_key.name]

    paf = core.get_draws([entity], ["population_attributable_fraction"], entity_config.locations)
    paf = normalize(paf)
    paf["cause"] = paf.cause_id.apply(lambda cause_id: CAUSE_BY_ID[cause_id].name)
    paf = paf[["year", "location", "cause", "sex", "age", "draw", "value"]]
    yield "population_attributable_fraction", paf


def _load_population(entity_config: EntityConfig) -> None:
    pop = core.get_populations(entity_config.locations)
    pop = normalize_for_simulation(pop)
    pop = get_age_group_midpoint_from_age_group_id(pop)
    yield "structure", pop

    bins = core.get_age_bins()[["age_group_years_start", "age_group_years_end", "age_group_name"]]
    bins = bins.rename(columns={"age_group_years_start": "age_group_start", "age_group_years_end": "age_group_end"})
    yield "age_bins", bins

    yield "theoretical_minimum_risk_life_expectancy", core.get_theoretical_minimum_risk_life_expectancy()


def _load_covariate(entity_config: EntityConfig) -> None:
    entity = covariates[entity_config.entity_key.name]
    location_ids = [core.get_location_ids_by_name()[l] for l in entity_config.locations]
    estimate = core.get_covariate_estimates([entity.gbd_id], location_ids)
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


def _load_subregions(entity_config: EntityConfig) -> None:
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
