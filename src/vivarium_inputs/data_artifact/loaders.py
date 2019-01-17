from typing import Set

from gbd_mapping import causes, risk_factors, sequelae, covariates, etiologies, coverage_gaps
from vivarium_public_health.dataset_manager import EntityKey

from vivarium_inputs.globals import InvalidQueryError
from vivarium_inputs.interface import (get_measure, get_population_structure, get_age_bins,
                                       get_theoretical_minimum_risk_life_expectancy, get_demographic_dimensions)
from vivarium_inputs.mapping_extension import alternative_risk_factors


CAUSE_BY_ID = {c.gbd_id: c for c in causes if c is not None}
RISK_BY_ID = {r.gbd_id: r for r in risk_factors}


def loader(entity_key: EntityKey, location: str, modeled_causes: Set[str], all_measures: bool = False):
    entity_data = {
        "cause": {
            "mapping": causes,
            "getter": get_cause_data,
            "measures": ["prevalence", "incidence", "cause_specific_mortality", "disability_weight",
                         "excess_mortality", "remission", "sequelae", "etiologies", "restrictions"]
        },
        "risk_factor": {
            "mapping": risk_factors,
            "getter": get_risk_data,
            "measures": ["affected_causes", "affected_risk_factors", "restrictions", "distribution",
                         "exposure_parameters", "categories", "tmred", "exposure", "exposure_standard_deviation",
                         "relative_risk", "population_attributable_fraction", "exposure_distribution_weights"],
        },
        "alternative_risk_factor": {
            "mapping": alternative_risk_factors,
            "getter": get_risk_data,
            "measures": ["affected_causes", "affected_risk_factors", "restrictions", "distribution",
                         "exposure_parameters", "categories", "tmred", "exposure", "exposure_standard_deviation",
                         "relative_risk", "population_attributable_fraction", "exposure_distribution_weights"],
        },
        "sequela": {
            "mapping": sequelae,
            "getter": get_sequela_data,
            "measures": ["healthstate", "prevalence", "incidence", "disability_weight"],
        },
        # "healthcare_entity": {
        #     "mapping": healthcare_entities,
        #     "getter": get_healthcare_entity_data,
        #     "measures": ["cost", "annual_visits"]
        # },
        # "health_technology": {
        #     "mapping": health_technologies,
        #     "getter": get_health_technology_data,
        #     "measures": ["cost", "effects", "coverage"]
        # },
        "coverage_gap": {
            "mapping": coverage_gaps,
            "getter": get_coverage_gap_data,
            "measures": ["affected_causes", "affected_risk_factors", "restrictions", "distribution", "levels",
                         "relative_risk", "exposure"]
        },
        "etiology": {
            "mapping": etiologies,
            "getter": get_etiology_data,
            "measures": ["population_attributable_fraction"],
        },
        "population": {
            "mapping": {'': None},
            "getter": get_population_data,
            "measures": ["structure", "age_bins", "theoretical_minimum_risk_life_expectancy", "demographic_dimensions"],
        },
        "covariate": {
            "mapping": covariates,
            "getter": get_covariate_data,
            "measures": ["estimate"]
        },
        # "subregions": {
        #     "mapping": {'': None},
        #     "getter": get_subregion_data,
        #     "measures": ["sub_region_ids"],
        # },
    }
    mapping, getter, measures = entity_data[entity_key.type].values()

    entity = mapping[entity_key.name]

    if entity_key.measure not in measures:
        raise InvalidQueryError(f"Unknown measure {entity_key.measure} for entity {entity.name}")

    if not all_measures:
        return getter(entity, entity_key.measure, location, modeled_causes)
    else:
        return data_generator(entity, measures, location, modeled_causes, getter)


def data_generator(entity, measures, location, modeled_causes, getter):
    for measure in measures:
        data = getter(entity, measure, location, modeled_causes)
        if data is not None:
            yield measure, data
            del data


def get_cause_data(cause, measure, location, _):
    if measure in ["sequelae", "etiologies", "restrictions"]:
        data = get_cause_metadata(cause, measure)
    else:
        data = get_measure(cause, measure, location)
    return data


def get_risk_data(risk, measure, location, modeled_causes):
    if measure in ["affected_causes", "affected_risk_factors", "restrictions",
                   "distribution", "exposure_parameters", "categories", "tmred"]:
        data = get_risk_metadata(risk, measure, modeled_causes)
    else:
        data = get_measure(risk, measure, location)
    return data


def get_sequela_data(sequela, measure, location, _):
    if measure == "healthstate":
        data = sequela.healthstate.name
    else:
        data = get_measure(sequela, measure, location)
    return data


# def get_healthcare_entity_data(healthcare_entity, measure, location, _):
#     if measure == "cost":
#         data = core.get_draws(healthcare_entity, "cost", location)
#         data = normalize(data)
#         data = data.loc[data.sex == 'Male', ["location", "draw", "value"] + YEAR_COLS]
#     elif measure == "annual_visits":
#         data = core.get_draws(healthcare_entity, "annual_visits", location)
#         data = normalize(data)
#         data = data[["sex", "value", "draw"] + AGE_COLS + YEAR_COLS]
#     else:
#         raise NotImplementedError(f"Unknown measure {measure} for healthcare_entity {healthcare_entity.name}")
#     return data


# def get_health_technology_data(healthcare_technology, measure, location, _):
#     if measure == "cost":
#         data = core.get_draws(healthcare_technology, "cost", location)
#         data = normalize(data)[["location", "draw", "value", "health_technology"] + YEAR_COLS]
#     elif measure == "effects":
#         data = core.get_draws(healthcare_technology, "effects", location)
#         data = normalize(data)[["location", "measure", "medication", "dosage", "draw", "value"]]
#     elif measure == "coverage":
#         data = core.get_draws(healthcare_technology, "coverage", location)
#         data = normalize(data)[AGE_COLS + YEAR_COLS
#                                + ["sex", "location", "measure", "medication", "dosage", "draw", "value"]]
#     else:
#         raise NotImplementedError(f"Unknown measure {measure} for healthcare_entity {healthcare_technology.name}")
#     return data


def get_coverage_gap_data(coverage_gap, measure, location, modeled_causes):
    if measure in ["affected_causes", "affected_risk_factors", "restrictions", "distribution", "levels"]:
        data = get_coverage_gap_metadata(coverage_gap, measure, modeled_causes)
    else:
        data = get_measure(coverage_gap, measure, location)
    return data


def get_etiology_data(etiology, measure, location, _):
    data = get_measure(etiology, measure, location)
    return data


def get_population_data(_, measure, location, __):
    if measure == "structure":
        data = get_population_structure(location)
    elif measure == "theoretical_minimum_risk_life_expectancy":
        data = get_theoretical_minimum_risk_life_expectancy()
    elif measure == "age_bins":
        data = get_age_bins()
    else:  # measure == "demographic_dimensions"
        data = get_demographic_dimensions(location)
    return data


def get_covariate_data(covariate, measure, location, _):
    data = get_measure(covariate, measure, location)
    return data


# def get_subregion_data(_, measure, location, __):
#     if measure == "sub_region_ids":
#         data = pd.DataFrame(core.get_subregions([location]))
#         data = data.melt(var_name="location", value_name="subregion_id")
#     else:
#         raise NotImplementedError(f"Unknown measure {measure} for subregion data.")
#     return data


# def get_dimension_data(_, measure, location, __):
#     if measure == "full_space":
#         age_bins = core.get_age_bins()
#         estimation_years = core.get_estimation_years()
#         data = [range(min(estimation_years), max(estimation_years) + 1),
#                 ["Male", "Female"], age_bins.age_group_id, [location]]
#         data = pd.MultiIndex.from_product(data, names=["year", "sex", "age_group_id", "location"])
#         data = data.to_frame().reset_index(drop=True)
#     else:
#         raise NotImplementedError(f"Unknown measure {measure} for dimensions")
#     return data


######################
# Metadata unpackers #
######################


def get_cause_metadata(entity, field):
    if field == "restrictions":
        data = entity.restrictions.to_dict()
    else:  # field in ["sequela", "etiologies"]:
        if entity[field] is not None:
            data = [sub_entity.name for sub_entity in entity[field]]
        else:
            data = None
    return data


def get_risk_metadata(risk, measure, modeled_causes):
    if measure in ["restrictions", "exposure_parameters", "categories", "tmred"]:
        if risk[measure] is not None:
            data = risk[measure].to_dict()
        else:
            data = None
    elif measure in ["affected_causes", 'affected_risk_factors']:
        data = [c.name for c in risk.affected_causes if c.name in modeled_causes]
    else:  # measure == "distribution"
        data = risk[measure]
    return data


def get_coverage_gap_metadata(coverage_gap, measure, modeled_causes):
    if measure in ["restrictions", "levels"]:
        data = coverage_gap[measure].to_dict()
    elif measure == "affected_causes":
        data = [c.name for c in coverage_gap.affected_causes if c.name in modeled_causes]
    elif measure == "affected_risk_factors":
        data = [r.name for r in coverage_gap.affected_risk_factors]
    else:  # measure == "distribution"
        data = coverage_gap[measure]
    return data
