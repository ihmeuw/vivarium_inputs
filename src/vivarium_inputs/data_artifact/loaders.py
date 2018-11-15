from typing import Set
import warnings

from gbd_mapping import causes, risk_factors, sequelae, coverage_gaps, covariates, etiologies
import pandas as pd

from vivarium_public_health.dataset_manager import EntityKey

from vivarium_inputs import core
from vivarium_inputs.data_artifact.utilities import normalize
from vivarium_inputs.utilities import normalize_for_simulation, get_age_group_bins_from_age_group_id
from vivarium_inputs.mapping_extension import healthcare_entities, health_technologies


class DataArtifactError(Exception):
    """Base error raised for issues in data artifact construction."""
    pass


class EntityError(DataArtifactError):
    """Error raised when modeled entities are improperly specified."""
    pass


CAUSE_BY_ID = {c.gbd_id: c for c in causes if c is not None}
RISK_BY_ID = {r.gbd_id: r for r in risk_factors}

AGE_COLS = ['age_group_start', 'age_group_end']
YEAR_COLS = ['year_start', 'year_end']


def loader(entity_key: EntityKey, location: str, modeled_causes: Set[str], all_measures: bool=False):
    entity_data = {
        "cause": {
            "mapping": causes,
            "getter": get_cause_data,
            "measures": ["death", "prevalence", "incidence", "cause_specific_mortality",
                         "excess_mortality", "remission", "sequelae", "etiologies", "restrictions"]
        },
        "risk_factor": {
            "mapping": risk_factors,
            "getter": get_risk_data,
            "measures": ["affected_causes", "affected_risk_factors", "restrictions", "distribution",
                         "exposure_parameters", "levels", "tmred", "exposure", "exposure_standard_deviation",
                         "relative_risk", "population_attributable_fraction", "ensemble_weights"],
        },
        "sequela": {
            "mapping": sequelae,
            "getter": get_sequela_data,
            "measures": ["healthstate", "prevalence", "incidence", "disability_weight"],
        },
        "healthcare_entity": {
            "mapping": healthcare_entities,
            "getter": get_healthcare_entity_data,
            "measures": ["cost", "annual_visits"]
        },
        "health_technology": {
            "mapping": health_technologies,
            "getter": get_health_technology_data,
            "measures": ["cost"]
        },
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
            "measures": ["structure", "age_bins", "theoretical_minimum_risk_life_expectancy"],
        },
        "covariate": {
            "mapping": covariates,
            "getter": get_covariate_data,
            "measures": ["estimate"]
        },
        "subregions": {
            "mapping": {'': None},
            "getter": get_subregion_data,
            "measures": ["sub_region_ids"],
        },
        "dimensions": {
            "mapping": {'': None},
            "getter": get_dimension_data,
            "measures": ["full_space"]
        },
    }
    mapping, getter, measures = entity_data[entity_key.type].values()
    entity = mapping[entity_key.name]
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
    elif measure in ["death", "prevalence", "incidence", "cause_specific_mortality", "excess_mortality"]:
        data = core.get_draws(cause, measure, location)
        data = normalize(data)[["location", "sex", "draw", "value"] + AGE_COLS + YEAR_COLS]
    elif measure == "remission":
        data = get_cause_remission(cause, location)
    else:
        raise NotImplementedError(f"Unknown measure {measure} for cause {cause.name}")

    return data


def get_risk_data(risk, measure, location, modeled_causes):
    if measure in ["affected_causes", "affected_risk_factors", "restrictions",
                   "distribution", "exposure_parameters", "levels", "tmred"]:
        data = get_risk_metadata(risk, measure, modeled_causes)
    elif measure == "exposure":
        data = get_risk_exposure(risk, location)
    elif measure == "exposure_standard_deviation":
        data = get_risk_exposure_standard_deviation(risk, location)
    elif measure == "relative_risk":
        data = get_risk_relative_risk(risk, location)
    elif measure == "population_attributable_fraction":
        if risk.distribution not in ['normal', 'lognormal', 'ensemble']:
            raise DataArtifactError(f"PAF for {risk.name} should not be loaded from the artifact. PAF for "
                                    f"Categorical risk should be computed directly ")
        else:
            data = get_risk_population_attributable_fraction(risk, location)
    elif measure == "ensemble_weights":
        data = get_risk_ensemble_weights(risk)
    else:
        raise NotImplementedError(f"Unknown measure {measure} for risk {risk.name}")
    return data


def get_sequela_data(sequela, measure, location, _):
    if measure == "healthstate":
        data = sequela.healthstate.name
    elif measure in ["incidence", "prevalence"]:
        data = core.get_draws(sequela, measure, location).drop("sequela_id", axis=1)
        data = normalize(data)[["location", "sex", "draw", "value"] + AGE_COLS + YEAR_COLS]
        data["sequela_id"] = sequela.gbd_id
    elif measure == "disability_weight":
        data = core.get_draws(sequela, "disability_weight", location)
        index_columns = [c for c in data.columns if "draw_" not in c]
        draw_columns = [c for c in data.columns if "draw_" in c]
        data = pd.melt(data, id_vars=index_columns, value_vars=draw_columns, var_name="draw")
        data["draw"] = data.draw.str.partition("_")[2].astype(int)
    else:
        raise NotImplementedError(f"Unknown measure {measure} for sequela {sequela.name}")
    return data


def get_healthcare_entity_data(healthcare_entity, measure, location, _):
    if measure == "cost":
        data = core.get_draws(healthcare_entity, "cost", location)
        data = normalize(data)
        data = data.loc[data.sex == 'Male', ["location", "draw", "value"] + YEAR_COLS]
    elif measure == "annual_visits":
        data = core.get_draws(healthcare_entity, "annual_visits", location)
        data = normalize(data)
        data = data[["sex", "value", "draw"] + AGE_COLS + YEAR_COLS]
    else:
        raise NotImplementedError(f"Unknown measure {measure} for healthcare_entity {healthcare_entity.name}")
    return data


def get_health_technology_data(healthcare_technology, measure, location, _):
    if measure == "cost":
        data = core.get_draws(healthcare_technology, "cost", location)
        data = normalize(data)[["location", "draw", "value", "health_technology"] + YEAR_COLS]
    else:
        raise NotImplementedError(f"Unknown measure {measure} for healthcare_entity {healthcare_technology.name}")
    return data


def get_coverage_gap_data(coverage_gap, measure, location, modeled_causes):
    if measure in ["affected_causes", "affected_risk_factors", "restrictions", "distribution", "levels"]:
        data = get_coverage_gap_metadata(coverage_gap, measure, modeled_causes)
    elif measure == "exposure":
        data = get_coverage_gap_exposure(coverage_gap, location)
    elif measure == "relative_risk":
        data = get_coverage_gap_relative_risk(coverage_gap, location)
    else:
        raise NotImplementedError(f"Unknown measure {measure} for coverage_gap {coverage_gap.name}")
    return data


def get_etiology_data(etiology, measure, location, _):
    if measure == "population_attributable_fraction":
        data = core.get_draws(etiology, "population_attributable_fraction", location)
        data = normalize(data)
        data["cause"] = data.cause_id.apply(lambda cause_id: CAUSE_BY_ID[cause_id].name)
        data = data[["location", "cause", "sex", "draw", "value"] + AGE_COLS + YEAR_COLS]
    else:
        raise NotImplementedError(f"Unknown measure {measure} for etiology {etiology.name}")
    return data


def get_population_data(_, measure, location, __):
    if measure == "structure":
        data = core.get_population(location)
        data = normalize_for_simulation(data)
        data = get_age_group_bins_from_age_group_id(data)
    elif measure == "age_bins":
        data = core.get_age_bins()[["age_group_years_start", "age_group_years_end", "age_group_name"]]
        data = data.rename(columns={"age_group_years_start": "age_group_start", "age_group_years_end": "age_group_end"})
    elif measure == "theoretical_minimum_risk_life_expectancy":
        data = core.get_theoretical_minimum_risk_life_expectancy()
    else:
        raise NotImplementedError(f"Unknown measure {measure} for population.")
    return data


def get_covariate_data(covariate, measure, location, _):
    if measure == "estimate":
        data = core.get_covariate_estimate(covariate, location)
        expected_columns = ["location", "mean_value", "lower_value", "upper_value",
                            "sex_id", "year_id", "age_group_id"]

        if not set(expected_columns).issubset(data.columns):
            raise NotImplementedError(f"Unsupported covariate {covariate.name}. It does not not "
                                       f"contain the columns {expected_columns}.")
        data = data[expected_columns]
        if (data['age_group_id'] == 22).all():
            data = data.drop(['age_group_id'], axis=1)
            warnings.warn(f"Covariate \"{covariate.name}\" contains data for the age group all ages, "
                          f"so the age column is being dropped.")
        else:
            data = get_age_group_bins_from_age_group_id(data)
        data = normalize_for_simulation(data)
    else:
        raise NotImplementedError(f"Unknown or unsupported measure {measure} for covariate {covariate.name}.")
    return data


def get_subregion_data(_, measure, location, __):
    if measure == "sub_region_ids":
        data = pd.DataFrame(core.get_subregions([location]))
        data = data.melt(var_name="location", value_name="subregion_id")
    else:
        raise NotImplementedError(f"Unknown measure {measure} for subregion data.")
    return data


def get_dimension_data(_, measure, location, __):
    if measure == "full_space":
        age_bins = core.get_age_bins()
        estimation_years = core.get_estimation_years()
        data = [range(min(estimation_years), max(estimation_years) + 1),
                ["Male", "Female"], age_bins.age_group_id, [location]]
        data = pd.MultiIndex.from_product(data, names=["year", "sex", "age_group_id", "location"])
        data = data.to_frame().reset_index(drop=True)
    else:
        raise NotImplementedError(f"Unknown measure {measure} for dimensions")
    return data


##########
# Causes #
##########


def get_cause_metadata(entity, field):
    if field == "restrictions":
        data = entity.restrictions.to_dict()
    else:  # field in ["sequela", "etiologies"]:
        if entity[field] is not None:
            data = [sub_entity.name for sub_entity in entity[field]]
        else:
            data = None
    return data


def get_cause_remission(cause, location):
    result = core.get_draws(cause, "remission", location)
    result = normalize(result)[["location", "sex", "draw", "value"] + AGE_COLS + YEAR_COLS]
    return result


#########
# Risks #
#########


def get_risk_metadata(risk, measure, modeled_causes):
    if measure in ["restrictions", "exposure_parameters", "levels", "tmred"]:
        if risk[measure] is not None:
            data = risk[measure].to_dict()
        else:
            data = None
    elif measure == "affected_causes":
        data = [c.name for c in risk.affected_causes if c.name in modeled_causes]
    elif measure == "affected_risk_factors":
        # FIXME: Update mapping to include affected risks (mediation)
        data = []
    else:  # measure == "distribution"
        data = risk[measure]
    return data


def get_risk_exposure(risk, location):
    exposure = core.get_draws(risk, "exposure", location)
    result = normalize(exposure)
    result = result[["location", "sex", "draw", "value", "parameter"] + AGE_COLS + YEAR_COLS]
    return result


def get_risk_exposure_standard_deviation(risk, location):
    if risk.exposure_parameters is not None:
        exposure_sds = core.get_draws(risk, "exposure_standard_deviation", location)
        exposure_sds = normalize(exposure_sds)
        data = exposure_sds[["location", "sex", "draw", "value"] + AGE_COLS + YEAR_COLS]
    else:
        data = None
    return data


def get_risk_relative_risk(risk, location):
    rrs = core.get_draws(risk, "relative_risk", location)
    rrs = normalize(rrs)
    rrs['cause'] = rrs['cause_id'].apply(lambda cause_id: CAUSE_BY_ID[cause_id].name)
    rrs.drop(['cause_id'], axis=1, inplace=True)
    result = rrs[["location", "sex", "draw", "value", "parameter", "cause"] + AGE_COLS + YEAR_COLS]
    return result


def get_risk_population_attributable_fraction(risk, location):
    paf = core.get_draws(risk, 'population_attributable_fraction', location)
    paf = normalize(paf)
    paf['cause'] = paf['cause_id'].apply(lambda cause_id: CAUSE_BY_ID[cause_id].name)
    paf.drop(['cause_id'], axis=1, inplace=True)
    result = paf[["location", "sex", "draw", "value", "cause"] + AGE_COLS + YEAR_COLS]
    return result


def get_risk_ensemble_weights(risk):
    if risk.distribution == "ensemble":
        weights = core.get_ensemble_weights(risk)
        weights = weights.drop(["location_id", "risk_id"], axis=1)
        weights = normalize_for_simulation(weights)
        data = get_age_group_bins_from_age_group_id(weights)
    else:
        data = None
    return data


#################
# Coverage Gaps #
#################


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


def get_coverage_gap_exposure(coverage_gap, location):
    exposure = core.get_draws(coverage_gap, "exposure", location)
    result = normalize(exposure)
    result = result[["location", "sex", "draw", "value", "parameter"] + AGE_COLS + YEAR_COLS]
    return result


def get_coverage_gap_relative_risk(coverage_gap, location):
    data = core.get_draws(coverage_gap, "relative_risk", location)
    if data.empty:
        data = None
    else:
        data = normalize(data)
        data = data.rename(columns={'cause_id': 'cause', 'rei_id': 'risk_factor'})
        data['cause'] = data['cause'].apply(lambda c:CAUSE_BY_ID[c].name if c in CAUSE_BY_ID else c)
        data['risk_factor'] = data['risk_factor'].apply(lambda c:RISK_BY_ID[c].name if c in RISK_BY_ID else c)
        data = data[['location', 'cause', 'risk_factor', 'sex', 'draw', 'value', 'parameter'] + AGE_COLS + YEAR_COLS]
    return data

