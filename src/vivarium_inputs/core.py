from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from gbd_mapping import (
    Cause,
    Covariate,
    Etiology,
    ModelableEntity,
    RiskFactor,
    Sequela,
    causes,
)
from loguru import logger

from vivarium_inputs import extract, utilities, utility_data
from vivarium_inputs.globals import (
    COVARIATE_VALUE_COLUMNS,
    DEMOGRAPHIC_COLUMNS,
    DISTRIBUTION_COLUMNS,
    EXTRA_RESIDUAL_CATEGORY,
    MEASURES,
    MINIMUM_EXPOSURE_VALUE,
    DataDoesNotExistError,
    InvalidQueryError,
    Population,
    gbd,
)
from vivarium_inputs.mapping_extension import AlternativeRiskFactor, HealthcareEntity


def get_data(
    entity: ModelableEntity,
    measure: str,
    location: str | int | list[str | int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:
    """Pull raw GBD data for measure and entity.

    This also sets all non-value columns to be the dataframe index.

    Parameters
    ----------
    entity
        Entity for which to pull `measure`.
    measure
        Measure for which to pull data, should be a measure available for the
        kind of entity which `entity` is.
    location
        Location for which to pull data. This can be a location id as an int,
        the location name as a string, or a list of these two data types.
    years
        Years for which to extract data. If None, get most recent year. If 'all',
        get all available data.
    data_type
        The DataType object for which to extract data.

    Returns
    -------
        Raw and slightly reshaped data for the given entity, measure, location, and years.
    """
    measure_handlers = {
        # Cause-like measures
        "incidence_rate": (get_incidence_rate, ("cause", "sequela")),
        "raw_incidence_rate": (get_raw_incidence_rate, ("cause", "sequela")),
        "prevalence": (get_prevalence, ("cause", "sequela")),
        "birth_prevalence": (get_birth_prevalence, ("cause", "sequela")),
        "disability_weight": (get_disability_weight, ("cause", "sequela")),
        "remission_rate": (get_remission_rate, ("cause",)),
        "cause_specific_mortality_rate": (get_cause_specific_mortality_rate, ("cause",)),
        "excess_mortality_rate": (get_excess_mortality_rate, ("cause",)),
        "deaths": (get_deaths, ("cause",)),
        # Risk-like measures
        "exposure": (
            get_exposure,
            (
                "risk_factor",
                "alternative_risk_factor",
            ),
        ),
        "exposure_standard_deviation": (
            get_exposure_standard_deviation,
            ("risk_factor", "alternative_risk_factor"),
        ),
        "exposure_distribution_weights": (
            get_exposure_distribution_weights,
            ("risk_factor", "alternative_risk_factor"),
        ),
        "relative_risk": (get_relative_risk, ("risk_factor",)),
        "population_attributable_fraction": (
            get_population_attributable_fraction,
            ("risk_factor", "etiology"),
        ),
        # Covariate measures
        "estimate": (get_estimate, ("covariate",)),
        # Population measures
        "structure": (get_structure, ("population",)),
        "theoretical_minimum_risk_life_expectancy": (
            get_theoretical_minimum_risk_life_expectancy,
            ("population",),
        ),
        "age_bins": (get_age_bins, ("population",)),
        "demographic_dimensions": (get_demographic_dimensions, ("population",)),
    }

    if measure not in measure_handlers:
        raise InvalidQueryError(f"No functions available to pull data for measure {measure}.")

    handler, entity_types = measure_handlers[measure]

    if entity.kind not in entity_types:
        raise InvalidQueryError(f"{measure.capitalize()} not available for {entity.kind}.")

    if isinstance(location, list):
        location_id = [
            utility_data.get_location_id(loc) if isinstance(loc, str) else loc
            for loc in location
        ]
    else:
        location_id = [
            utility_data.get_location_id(location) if isinstance(location, str) else location
        ]

    data = handler(entity, location_id, years, data_type)
    data = utilities.reshape(data, data_type.value_columns)

    return data


#####################
# HANDLER FUNCTIONS #
#####################


def get_incidence_rate(
    entity: Cause | Sequela,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:
    data = get_data(
        entity,
        "raw_incidence_rate",
        location_id,
        years,
        utilities.DataType("raw_incidence_rate", data_type.type),
    )
    prevalence = get_data(
        entity,
        "prevalence",
        location_id,
        years,
        utilities.DataType("prevalence", data_type.type),
    )
    # Convert from "True incidence" to the incidence rate among susceptibles
    data /= 1 - prevalence
    return data.fillna(0)


def get_raw_incidence_rate(
    entity: Cause | Sequela,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:
    data = extract.extract_data(entity, "incidence_rate", location_id, years, data_type)
    if entity.kind == "cause":
        restrictions_entity = entity
    else:  # sequela
        cause = [c for c in causes if c.sequelae and entity in c.sequelae][0]
        restrictions_entity = cause

    data = utilities.filter_data_by_restrictions(
        data, restrictions_entity, "yld", utility_data.get_age_group_ids()
    )
    data = utilities.normalize(data, data_type.value_columns, fill_value=0)
    data = data.filter(DEMOGRAPHIC_COLUMNS + data_type.value_columns)
    return data


def get_prevalence(
    entity: Cause | Sequela,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:
    data = extract.extract_data(entity, "prevalence", location_id, years, data_type)
    if entity.kind == "cause":
        restrictions_entity = entity
    else:  # sequela
        cause = [c for c in causes if c.sequelae and entity in c.sequelae][0]
        restrictions_entity = cause

    data = utilities.filter_data_by_restrictions(
        data, restrictions_entity, "yld", utility_data.get_age_group_ids()
    )
    data = utilities.normalize(data, data_type.value_columns, fill_value=0)
    data = data.filter(DEMOGRAPHIC_COLUMNS + data_type.value_columns)
    return data


def get_birth_prevalence(
    entity: Cause | Sequela,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:
    data = extract.extract_data(entity, "birth_prevalence", location_id, years, data_type)
    data = data.filter(["years", "sex_id", "location_id"] + data_type.value_columns)
    data = utilities.normalize(data, data_type.value_columns, fill_value=0)
    return data


def get_disability_weight(
    entity: Cause | Sequela,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    if data_type.type != "draws":
        raise utilities.DataTypeNotImplementedError(
            f"Data type(s) {data_type.type} are not supported for this function."
        )

    if entity.kind == "cause":
        data = utility_data.get_demographic_dimensions(
            location_id, draws=True, value=0.0, years=years
        )
        data = data.set_index(
            utilities.get_ordered_index_cols(data.columns.difference(data_type.value_columns))
        )
        if entity.sequelae:
            for sequela in entity.sequelae:
                try:
                    prevalence = get_data(
                        sequela,
                        "prevalence",
                        location_id,
                        years,
                        utilities.DataType("prevalence", data_type.type),
                    )
                except DataDoesNotExistError:
                    # sequela prevalence does not exist so no point continuing with this sequela
                    continue
                disability = get_data(
                    sequela,
                    "disability_weight",
                    location_id,
                    years,
                    utilities.DataType("disability_weight", data_type.type),
                )
                data += prevalence * disability
        cause_prevalence = get_data(
            entity,
            "prevalence",
            location_id,
            years,
            utilities.DataType("prevalence", data_type.type),
        )
        data = (data / cause_prevalence).fillna(0).reset_index()
    else:  # entity.kind == 'sequela'
        try:
            data = extract.extract_data(
                entity, "disability_weight", location_id, years, data_type
            )
            data = utilities.normalize(data, data_type.value_columns)

            cause = [c for c in causes if c.sequelae and entity in c.sequelae][0]
            data = utilities.clear_disability_weight_outside_restrictions(
                data, cause, 0.0, utility_data.get_age_group_ids()
            )
            data = data.filter(DEMOGRAPHIC_COLUMNS + data_type.value_columns)
        except (IndexError, DataDoesNotExistError):
            logger.warning(
                f"{entity.name.capitalize()} has no disability weight data. All values will be 0."
            )
            data = utility_data.get_demographic_dimensions(
                location_id, draws=True, value=0.0, years=years
            )
    return data


def get_remission_rate(
    entity: Cause,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    if data_type.type != "draws":
        raise utilities.DataTypeNotImplementedError(
            f"Data type(s) {data_type.type} are not supported for this function."
        )

    data = extract.extract_data(entity, "remission_rate", location_id, years, data_type)
    data = utilities.filter_data_by_restrictions(
        data, entity, "yld", utility_data.get_age_group_ids()
    )
    data = utilities.normalize(data, data_type.value_columns, fill_value=0)
    data = data.filter(DEMOGRAPHIC_COLUMNS + data_type.value_columns)
    return data


def get_cause_specific_mortality_rate(
    entity: Cause,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    if data_type.type != "draws":
        raise utilities.DataTypeNotImplementedError(
            f"Data type(s) {data_type.type} are not supported for this function."
        )

    deaths = get_data(
        entity, "deaths", location_id, years, utilities.DataType("deaths", data_type.type)
    )
    # population isn't by draws
    pop = get_data(
        Population(),
        "structure",
        location_id,
        years,
        utilities.DataType("structure", data_type.type),
    )
    data = deaths.join(pop, lsuffix="_deaths", rsuffix="_pop")
    data[data_type.value_columns] = data[data_type.value_columns].divide(data.value, axis=0)
    data = data.drop(columns="value")
    return data


def get_excess_mortality_rate(
    entity: Cause,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    if data_type.type != "draws":
        raise utilities.DataTypeNotImplementedError(
            f"Data type(s) {data_type.type} are not supported for this function."
        )

    csmr = get_data(
        entity,
        "cause_specific_mortality_rate",
        location_id,
        years,
        utilities.DataType("cause_specific_mortality_rate", data_type.type),
    )
    prevalence = get_data(
        entity,
        "prevalence",
        location_id,
        years,
        utilities.DataType("prevalence", data_type.type),
    )
    data = (csmr / prevalence).fillna(0)
    data = data.replace([np.inf, -np.inf], 0)
    return data


def get_deaths(
    entity: Cause,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    if data_type.type != "draws":
        raise utilities.DataTypeNotImplementedError(
            f"Data type(s) {data_type.type} are not supported for this function."
        )

    data = extract.extract_data(entity, "deaths", location_id, years, data_type)
    data = utilities.filter_data_by_restrictions(
        data, entity, "yll", utility_data.get_age_group_ids()
    )
    data = utilities.normalize(data, data_type.value_columns, fill_value=0)
    data = data.filter(DEMOGRAPHIC_COLUMNS + data_type.value_columns)
    return data


def get_exposure(
    entity: RiskFactor | AlternativeRiskFactor,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    if data_type.type != "draws":
        raise utilities.DataTypeNotImplementedError(
            f"Data type(s) {data_type.type} are not supported for this function."
        )

    data = extract.extract_data(entity, "exposure", location_id, years, data_type)
    data = data.drop(columns="modelable_entity_id")

    value_columns = data_type.value_columns

    if entity.name in EXTRA_RESIDUAL_CATEGORY:
        cat = EXTRA_RESIDUAL_CATEGORY[entity.name]
        data = data.drop(labels=data.query("parameter == @cat").index)
        data[value_columns] = data[value_columns].clip(lower=MINIMUM_EXPOSURE_VALUE)

    if entity.kind in ["risk_factor", "alternative_risk_factor"]:
        data = utilities.filter_data_by_restrictions(
            data, entity, "outer", utility_data.get_age_group_ids()
        )

    if entity.distribution in ["dichotomous", "ordered_polytomous", "unordered_polytomous"]:
        tmrel_cat = utility_data.get_tmrel_category(entity)
        exposed = data[data.parameter != tmrel_cat]
        unexposed = data[data.parameter == tmrel_cat]

        #  FIXME: We fill 1 as exposure of tmrel category, which is not correct.
        data = pd.concat(
            [
                utilities.normalize(exposed, value_columns, fill_value=0),
                utilities.normalize(unexposed, value_columns, fill_value=1),
            ],
            ignore_index=True,
        )

        # normalize so all categories sum to 1
        cols = list(set(data.columns).difference(value_columns + ["parameter"]))
        sums = data.groupby(cols)[value_columns].sum()
        data = (
            data.groupby("parameter")
            .apply(lambda df: df.set_index(cols).loc[:, value_columns].divide(sums))
            .reset_index()
        )
    else:
        data = utilities.normalize(data, value_columns, fill_value=0)
    data = data.filter(DEMOGRAPHIC_COLUMNS + value_columns + ["parameter"])
    return data


def get_exposure_standard_deviation(
    entity: RiskFactor | AlternativeRiskFactor,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    if data_type.type != "draws":
        raise utilities.DataTypeNotImplementedError(
            f"Data type(s) {data_type.type} are not supported for this function."
        )

    data = extract.extract_data(
        entity, "exposure_standard_deviation", location_id, years, data_type
    )
    data = data.drop(columns="modelable_entity_id")

    exposure = extract.extract_data(entity, "exposure", location_id, years, data_type)
    valid_age_groups = utilities.get_exposure_and_restriction_ages(exposure, entity)
    data = data[data.age_group_id.isin(valid_age_groups)]

    data = utilities.normalize(data, data_type.value_columns, fill_value=0)
    data = data.filter(DEMOGRAPHIC_COLUMNS + data_type.value_columns)
    return data


def get_exposure_distribution_weights(
    entity: RiskFactor | AlternativeRiskFactor,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    if data_type.type and data_type.type != "draws":
        raise utilities.DataTypeNotImplementedError(
            f"Data type(s) {data_type.type} are not supported for this function."
        )

    data = extract.extract_data(
        entity, "exposure_distribution_weights", location_id, years, data_type
    )

    exposure = extract.extract_data(entity, "exposure", location_id, years, data_type)
    valid_ages = utilities.get_exposure_and_restriction_ages(exposure, entity)

    data = data.drop(columns="age_group_id")
    df = []
    for age_id in valid_ages:
        copied = data.copy()
        copied["age_group_id"] = age_id
        df.append(copied)
    data = pd.concat(df)
    data = utilities.normalize(data, DISTRIBUTION_COLUMNS, fill_value=0)
    if years != "all":
        if years:
            years = [years] if isinstance(years, int) else years
            data = data.query(f"year_id in {years}")
        else:
            most_recent_year = utility_data.get_most_recent_year()
            data = data.query(f"year_id=={most_recent_year}")
    data = data.filter(DEMOGRAPHIC_COLUMNS + DISTRIBUTION_COLUMNS)
    data = utilities.wide_to_long(data, DISTRIBUTION_COLUMNS, var_name="parameter")
    return data


def get_relative_risk(
    entity: RiskFactor,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    if data_type.type != "draws":
        raise utilities.DataTypeNotImplementedError(
            f"Data type(s) {data_type.type} are not supported for this function."
        )

    if len(set(location_id)) > 1:
        raise ValueError(
            "Extracting relative risk only supports one location at a time. Provided "
            f"{location_id}."
        )

    data = extract.extract_data(entity, "relative_risk", location_id, years, data_type)

    # FIXME: we don't currently support yll-only causes so I'm dropping them because the data in some cases is
    #  very messed up, with mort = morb = 1 (e.g., aortic aneurysm in the RR data for high systolic bp) -
    #  2/8/19 K.W.
    yll_only_causes = set([c.gbd_id for c in causes if c.restrictions.yll_only])
    data = data[~data.cause_id.isin(yll_only_causes)]

    data = utilities.convert_affected_entity(data, "cause_id")
    morbidity = data.morbidity == 1
    mortality = data.mortality == 1
    data.loc[morbidity & mortality, "affected_measure"] = "incidence_rate"
    data.loc[morbidity & ~mortality, "affected_measure"] = "incidence_rate"
    data.loc[~morbidity & mortality, "affected_measure"] = "cause_specific_mortality_rate"
    data = _filter_relative_risk_to_cause_restrictions(data)
    value_columns = data_type.value_columns
    data = data.filter(
        DEMOGRAPHIC_COLUMNS
        + ["affected_entity", "affected_measure", "parameter"]
        + value_columns
    )
    data = (
        data.groupby(["affected_entity", "parameter"])
        .apply(utilities.normalize, cols_to_fill=value_columns, fill_value=1)
        .reset_index(drop=True)
    )
    if entity.distribution in ["dichotomous", "ordered_polytomous", "unordered_polytomous"]:
        tmrel_cat = utility_data.get_tmrel_category(entity)
        tmrel_mask = data.parameter == tmrel_cat
        data.loc[tmrel_mask, value_columns] = data.loc[tmrel_mask, value_columns].mask(
            np.isclose(data.loc[tmrel_mask, value_columns], 1.0), 1.0
        )
    # Coerce location_id from global to requested location - location_id is list of length 1
    data["location_id"] = location_id[0]

    return data


def get_population_attributable_fraction(
    entity: RiskFactor | Etiology,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    if data_type.type != "draws":
        raise utilities.DataTypeNotImplementedError(
            f"Data type(s) {data_type.type} are not supported for this function."
        )

    value_columns = data_type.value_columns
    causes_map = {c.gbd_id: c for c in causes}
    if entity.kind == "risk_factor":
        data = extract.extract_data(
            entity, "population_attributable_fraction", location_id, years, data_type
        )
        relative_risk = extract.extract_data(
            entity, "relative_risk", location_id, years, data_type
        )

        # FIXME: we don't currently support yll-only causes so I'm dropping them because the data in some cases is
        #  very messed up, with mort = morb = 1 (e.g., aortic aneurysm in the RR data for high systolic bp) -
        #  2/8/19 K.W.
        yll_only_causes = set([c.gbd_id for c in causes if c.restrictions.yll_only])
        data = data[~data.cause_id.isin(yll_only_causes)]
        relative_risk = relative_risk[~relative_risk.cause_id.isin(yll_only_causes)]

        data = (
            data.groupby("cause_id", as_index=False)
            .apply(_filter_by_relative_risk, relative_risk)
            .reset_index(drop=True)
        )

        temp = []
        # We filter paf age groups by cause level restrictions.
        for (c_id, measure), df in data.groupby(["cause_id", "measure_id"]):
            cause = causes_map[c_id]
            measure = "yll" if measure == MEASURES["YLLs"] else "yld"
            df = utilities.filter_data_by_restrictions(
                df, cause, measure, utility_data.get_age_group_ids()
            )
            temp.append(df)
        data = pd.concat(temp, ignore_index=True)

    else:  # etiology
        data = extract.extract_data(
            entity, "etiology_population_attributable_fraction", location_id, years, data_type
        )
        cause = [c for c in causes if entity in c.etiologies][0]
        data = utilities.filter_data_by_restrictions(
            data, cause, "inner", utility_data.get_age_group_ids()
        )
        if np.any(data[value_columns] < 0):
            logger.warning(
                f"{entity.name.capitalize()} has negative values for paf. These will be replaced with 0."
            )
            other_cols = [c for c in data.columns if c not in value_columns]
            data.set_index(other_cols, inplace=True)
            data = data.where(data[value_columns] > 0, 0).reset_index()

    data = utilities.convert_affected_entity(data, "cause_id")
    data.loc[
        data["measure_id"] == MEASURES["YLLs"], "affected_measure"
    ] = "excess_mortality_rate"
    data.loc[data["measure_id"] == MEASURES["YLDs"], "affected_measure"] = "incidence_rate"
    data = (
        data.groupby(["affected_entity", "affected_measure"])
        .apply(utilities.normalize, cols_to_fill=value_columns, fill_value=0)
        .reset_index(drop=True)
    )
    data = data.filter(
        DEMOGRAPHIC_COLUMNS + ["affected_entity", "affected_measure"] + value_columns
    )
    return data


def get_estimate(
    entity: Covariate,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    if data_type.type != "draws":
        raise utilities.DataTypeNotImplementedError(
            f"Data type(s) {data_type.type} are not supported for this function."
        )

    data = extract.extract_data(entity, "estimate", location_id, years, data_type)

    key_columns = ["location_id", "year_id"]
    if entity.by_age:
        key_columns.append("age_group_id")
    if entity.by_sex:
        key_columns.append("sex_id")

    data = data.filter(key_columns + COVARIATE_VALUE_COLUMNS)
    data = utilities.normalize(data, data_type.value_columns)
    data = utilities.wide_to_long(data, COVARIATE_VALUE_COLUMNS, var_name="parameter")
    return data


# FIXME: can this be deleted? It's not in the get_data() mapping.
def get_utilization_rate(
    entity: HealthcareEntity,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:
    data = extract.extract_data(entity, "utilization_rate", location_id, years, data_type)
    data = utilities.normalize(data, data_type.value_columns, fill_value=0)
    data = data.filter(DEMOGRAPHIC_COLUMNS + data_type.value_columns)
    return data


def get_structure(
    entity: Population,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    if data_type.type and data_type.type != "draws":
        raise utilities.DataTypeNotImplementedError(
            f"Data type(s) {data_type.type} are not supported for this function."
        )
    data = extract.extract_data(entity, "structure", location_id, years, data_type)
    data = data.drop(columns="run_id").rename(columns={"population": "value"})
    data = utilities.normalize(data, data_type.value_columns)
    return data


def get_theoretical_minimum_risk_life_expectancy(
    entity: Population,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    if data_type.type and data_type.type != "draws":
        raise utilities.DataTypeNotImplementedError(
            f"Data type(s) {data_type.type} are not supported for this function."
        )

    data = extract.extract_data(
        entity, "theoretical_minimum_risk_life_expectancy", location_id, years, data_type
    )
    data = data.rename(columns={"age": "age_start", "life_expectancy": "value"})
    data["age_end"] = data.age_start.shift(-1).fillna(125.0)
    return data


def get_age_bins(
    entity: Population,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    age_bins = utility_data.get_age_bins()[["age_group_name", "age_start", "age_end"]]
    return age_bins


def get_demographic_dimensions(
    entity: Population,
    location_id: list[int],
    years: int | str | list[int] | None,
    data_type: utilities.DataType,
) -> pd.DataFrame:

    demographic_dimensions = utility_data.get_demographic_dimensions(location_id, years=years)
    demographic_dimensions = utilities.normalize(
        demographic_dimensions, data_type.value_columns
    )
    return demographic_dimensions


####################
# HELPER FUNCTIONS #
####################


def _filter_relative_risk_to_cause_restrictions(data: pd.DataFrame) -> pd.DataFrame:
    """It applies age restrictions according to affected causes
    and affected measures. If affected measure is incidence_rate,
    it applies the yld_age_restrictions. If affected measure is
    excess_mortality_rate, it applies the yll_age_restrictions to filter
    the relative_risk data"""

    age_bins = utility_data.get_age_bins()
    ordered_age_ids = age_bins["age_group_id"].values
    causes_map = {c.name: c for c in causes}
    temp = []
    affected_entities = set(data.affected_entity)
    affected_measures = set(data.affected_measure)
    for cause, measure in product(affected_entities, affected_measures):
        df = data[(data.affected_entity == cause) & (data.affected_measure == measure)]
        cause = causes_map[cause]
        if measure == "cause_specific_mortality_rate":
            start, end = utilities.get_age_group_ids_by_restriction(cause, "yll")
        else:  # incidence_rate
            start, end = utilities.get_age_group_ids_by_restriction(cause, "yld")
        start_index = list(ordered_age_ids).index(start)
        end_index = list(ordered_age_ids).index(end)
        allowed_ids = ordered_age_ids[start_index : (end_index + 1)]
        temp.append(df[df.age_group_id.isin(allowed_ids)])
    data = pd.concat(temp)
    return data


def _filter_by_relative_risk(df: pd.DataFrame, relative_risk: pd.DataFrame) -> pd.DataFrame:
    c_id = df.cause_id.unique()[0]
    rr = relative_risk[relative_risk.cause_id == c_id]
    #  We presume all attributable mortality moves through incidence.
    if set(rr.mortality) == {1} and set(rr.morbidity) == {1}:
        df = df[df.measure_id == MEASURES["YLDs"]]
    return df
