from typing import List, Optional, Union

import pandas as pd
from gbd_mapping import Cause, Covariate, Etiology, ModelableEntity, RiskFactor, Sequela

import vivarium_inputs.validation.raw as validation
from vivarium_inputs.globals import (
    DRAW_COLUMNS,
    MEASURES,
    METRICS,
    OTHER_MEID,
    DataAbnormalError,
    DataDoesNotExistError,
    EmptyDataFrameException,
    InputsException,
    NoBestVersionsException,
    Population,
    gbd,
)
from vivarium_inputs.mapping_extension import AlternativeRiskFactor, HealthcareEntity
from vivarium_inputs.utilities import (
    filter_to_most_detailed_causes,
    process_kidney_dysfunction_exposure,
)


def extract_data(
    entity: ModelableEntity,
    measure: str,
    location_id: List[int],
    validate: bool = True,
    years: Optional[Union[int, str, List[int]]] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Check metadata for the requested entity-measure pair. Pull raw data from
    GBD. The only filtering that occurs is by applicable measure id, metric id,
    or to most detailed causes where relevant. If validate is turned on, will
    also pull any additional data needed for raw validation and call raw
    validation on the extracted data.

    Parameters
    ----------
    entity
        Entity for which to extract data.
    measure
        Measure for which to extract data.
    location_id
        Location for which to extract data.
    validate
        Flag indicating whether additional data needed for raw validation
        should be extracted and whether raw validation should be performed.
        Should only be set to False if data is being extracted for
        investigation. Never extract data for a simulation without validation.
    years
        Years for which to extract data. If None, get most recent year. If 'all',
        get all available data. Defaults to None.

    Returns
    -------
    Data for the entity-measure pair and specific location requested.

    Raises
    ------
    DataDoesNotExistError
        If data for the entity-measure-location set requested does not exist.

    """
    extractors = {
        # Cause-like measures
        "incidence_rate": (extract_incidence_rate, {}),
        "prevalence": (extract_prevalence, {}),
        "birth_prevalence": (extract_birth_prevalence, {}),
        "disability_weight": (extract_disability_weight, {}),
        "remission_rate": (extract_remission_rate, {}),
        "deaths": (extract_deaths, {"population": extract_structure}),
        # Risk-like measures
        "exposure": (extract_exposure, {}),
        "exposure_standard_deviation": (
            extract_exposure_standard_deviation,
            {"exposure": extract_exposure},
        ),
        "exposure_distribution_weights": (extract_exposure_distribution_weights, {}),
        "relative_risk": (extract_relative_risk, {"exposure": extract_exposure}),
        "population_attributable_fraction": (
            extract_population_attributable_fraction,
            {"exposure": extract_exposure, "relative_risk": extract_relative_risk},
        ),
        "etiology_population_attributable_fraction": (
            extract_population_attributable_fraction,
            {},
        ),
        "mediation_factors": (extract_mediation_factors, {}),
        # Covariate measures
        "estimate": (extract_estimate, {}),
        # Health system measures
        "utilization_rate": (extract_utilization_rate, {}),
        # Population measures
        "structure": (extract_structure, {}),
        "theoretical_minimum_risk_life_expectancy": (
            extract_theoretical_minimum_risk_life_expectancy,
            {},
        ),
    }

    validation.check_metadata(entity, measure)

    # update year_id value for gbd calls
    if years == None:  # default to most recent year
        year_id = gbd.get_most_recent_year()
    elif years == "all":
        year_id = None
    else:
        estimation_years = gbd.get_estimation_years()
        if years not in estimation_years:
            raise ValueError(f"years must be in {estimation_years}. You provided {years}.")
        year_id = years

    try:
        main_extractor, additional_extractors = extractors[measure]
        data = main_extractor(entity, location_id, year_id)
    except (
        ValueError,
        AssertionError,
        EmptyDataFrameException,
        NoBestVersionsException,
        InputsException,
    ) as e:
        if isinstance(
            e, ValueError
        ) and f"Metadata associated with rei_id = {entity.gbd_id}" not in str(e):
            raise e
        elif (
            isinstance(e, AssertionError)
            and f"Invalid covariate_id {entity.gbd_id}" not in str(e)
            and "No best model found" not in str(e)
        ):
            raise e
        elif isinstance(e, InputsException) and measure != "birth_prevalence":
            raise e
        else:
            raise DataDoesNotExistError(
                f"{measure.capitalize()} data for {entity.name} does not exist."
            )

    # drop extra draw columns
    existing_draw_cols = [col for col in data if col.startswith("draw_")]
    extra_draw_cols = [col for col in existing_draw_cols if col not in DRAW_COLUMNS]
    data = data.drop(columns=extra_draw_cols, errors="ignore")

    if validate:
        additional_data = {
            name: extractor(entity, location_id, year_id)
            for name, extractor in additional_extractors.items()
        }
        if year_id:  # if not pulling all years
            additional_data["estimation_years"] = [year_id]
        validation.validate_raw_data(data, entity, measure, location_id, **additional_data)

    return data


def extract_prevalence(
    entity: Union[Cause, Sequela],
    location_id: List[int],
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    data = gbd.get_incidence_prevalence(
        entity_id=entity.gbd_id,
        location_id=location_id,
        entity_type=entity.kind,
        year_id=year_id,
    )
    data = data[data.measure_id == MEASURES["Prevalence"]]
    return data


def extract_incidence_rate(
    entity: Union[Cause, Sequela],
    location_id: List[int],
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    data = gbd.get_incidence_prevalence(
        entity_id=entity.gbd_id,
        location_id=location_id,
        entity_type=entity.kind,
        year_id=year_id,
    )
    data = data[data.measure_id == MEASURES["Incidence rate"]]
    return data


def extract_birth_prevalence(
    entity: Union[Cause, Sequela],
    location_id: List[int],
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    data = gbd.get_birth_prevalence(
        entity_id=entity.gbd_id,
        location_id=location_id,
        entity_type=entity.kind,
        year_id=year_id,
    )
    data = data[data.measure_id == MEASURES["Incidence rate"]]
    return data


def extract_remission_rate(
    entity: Cause,
    location_id: List[int],
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    data = gbd.get_modelable_entity_draws(entity.me_id, location_id, year_id=year_id)
    data = data[data.measure_id == MEASURES["Remission rate"]]
    return data


def extract_disability_weight(
    entity: Sequela,
    location_id: List[int],
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    disability_weights = gbd.get_auxiliary_data(
        "disability_weight",
        entity.kind,
        "all",
        location_id,
    )
    disability_data = disability_weights.loc[
        disability_weights.healthstate_id == entity.healthstate.gbd_id, :
    ]
    # Update location_id to match original location id
    # Note: The flat file we read data from in gbd.get_auxiliary_data only has location_id 1
    # because disability weights are the same for all locations
    data = []
    for loc_id in location_id:
        loc_data = disability_data.copy()
        loc_data["location_id"] = loc_id
        data.append(loc_data)
    data = pd.concat(data)
    if year_id:  # if not pulling all years
        data["year_id"] = year_id
    return data


def extract_deaths(
    entity: Cause,
    location_id: List[int],
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    data = gbd.get_codcorrect_draws(entity.gbd_id, location_id, year_id=year_id)
    data = data[data.measure_id == MEASURES["Deaths"]]
    return data


def extract_exposure(
    entity: Union[RiskFactor, AlternativeRiskFactor],
    location_id: int,
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    if entity.kind == "risk_factor":
        data = gbd.get_exposure(entity.gbd_id, location_id, year_id=year_id)
        if entity.gbd_id == 341:
            data = process_kidney_dysfunction_exposure(data)
        allowable_measures = [
            MEASURES["Proportion"],
            MEASURES["Continuous"],
            MEASURES["Prevalence"],
        ]
        proper_measure_id = set(data.measure_id).intersection(allowable_measures)
        if len(proper_measure_id) != 1:
            raise DataAbnormalError(
                f"Exposure data have {len(proper_measure_id)} measure id(s). Data should have"
                f"exactly one id out of {allowable_measures} but came back with {proper_measure_id}."
            )
        else:
            data = data[data.measure_id == proper_measure_id.pop()]

    else:  # alternative_risk_factor
        data = gbd.get_auxiliary_data("exposure", entity.kind, entity.name, location_id)

    return data


def extract_exposure_standard_deviation(
    entity: Union[RiskFactor, AlternativeRiskFactor],
    location_id: int,
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    if entity.kind == "risk_factor" and entity.name in OTHER_MEID:
        data = gbd.get_modelable_entity_draws(
            OTHER_MEID[entity.name], location_id, year_id=year_id
        )
    elif entity.kind == "risk_factor":
        data = gbd.get_exposure_standard_deviation(
            entity.gbd_id, location_id, year_id=year_id
        )
    else:  # alternative_risk_factor
        data = gbd.get_auxiliary_data(
            "exposure_standard_deviation", entity.kind, entity.name, location_id
        )
    return data


def extract_exposure_distribution_weights(
    entity: Union[RiskFactor, AlternativeRiskFactor],
    location_id: int,
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    data = gbd.get_auxiliary_data(
        "exposure_distribution_weights", entity.kind, entity.name, location_id
    )
    return data


def extract_relative_risk(
    entity: RiskFactor,
    location_id: int,
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    data = gbd.get_relative_risk(entity.gbd_id, location_id, year_id=year_id)
    # TODO: [MIC-4891] Process new relative risk data format properly
    if not data["exposure"].isna().all():
        raise DataAbnormalError(
            "Relative risk data in new format with 1000 exposure values. Our processing is not "
            "currently able to process data in this format."
        )
    data = filter_to_most_detailed_causes(data)
    if entity.gbd_id == 136:  # non-exclusive breastfeeding
        data = data.loc[data["age_group_id"].isin([3, 388])]
    elif entity.gbd_id == 137:  # discontinued breastfeeding
        data = data.loc[data["age_group_id"].isin([238, 389])]
    return data


def extract_population_attributable_fraction(
    entity: Union[RiskFactor, Etiology],
    location_id: int,
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    data = gbd.get_paf(entity.gbd_id, location_id, year_id=year_id)
    data = data[data.metric_id == METRICS["Percent"]]
    data = data[data.measure_id.isin([MEASURES["YLDs"], MEASURES["YLLs"]])]
    data = filter_to_most_detailed_causes(data)
    # clip PAFs between 0 and 1 (data outside these bounds is expected from GBD)
    draw_cols = [col for col in data.columns if col.startswith("draw_")]
    data.loc[:, draw_cols] = data[draw_cols].clip(lower=0)
    data.loc[:, draw_cols] = data[draw_cols].clip(upper=1)
    return data


def extract_mediation_factors(
    entity: RiskFactor,
    location_id: int,
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    data = gbd.get_auxiliary_data("mediation_factor", entity.kind, entity.name, location_id)
    return data


def extract_estimate(
    entity: Covariate,
    location_id: int,
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    data = gbd.get_covariate_estimate(int(entity.gbd_id), location_id, year_id=year_id)
    return data


def extract_utilization_rate(entity: HealthcareEntity, location_id: int) -> pd.DataFrame:
    data = gbd.get_modelable_entity_draws(entity.gbd_id, location_id)
    return data


def extract_structure(
    entity: Population,
    location_id: int,
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    data = gbd.get_population(location_id, year_id=year_id)
    return data


def extract_theoretical_minimum_risk_life_expectancy(
    entity: Population,
    location_id: int,
    year_id: Optional[Union[int, str, List[int]]] = None,
) -> pd.DataFrame:
    data = gbd.get_theoretical_minimum_risk_life_expectancy()
    return data
