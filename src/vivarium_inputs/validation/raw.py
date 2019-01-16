from typing import List
import warnings

import pandas as pd
import numpy as np

from vivarium_inputs.globals import (DRAW_COLUMNS, DEMOGRAPHIC_COLUMNS, METRICS, MEASURES,
                                     DataAbnormalError, InvalidQueryError, DataNotExistError, gbd)

MAX_INCIDENCE = 8   # ceiling-ed max incidence for diarrheal diseases among all locations
MAX_REMISSION = 80  # ceiling-ed max remission for diarrheal diseases among all locations


def check_metadata(entity, measure):
    metadata_checkers = {
        'sequela': _check_sequela_metadata,
        'cause': _check_cause_metadata,
        'risk_factor': _check_risk_factor_metadata,
        'etiology': _check_etiology_metadata,
        'covariate': _check_covariate_metadata,
        'coverage_gap': _check_coverage_gap_metadata,
        'health_technology': _check_health_technology_metadata,
        'healthcare_entity': _check_healthcare_entity_metadata,
        'population': _check_population_metadata,
    }

    metadata_checkers[entity.kind](entity, measure)


def validate_raw_data(data, entity, measure, location_id):
    validators = {
        # Cause-like measures
        'incidence': _validate_incidence,
        'prevalence': _validate_prevalence,
        'birth_prevalence': _validate_birth_prevalence,
        'disability_weight': _validate_disability_weight,
        'remission': _validate_remission,
        'deaths': _validate_deaths,
        # Risk-like measures
        'exposure': _validate_exposure,
        'exposure_standard_deviation': _validate_exposure_standard_deviation,
        'exposure_distribution_weights': _validate_exposure_distribution_weights,
        'relative_risk': _validate_relative_risk,
        'population_attributable_fraction': _validate_population_attributable_fraction,
        'mediation_factors': _validate_mediation_factors,
        # Covariate measures
        'estimate': _validate_estimate,
        # Health system measures
        'cost': _validate_cost,
        'utilization': _validate_utilization,
        # Population measures
        'structure': _validate_structure,
        'theoretical_minimum_risk_life_expectancy': _validate_theoretical_minimum_risk_life_expectancy,
    }

    if measure not in validators:
        raise NotImplementedError()

    validators[measure](data, entity, location_id)


def _check_sequela_metadata(entity, measure):
    if measure in ['incidence', 'prevalence', 'birth_prevalence']:
        exists = entity[f'{measure}_exists']
        if not exists:
            warnings.warn(f'{measure.capitalize()} data for sequela {entity.name} may not exist for all locations.')
        if exists and not entity[f'{measure}_in_range']:
            warnings.warn(f'{measure.capitalize()} for sequela {entity.name} may be outside the normal range.')
    else:  # measure == 'disability_weight
        if not entity.healthstate[f'{measure}_exists']:  # this is not location specific so we can actually throw error
            raise InvalidQueryError(f'Sequela {entity.name} does not have {measure} data.')


def _check_cause_metadata(entity, measure):
    exists = entity[f'{measure}_exists']
    if not exists:
        warnings.warn(f'{measure.capitalize()} data for cause {entity.name} may not exist for all locations.')
    if exists and not entity[f'{measure}_in_range']:
        warnings.warn(f"{measure.capitalize()} for cause {entity.name} may be outside the normal range.")

    violated_restrictions = [r.replace('_', ' ') for r in entity.restrictions.violated if measure in r]
    if violated_restrictions:
        warnings.warn(f'Cause {entity.name} {measure} data may violate the following restrictions: '
                      f'{", ".join(violated_restrictions)}.')

    consistent = entity[f"{measure}_consistent"]
    children = "subcauses" if measure == "deaths" else "sequela"

    if consistent is not None and not consistent:
        warnings.warn(f"{measure.capitalize()} data for cause {entity.name} may not exist for {children} in all "
                      f"locations. {children.capitalize()} models may not be consistent with models for this cause.")

    if consistent and not entity[f"{measure}_aggregates"]:
        warnings.warn(f"{children.capitalize()} {measure} data for cause {entity.name} may not correctly "
                      f"aggregate up to the cause level in all locations. {children.capitalize()} models may not "
                      f"be consistent with models for this cause.")


def _check_risk_factor_metadata(entity, measure):
    mapping_names = {'relative_risk': 'rr',
                     'population_attributable_fraction': 'paf',
                     'exposure': 'exposure',
                     'exposure_standard_deviation': 'exposure_sd'}

    if measure == 'population_attributable_fraction':
        paf_types = np.array(['yll', 'yld'])
        missing_pafs = paf_types[not entity.paf_yll_exists, not entity.paf_yld_exists]
        if missing_pafs:
            warnings.warn(f'{measure.captitalize()} data for {", ".join(missing_pafs)} for risk factor {entity.name}'
                          f'may not exist for all locations.')

        abnormal_range = paf_types[not entity.paf_yll_in_range, not entity.paf_yld_in_range]
        if not missing_pafs and abnormal_range:
            warnings.warn(f'{measure.capitalize()} data for {", ".join(abnormal_range)} for risk factor {entity.name} '
                          f'may be outside expected range [0, 1].')
    else:
        exists = entity[f'{mapping_names[measure]}_exists']
        if not exists:
            warnings.warn(f'{measure.capitalize()} data for risk factor {entity.name} may not exist for all locations.')

        if measure == 'relative_risk' and exists and not entity.rr_in_range:
            warnings.warn(f'{measure.capitalize()} data for risk factor {entity.name} may be outside '
                          f'expected range >1.')

        if measure == 'exposure' and exists and entity.exposure_year_type in ('mix', 'incomplete'):
            warnings.warn(f'{measure.capitalize()} data for risk factor {entity.name} may contain unexpected '
                          f'or missing years.')

    violated_restrictions = [r.replace('_', ' ') for r in entity.restrictions.violated if mapping_names[measure] in r]
    if violated_restrictions:
        warnings.warn(f'Risk factor {entity.name} {measure} data may violate the following restrictions: '
                      f'{", ".join(violated_restrictions)}.')


def _check_etiology_metadata(entity, measure):
    mapping_measure = 'paf_yld'
    if not entity[f'{mapping_measure}_exists']:
        warnings.warn(f'{measure.capitalize()} data for etiology {entity.name} may not exist for all locations.')
    if not entity[f'{mapping_measure}_in_range']:
        warnings.warn(f'{measure.capitalize()} for etiology {entity.name} may be abnormal.')


def _check_covariate_metadata(entity, measure):
    if not entity.mean_value_exists:
        warnings.warn(f'{measure.capitalize()} data for covariate {entity.name} may not contain'
                      f'mean values for all locations.')

    if not entity.uncertainty_exists:
        warnings.warn(f'{measure.capitalize()} data for covariate {entity.name} may not contain '
                      f'uncertainty values for all locations.')

    restrictions = [entity[f'by_{r}_violated'] for r in ['sex', 'age']]
    violated_restrictions = [r.replace('_violated', '').replace('_', ' ') for r in restrictions if r]
    if violated_restrictions:
        warnings.warn(f'Covariate {entity.name} may violate the following '
                      f'restrictions: {", ".join(violated_restrictions)}.')


def _check_coverage_gap_metadata(entity, measure):
    pass


def _check_health_technology_metadata(entity, measure):
    raise NotImplementedError()


def _check_healthcare_entity_metadata(entity, measure):
    raise NotImplementedError()


def _check_population_metadata(entity, measure):
    pass


def _validate_incidence(data, entity, location_id):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ('measure_id', 'metric_id', f'{entity.kind}_id') + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data.measure_id.unique(), 'incidence')
    check_metric_id(data.metric_id.unique(), 'rate')
    check_draw_columns_range(data, 0, MAX_INCIDENCE, warn=True)
    check_years(data, 'annual')
    check_location(data, location_id)
    check_all_ages_present(data)
    check_all_sexes_present(data)

    if entity.kind == 'cause':
        check_age_restrictions(data, entity.restrictions.yld_age_group_id_start, entity.restrictions.yld_age_group_id_end)
        check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only)


def _validate_prevalence(data, entity, location_id):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ('measure_id', 'metric_id', f'{entity.kind}_id') + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data.measure_id.unique(), 'prevalence')
    check_metric_id(data.metric_id.unique(), 'rate')
    check_draw_columns_range(data, 0, 1)
    check_years(data, 'annual')
    check_location(data, location_id)
    check_all_ages_present(data)
    check_all_sexes_present(data)

    if entity.kind == 'cause':
        check_age_restrictions(data, entity.restrictions.yld_age_group_id_start, entity.restrictions.yld_age_group_id_end)
        check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only)


def _validate_birth_prevalence(data, entity, location_id):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ('measure_id', 'metric_id', f'{entity.kind}_id') + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data.measure_id.unique(), 'incidence')
    check_metric_id(data.metric_id.unique(), 'rate')
    check_draw_columns_range(data, 0, 1)
    check_years(data, 'annual')
    check_location(data, location_id)
    check_all_sexes_present(data)

    birth_age_group_id = 164
    if data.age_group_id.unique() != birth_age_group_id:
        raise DataAbnormalError(f'Birth prevalence data for {entity.kind} {entity.name} includes age groups beyond '
                                f'the expected birth age group (id {birth_age_group_id}.')

    if entity.kind == 'cause':
        check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only)


def _validate_disability_weight(data, entity, location_id):
    # TODO
    expected_columns = ('location_id', 'age_group_id', 'sex_id', 'measure',
                        'healthstate', 'healthstate_id') + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)
    check_location(data, location_id)


def _validate_remission(data, entity, location_id):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ('measure_id', 'metric_id', 'model_version_id',
                        'modelable_entity_id') + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data.measure_id.unique(), 'remission')
    check_metric_id(data.metric_id.unique(), 'rate')
    check_draw_columns_range(data, 0, MAX_REMISSION, warn=True)
    check_years(data, 'binned')
    check_location(data, location_id)
    check_age_restrictions(data, entity.restrictions.yld_age_group_id_start, entity.restrictions.yld_age_group_id_end)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only)


def _validate_deaths(data, entity, location_id):
    check_data_exist(data, zeros_missing=True)

    expected_columns = ('measure_id', f'{entity.kind}_id', 'metric_id') + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    check_measure_id(data.measure_id.unique(), 'deaths')
    check_metric_id(data.metric_id.unique(), 'number')
    # check_draw_columns_range(data, 0, population) FIXME: what should be the upper bound for deaths?
    check_years(data, 'annual')
    check_location(data, location_id)
    check_age_restrictions(data, entity.restrictions.yll_age_group_id_start, entity.restrictions.yll_age_group_id_end)
    check_sex_restrictions(data, entity.restrictions.male_only, entity.restrictions.female_only)


def _validate_exposure(data, entity, location_id):
    expected_columns = ('rei_id', 'modelable_entity_id', 'parameter',
                        'measure_id', 'metric_id') + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)

    if len(set(data.measure_id)) > 1:
        raise DataAbnormalError(f'{entity.kind.capitalize()} {entity.name} has '
                                f'multiple measure ids: {set(data.measure_id)}.')

    if not set(data.measure_id).issubset({MEASURES['Prevalence'], MEASURES['Proportion']}):
        raise DataAbnormalError(f'{entity.kind.capitalize()} {entity.name} contains '
                                f'an invalid measure id {set(data.measure_id)}.')

    # TODO: do the exposure year types vary by loc? Can I throw an error for
    #  mix/incomplete above so I don't have to handle it here?
    check_years(data, entity.exposure_year_type)
    check_location(data, location_id)


def _validate_exposure_standard_deviation(data, entity, location_id):
    del entity  # unused
    expected_columns = ('rei_id', 'modelable_entity_id', 'measure_id',
                        'metric_id') + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')
    check_location(data, location_id)


def _validate_exposure_distribution_weights(data, entity, location_id):
    del entity  # unused
    expected_columns = ['location_id', 'sex_id', 'age_group_id', 'measure', 'risk_id',
                        'betasr', 'exp', 'gamma', 'glnorm', 'gumbel', 'invgamma', 'llogis',
                        'invweibull', 'lnorm', 'mgamma', 'mgumbel', 'norm', 'weibull']
    check_columns(expected_columns, data.columns)
    check_location(data, location_id)


def _validate_relative_risk(data, entity, location_id):
    del entity  # unused
    expected_columns = ('rei_id', 'modelable_entity_id', 'cause_id', 'mortality',
                        'morbidity', 'metric_id', 'parameter') + DEMOGRAPHIC_COLUMNS + DRAW_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'binned')
    check_location(data, location_id)


def _validate_population_attributable_fraction(data, entity, location_id):
    del entity  # unused
    expected_columns = ('metric_id', 'measure_id', 'rei_id', 'cause_id') + DRAW_COLUMNS + DEMOGRAPHIC_COLUMNS
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')
    check_location(data, location_id)


def _validate_mediation_factors(data, entity, location_id):
    raise NotImplementedError()


def _validate_estimate(data, entity, location_id):
    del entity  # unused
    expected_columns = ['model_version_id', 'covariate_id', 'covariate_name_short', 'location_id',
                        'location_name', 'year_id', 'age_group_id', 'age_group_name', 'sex_id',
                        'sex', 'mean_value', 'lower_value', 'upper_value']
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')
    check_location(data, location_id)


def _validate_cost(data, entity, location_id):
    raise NotImplementedError()


def _validate_utilization(data, entity, location_id):
    raise NotImplementedError()


def _validate_structure(data, entity, location_id):
    expected_columns = ['age_group_id', 'location_id', 'year_id', 'sex_id', 'population', 'run_id']
    check_columns(expected_columns, data.columns)
    check_years(data, 'annual')
    check_location(data, location_id)


def _validate_theoretical_minimum_risk_life_expectancy(data, entity, location_id):
    pass


def check_years(df: pd.DataFrame, year_type: str):
    years = {'annual': list(range(1990, 2018)), 'binned': gbd.get_estimation_years()}
    expected_years = years[year_type]
    if set(df.year_id.unique()) < set(expected_years):
        raise DataAbnormalError(f'Data has missing years: {set(expected_years).difference(set(df.year_id.unique()))}.')
    # if is it annual, we expect to have extra years from some cases like codcorrect/covariate
    if year_type == 'binned' and set(df.year_id.unique()) > set(expected_years):
        raise DataAbnormalError(f'Data has extra years: {set(df.year_id.unique()).difference(set(expected_years))}.')


def check_location(data: pd.DataFrame, location_id: str):
    if data.empty:
        raise InvalidQueryError(f'Data does not have location_id {location_id}.')
    if len(data['location_id'].unique()) > 1:
        raise DataAbnormalError(f'Data has extra location ids.')
    data_location_id = data['location_id'].unique()[0]
    global_loc_id = 1
    if data_location_id not in [global_loc_id, location_id]:
        raise DataAbnormalError(f'Data called for {location_id} has a location id {data_location_id}.')


def check_columns(expected_cols: List, existing_cols: List):
    if set(existing_cols) < set(expected_cols):
        raise DataAbnormalError(f'Data is missing columns: {set(expected_cols).difference(set(existing_cols))}.')
    elif set(existing_cols) > set(expected_cols):
        raise DataAbnormalError(f'Data returned extra columns: {set(existing_cols).difference(set(expected_cols))}.')


def check_data_exist(data: pd.DataFrame, zeros_missing: bool = True):
    """

    Parameters
    ----------
    data
        Dataframe contain DRAW_COLUMNS.
    zeros_missing
        Boolean indicating whether to treat all zeros in DRAW_COLUMNS as
        missing or not.

    Raises
    -------
    DataNotExistError
        If data is empty or contains all NaN values in DRAW_COLUMNS, or
        contains all zeros in DRAW_COLUMNS and zeros_missing is True.

    """
    if data.empty:
        raise DataNotExistError('Data is empty.')
    if data.dropna().empty:
        raise DataNotExistError('Data contains no non-missing values.')
    if zeros_missing and not np.all(data[DRAW_COLUMNS]):
        raise DataNotExistError('Data contains no non-zero values.')


def check_all_ages_present(data: pd.DataFrame):
    expected_ages = set(gbd.get_age_group_id())
    if set(data.age_group_id) != expected_ages:
        raise DataAbnormalError('Data does not contain the full set of GBD age groups as expected.')


def check_all_sexes_present(data: pd.DataFrame):
    expected_sexes = {gbd.MALE, gbd.FEMALE}
    if set(data.sex_id) != expected_sexes:
        raise DataAbnormalError('Data does not contain the full set of GBD sex ids as expected.')


def check_age_restrictions(data: pd.DataFrame, age_group_id_start: int, age_group_id_end: int):
    """Check that all expected age groups between age_group_id_start and
    age_group_id_end, inclusive, and only those age groups, appear in data.

    Parameters
    ----------
    data
        Dataframe containing an age_group_id column.
    age_group_id_start
        Lower boundary of age group ids expected in data, inclusive.
    age_group_id_end
        Upper boundary of age group ids expected in data, exclusive.

    Raises
    ------
    DataAbnormalError
        If any age group ids in the range
        [`age_group_id_start`, `age_group_id_end`] don't appear in the data or
        if any additional age group ids (with the exception of 235) appear in
        the data.

    """
    data = data.copy().dropna()

    gbd_age_ids = gbd.get_age_group_id()
    start_index = gbd_age_ids.index(age_group_id_start)
    end_index = gbd_age_ids.index(age_group_id_end)

    expected_gbd_age_ids = gbd_age_ids[start_index:end_index+1]

    # age groups we expected in data but that are not
    missing_age_groups = set(expected_gbd_age_ids).difference(set(data.age_group_id))
    # age groups we do not expect in data but that are (allow 235 because of metadata oversight)
    extra_age_groups = set(data.age_group_id).difference(set(expected_gbd_age_ids)) - {235}

    if missing_age_groups:
        raise DataAbnormalError(f'Data was expected to contain all age groups between ids '
                                f'{age_group_id_start} and {age_group_id_end}, '
                                f'but was missing the following: {missing_age_groups}.')
    if extra_age_groups:
        # we treat all 0s as missing in accordance with gbd so if extra age groups have all 0 data, that's fine
        should_be_zero = data[data.age_group_id.isin(extra_age_groups)]
        if not np.all(should_be_zero[DRAW_COLUMNS]):
            raise DataAbnormalError(f'Data was only expected to contain age groups between ids '
                                    f'{age_group_id_start} and {age_group_id_end} (with the possible addition of 235), '
                                    f'but also included {extra_age_groups}.')


def check_draw_columns_range(data: pd.DataFrame, min_value: float, max_value: float, warn: bool = False):
    """Check that all values in data in draw columns fall between min_value
    and max_value, inclusive.

    Parameters
    ----------
    data
        Dataframe containing DRAW_COLUMNS.
    min_value
        Lower boundary of expected values, inclusive.
    max_value
        Upper boundary of expected values, inclusive.
    warn
        Boolean indicating whether values out of range should trigger
        a warning (True) or an error (False).

    Raises
    ------
    DataAbnormalError
        If any values in DRAW_COLUMNS in data fall outside the range
        [`min_value`, `max_value`].

    """
    msg = f'Data contains values outside the expected range [{min_value}, {max_value}].'
    if not np.all(data[DRAW_COLUMNS] >= min_value) and not np.all(data[DRAW_COLUMNS] <= max_value):
        if warn:
            warnings.warn(msg)
        else:
            raise DataAbnormalError(msg)


def check_sex_restrictions(data: pd.DataFrame, male_only: bool, female_only: bool):
    """Check that data only contains sex ids expected from GBD and that any
    sex restrictions match data.

    Parameters
    ----------
    data
        Dataframe contained sex_id column.
    male_only
        Boolean indicating whether data is restricted to male only estimates.
    female_only
        Boolean indicating whether data is restricted to female only estimates.

    Raises
    -------
    DataAbnormalError
        If data contains any sex ids not in the set defined by GBD or data
        violates passed sex restrictions.
    """
    data = data.copy().dropna()

    gbd_sex_ids = {gbd.MALE, gbd.FEMALE, gbd.COMBINED}
    if not set(data.sex_id).issubset(gbd_sex_ids):
        raise DataAbnormalError(f'Data contains unexpected sex ids: {set(data.sex_id).difference(gbd_sex_ids)}')

    if male_only:
        if not np.all(data[data.sex_id == gbd.MALE]):
            raise DataAbnormalError('Data is restricted to male only, but contains all zero values for males.')

        if set(data.sex_id) != {gbd.MALE} and np.all(data[data.sex_id != gbd.MALE][DRAW_COLUMNS]):
            raise DataAbnormalError('Data is restricted to male only, but contains '
                                    'non-male sex ids for which data values are not all 0.')

    if female_only:
        if np.all(data[data.sex_id == gbd.FEMALE]):
            raise DataAbnormalError('Data is restricted to female only, but contains all zero values for females.')

        if set(data.sex_id) != {gbd.FEMALE} and np.all(data[data.sex_id != gbd.FEMALE][DRAW_COLUMNS]):
            raise DataAbnormalError('Data is restricted to female only, but contains '
                                    'non-female sex ids for which data values are not all 0.')

    if (not male_only and not female_only and
            (not {3}.issubset(set(data.sex_id)) or not np.all(data[data.sex_id == gbd.COMBINED][DRAW_COLUMNS])
             or (not {1, 2}.issubset(set(data.sex_id)) or (not np.all(data[data.sex_id == gbd.MALE][DRAW_COLUMNS])
                 or not np.all(data[data.sex_id == gbd.FEMALE]))))):
        raise DataAbnormalError('Data has no sex restrictions, but does not contain non-zero '
                                'values for both males and females.')


def check_measure_id(data_measure_id, expected_measure):
    if data_measure_id != MEASURES[expected_measure.capitalize()]:
        raise DataAbnormalError(f'Data includes measures beyond the expected {expected_measure} '
                                f'(measure id {MEASURES[expected_measure.capitalize()]}).')


def check_metric_id(data_metric_id, expected_metric):
    if data_metric_id != METRICS[expected_metric.capitalize()]:
        raise DataAbnormalError(f'Data includes metrics beyond the expected {expected_metric.lower()} '
                                f'(metric_id {METRICS[expected_metric.capitalize()]}')

