"""Microsim functions
This notebook contains the functions that will be used to
re-format GBD data into a format that can be used for the cost-effectiveness
microsim. Wherever possible, these functions will leverage the existing
central comp functions (please see this link for more information on the
central computation functions
https://hub.ihme.washington.edu/display/GBD2016/Shared+functions
"""
import inspect
from itertools import product

import numpy as np
import pandas as pd

from vivarium import config
from vivarium.framework.util import rate_to_probability

from ceam_inputs import gbd
from ceam_inputs.gbd_mapping import cid, meid, hid
from ceam_inputs.gbd_ms_auxiliary_functions import (normalize_for_simulation,
                                                    expand_ages_for_dfs_w_all_age_estimates, expand_ages,
                                                    get_age_group_midpoint_from_age_group_id)
import logging

_log = logging.getLogger(__name__)


class UnhandledRiskError(ValueError):
    pass


class GBDError(Exception):
    """Base exception for errors related to GBD data."""
    pass


class UnmodelledDataError(GBDError):
    """Raised when the requested data is not modeled by the GBD"""
    pass


def get_gbd_draws(location_id, year_start, year_end, measure, gbd_id, gbd_round_id):
    """Returns draws for a given measure and modelable entity

    Gives you incidence, prevalence, csmr, excess mortality, and
    other metrics at draw level.

    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    year_end : int, end year
        year_end is the year in which you want to end the simulation

    measure : int, measure
        defines which measure (e.g. prevalence) you want to pull. Use central
        comp's get_ids functions to learn about which measures are available
        and what numbers correspond with each measure

    gbd_id: int, gbd entity id
        gbd_id takes same id values as are used for GBD

    Returns
    -------
    df with year_id, sex_id, age and 1k draws

    Notes
    -----
    Used by -- get_cause_level_prevalence, sum_up_csmrs_for_all_causes_in_microsim, get_post_mi_heart_failure_proportion_draws, get_excess_mortality, get_incidence, get_continuous, get_proportion, get_prevalence

    Assumptions -- None

    Questions -- None

    Unit test in place? -- No. Don't think it's necessary, since this function merely pulls draws from the database and then filters a dataframe so that only one measure is included in the output and that only the years in b/w the simulation year start and year end are included in the df.
    """
    if isinstance(gbd_id, cid):
        draws = gbd.get_como_draws(location_id, gbd_id, gbd_round_id, config.input_data.gbd_publication_ids)
    else:
        draws = gbd.get_modelable_entity_draws(location_id, gbd_id, gbd_round_id, config.input_data.gbd_publication_ids)
    draws = draws[draws.measure_id == measure]
    draws = draws.query('year_id>={ys} and year_id<={ye}'.format(ys=year_start, ye=year_end))
    draws = get_age_group_midpoint_from_age_group_id(draws)

    keepcol = ['year_id', 'sex_id', 'age']
    keepcol.extend(('draw_{i}'.format(i=i) for i in range(0, 1000)))

    # assert an error to make sure data is dense (i.e. no missing data)
    assert draws.isnull().values.any() == False, "there are nulls in the dataframe that get_gbd_draws just tried to output. check that the cache to make sure the data you're pulling is correct"

    # assert an error if there are duplicate rows
    assert draws.duplicated(['age', 'year_id', 'sex_id']).sum(
    ) == 0, "there are duplicates in the dataframe that get_gbd_draws just tried to output. check the cache to make sure that the data you're pulling is correct"

    return draws[keepcol].sort_values(by=['year_id', 'age', 'sex_id'])


def get_populations(location_id, year=-1, sex='All', gbd_round_id=3):
    """Get demographic population structure.

    Parameters
    ----------
    location_id : int
        The GBD location id of the region of interest.
    year : int, optional
        If specified, return only the selected year's populations. Otherwise return all years.
    sex: str in ['Male', 'Female', 'Both', 'All'], optional
        If specified, return only the selected sex's populations.  Otherwise return Male, Female,
        and the combined category Both.
    gbd_round_id: int
        The GBD round to pull data for.  Defaults to GBD 2015.

    Returns
    -------
    `DataFrame` :
        DataFrame with columns ['age', 'year', 'sex', 'location_id', 'pop_scaled']
        where 'pop_scaled' is the total population of the demographic subgroup defined by
        the first four columns.
    """
    pop = gbd.get_populations(location_id=location_id, gbd_round_id=gbd_round_id)

    if year != -1:
        pop = pop[pop.year_id == year]
    if sex != 'All':
        sex_map = {'Male': 1, 'Female': 2, 'Both': 3}
        if sex not in sex_map:
            raise ValueError("Sex must be one of {} or 'All'".format(list(sex_map.keys())))
        pop = pop[pop.sex_id == sex_map[sex]]
    pop = get_age_group_midpoint_from_age_group_id(pop)

    # The population column was called pop_scaled in GBD 2015, but name was changed.
    # Changing it back since all of our code uses pop_scaled as the col name
    pop = pop.rename(columns={'population': 'pop_scaled'})
    pop = pop[['year_id', 'location_id', 'age', 'age_group_start', 'age_group_end', 'sex_id', 'pop_scaled']]
    validate_data(pop, ['age', 'year_id', 'sex_id'])
    return normalize_for_simulation(pop)


def get_cause_specific_mortality(location_id, year_start, year_end, cause_id, gbd_round_id, draw_number):
    """
    location_id : int
        location_id takes same location_id values as are used for GBD
    year_start : int, year
        year_start is the year in which you want to start the simulation
    year_end : int, end year
        year_end is the year in which you want to end the simulation
    cause_id: int
        cause_id takes same cause_id values as are used for GBD
    gbd_round_id: int
        GBD round of interest
    draw_number: int
        GBD draw of interest
    """
    assert isinstance(cause_id, cid)
    draws = gbd.get_codcorrect_draws(location_id, cause_id, gbd_round_id, config.input_data.gbd_publication_ids)
    draws = draws[(draws.year_id >= year_start) & (draws.year_id <= year_end)]
    draws = get_age_group_midpoint_from_age_group_id(draws)
    keep_columns = ['year_id', 'sex_id', 'age'] + ['draw_{}'.format(i) for i in range(1000)]
    draws = draws[keep_columns]
    validate_data(draws, ['year_id', 'sex_id', 'age'])
    cause_specific_deaths = select_draw_data(draws, draw_number, 'deaths')

    pop = gbd.get_populations(location_id, gbd_round_id)
    pop = pop[pop.sex_id != 3]
    pop = get_age_group_midpoint_from_age_group_id(pop)
    pop = pop[['year_id', 'age', 'sex_id', 'population']]
    pop = pop.rename(columns={'population': 'pop_scaled'})
    validate_data(pop, ['year_id', 'sex_id', 'age'])
    pop = normalize_for_simulation(pop)

    csmr = cause_specific_deaths.merge(pop, on=['age', 'sex', 'year'])
    csmr['rate'] = np.divide(csmr.deaths.values, csmr.pop_scaled.values)

    return csmr[['age', 'sex', 'year', 'rate']]


def get_post_mi_heart_failure_proportion_draws(location_id, year_start, year_end, draw_number, gbd_round_id):
    """Returns post-mi proportion draws for hf due to ihd

    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD
    year_start : int
        year_start is the year in which you want to start the simulation
    year_end : int
        year_end is the year in which you want to end the
    draw_number : int
        GBD draw to pull data for

    Returns
    -------
    df with year_id, sex_id, age and 1k draws

    Notes
    -----
    Assumptions -- That the proportional prevalence is a good enough estimation of the proportional incidence.
    Questions -- More of a general python question -- should I be using np.multiply for
    multiplication? Maybe it has to do with python's floating point issues, but I was
    getting different results when using A*B instead of np.multiply(A,B).
    """
    # TODO: NEED TO WRITE TESTS TO MAKE SURE THAT POST_MI TRANSITIONS SCALE TO 1
    hf_envelope = get_gbd_draws(location_id, year_start, year_end,
                                             measure=6, gbd_id=meid(2412), gbd_round_id=gbd_round_id)
    proportion_draws = get_gbd_draws(location_id, year_start, year_end,
                                                  measure=18, gbd_id=meid(2414), gbd_round_id=gbd_round_id)

    cause_of_hf = pd.merge(hf_envelope, proportion_draws,
                           on=['age', 'year_id', 'sex_id'], suffixes=('_env', '_prop'))

    envelope = cause_of_hf[['draw_{}_env'.format(i) for i in range(1000)]].values
    proportion = cause_of_hf[['draw_{}_prop'.format(i) for i in range(1000)]].values
    cause_of_hf.set_index(['year_id', 'sex_id', 'age'], inplace=True)

    # TODO: Manual calculation of the multiplication below gave a little bit different values.
    # Should I be using np.multiply or something else to make sure python is handling these floats correctly?
    # TODO: Ensure rate_to_probability is calculating annual rates
    output_df = pd.DataFrame(rate_to_probability(np.multiply(envelope, proportion)),
                             columns=['draw_{}'.format(i) for i in range(1000)], index=cause_of_hf.index)
    output_df = output_df.reset_index()

    keep_columns = ['year_id', 'sex_id', 'age']
    keep_columns.extend(('draw_{i}'.format(i=i) for i in range(0, 1000)))
    output_df = output_df[keep_columns]

    validate_data(output_df, ['age', 'year_id', 'sex_id'])
    assert output_df['draw_{}'.format(draw_number)].all() <= 1, ("Something went wrong with the "
                                                                 "get_post_mi_heart_failure_proportion_draws "
                                                                 "calculation. incidence rate can't be GT 1. "
                                                                 "Check to see if the numerator/denominator "
                                                                 "were flipped")
    return output_df


def get_relative_risks(location_id, risk_id, cause_id, gbd_round_id, rr_type='morbidity'):
    """
    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD
    risk_id: rid
        risk_id takes same risk_id values as are used for GBD
    cause_id: cid
        cause_id takes same cause_id values as are used for GBD
    gbd_round_id: int
       GBD round to pull data for
    rr_type: str
        can specify morbidity if you want RRs for incidence or mortality if you want RRs for mortality

    Returns
    -------
    df with columns year_id, sex_id, age, 1k relative risk draws

    Notes
    -----
    Assumptions -- Some risks in GBD (e.g. Zinc deficiency and high sbp) don't
    have estimates for all ages. I have set up the code so that each age group for which we
    don't have GBD estimates has an RR of 1 (i.e. no elevated risk).
    Questions -- Should we set the RR to 1 for age groups for which we do not have rr estimates?
    """
    rr = gbd.get_relative_risks(location_id=location_id, risk_id=risk_id, gbd_round_id=gbd_round_id)

    draw_columns = ['rr_{}'.format(i) for i in range(1000)]
    key_columns = ['year_id', 'sex_id', 'age', 'parameter']

    correct_measurement = (rr[rr_type] == 1) & (rr['cause_id'] == cause_id)
    rr = rr[correct_measurement]

    if rr.empty:
        err_msg = "No data for risk_id {} on cause_id {} for type {}".format(risk_id, cause_id, rr_type)
        raise UnmodelledDataError(err_msg)

    rr = get_age_group_midpoint_from_age_group_id(rr)
    rr = expand_ages(rr)
    # We frequently have N/As in the data where the risk wasn't modelled.  We assume there is no
    # exposure or that the exposure does not increase the relative risk on the cause.
    rr[draw_columns] = rr[draw_columns].fillna(value=1)

    rr = rr[key_columns + draw_columns]
    validate_data(rr, key_columns)

    return rr


def get_pafs(location_id, risk_id, cause_id, gbd_round_id, paf_type='morbidity'):
    """
    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD
    year_start : int, year
        year_start is the year in which you want to start the simulation
    year_end : int, end year
        year_end is the year in which you want to end the simulation
    risk_id: int, risk id
        risk_id takes same risk_id values as are used for GBD
    cause_id: int, cause id
        cause_id takes same cause_id values as are used for GBD
    gbd_round_id: int
        GBD round to pull data for
    draw_number: int
        GBD draw to pull data for
    paf_type: str
        specify whether you want morbidity (YLDs) or mortality (Deaths) PAFs

    Returns
    -------
        df with columns year_id, sex_id, age, val, upper, and lower

    Notes
    -----
    Assumptions -- We should use PAFs for YLDs, since we use PAFs to affect incidence in CEAM.
    Some risks in GBD (e.g. Zinc deficiency and high sbp) don't have estimates for all ages.
    I have set up the code so that each age group for which we don't have GBD estimates has a PAF of 0
    Questions -- Should we set the PAF to 0 for age groups for which we do not have rr estimates?
    Need to submit an epi help ticket to determine whether we should use get_draws or transmogrifier.risk.risk_draws.
    """
    pafs = gbd.get_pafs(location_id=location_id, cause_id=cause_id, gbd_round_id=gbd_round_id)

    measure_id = {'morbidity': 3, 'mortality': 1}[paf_type]
    draw_columns = ['draw_{}'.format(i) for i in range(1000)]
    key_columns = ['year_id', 'sex_id', 'age']

    correct_measurement = (pafs.measure_id == measure_id) & (pafs.rei_id == risk_id) & (pafs.metric_id == 2)
    pafs = pafs[correct_measurement]

    pafs = get_age_group_midpoint_from_age_group_id(pafs)
    pafs = expand_ages(pafs)

    pafs[draw_columns] = pafs[draw_columns].fillna(value=0)

    pafs = pafs[key_columns + draw_columns]
    validate_data(pafs, key_columns)

    return pafs


def get_exposures(location_id, risk_id, gbd_round_id):
    """
    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    year_end : int, end year
        year_end is the year in which you want to end the simulation

    risk_id: int, risk id
        risk_id takes same risk_id values as are used for GBD

    gbd_round_id: int
        GBD round to pull

    Returns
    -------
    df with columns year_id, sex_id, age and 1k exposure draws

    Notes
    -----
    Assumptions -- Some risks in GBD (e.g. Zinc deficiency and high sbp) don't have estimates for all ages.
    I have set up the code so that each age group for which we don't have GBD estimates has an exposure of 0
    Questions -- Should we set the exposure to 0 for age groups for which we do not have rr estimates?
    Need to submit an epi help ticket to determine whether we should use get_draws or transmogrifier.risk.risk_draws.
    """

    exposure = gbd.get_exposures(location_id=location_id, risk_id=risk_id, gbd_round_id=gbd_round_id)

    draw_columns = ['draw_{}'.format(i) for i in range(1000)]
    key_columns = ['year_id', 'sex_id', 'age', 'parameter']

    exposure = get_age_group_midpoint_from_age_group_id(exposure)

    if risk_id == 100:
        exposure = _extract_secondhand_smoking_exposures(exposure)
    elif risk_id == 83:
        # Just fixing some poorly structured data.  cat10 and cat11 are zero, cat12 is the balance of the exposure
        # (indicating unexposed), but needs to be renamed cat10 so it corresponds to the rr data.  Yuck.
        good_rows = (exposure.parameter != 'cat10') & (exposure.parameter != 'cat11')
        exposure = exposure[good_rows]
        exposure.loc[exposure.parameter == 'cat12', 'parameter'] = 'cat10'
    # unsafe sanitation (rei_id 84) and underweight (rei_id 94) both have different modelable entity ids
    # for each different category. this is ok with us, so we don't want to generate the UnhandledRiskError
    elif risk_id in [84, 94, 136, 240, 241]:
        pass
    # TODO: Need to set age, year, sex index here again to make sure that we assign
    # the correct value to points outside of the range
    # TODO: Confirm that we want to be using cat1 here.
    # Cat1 seems really high for risk_id=238 (hand washing without soap) for Kenya
    # TODO: Do we want to set the exposure to 0 for the younger ages for which we don't have data?
    # It's an exceptionally strong assumption. We could use the exposure for the youngest age
    # for which we do have data, or do something else, if we wanted to. --EM 12/12
    else:
        list_of_meids = pd.unique(exposure.modelable_entity_id.values)
        # Some risks have a few nulls in the modelable_entity_id column.
        # This is ok, think it's just an artifact of how the risk is processed by central comp.
        list_of_meids = [x for x in list_of_meids if str(x) != 'nan']
        if len(list_of_meids) > 1:
            raise UnhandledRiskError("The risk -- rei_id {} --that you are trying to pull has multiple ".format(risk_id)
                                     + "modelable entity ids. are you sure you know how this risk is modeled? "
                                     + "If not, go talk to the modeler. After talking to the modeler, "
                                     + "you'll probably want to write some code to handle the risk, since "
                                     + "it's modeled differently than most risks. you can override this "
                                     + "error by adding a multiple_meids_override=True argument to "
                                     + "your get_exposures query after you determine how to incorporate "
                                     + "this risk into your simulation")

    # Hacky fix to an inconsistency in the database schema.  I'll do something nicer when I revise this
    # bit of ceam inputs - J.C. 2017-07-26
    if 'parameter' not in exposure.columns:
        exposure['parameter'] = 'continuous'

    exposure = expand_ages(exposure)

    exposure[draw_columns] = exposure[draw_columns].fillna(value=0)
    exposure = exposure[key_columns + draw_columns]
    validate_data(exposure, key_columns)

    return exposure


def _extract_secondhand_smoking_exposures(exposure):
    """
    there are 2 modelable entity ids for secondhand smoking, one for females and  males under age 15, and then one
    for males over age 15. Because of this, we  need to handle this risk separately from other risks.
    Apparently this should be fixed for the next round of GBD, but for now we need to deal with this annoying
    work around."""
    male_old_exposure = exposure.query("sex_id == 1 and modelable_entity_id == 2512 and age >= 17.5").copy()
    male_young_exposure = exposure.query("sex_id == 2 and modelable_entity_id == 9419 and age < 17.5").copy()
    male_young_exposure['sex_id'] = 1
    female_exposure = exposure.query("sex_id == 2 and modelable_entity_id == 9419").copy()
    exposure = male_young_exposure.append([male_old_exposure, female_exposure])

    residual_exposure = exposure.copy()
    residual_exposure['parameter'] = 'cat2'
    residual_exposure = residual_exposure.set_index([c for c in residual_exposure.columns if 'draw_' not in c])
    residual_exposure = 1 - residual_exposure
    residual_exposure = residual_exposure.reset_index()
    return exposure.append(residual_exposure)


def select_draw_data(data, draw, column_name, src_column=None):
    if column_name:
        if src_column is not None:
            if isinstance(src_column, str):
                column_map = {src_column.format(draw=draw): column_name}
            else:
                column_map = {src.format(draw=draw): dest for src, dest in zip(src_column, column_name)}
        else:
            column_map = {'draw_{draw}'.format(draw=draw): column_name}

        # if 'parameter' is in columns, then keep it, else do
        # not keep it (need parameter for the relative risk estimations)
        if 'parameter' in data.columns:
            keep_columns = ['year_id', 'age', 'sex_id', 'parameter'] + list(column_map.keys())
        else:
            keep_columns = ['year_id', 'age', 'sex_id'] + list(column_map.keys())

        data = data[keep_columns]
        data = data.rename(columns=column_map)

        return normalize_for_simulation(data)
    return data


def get_sbp_mean_sd(location_id, year_start, year_end):
    """ Returns a dataframe of mean and sd of sbp in LOG SPACE

    Parameters
    ----------
    location_id : int
    year_start : int
    year_end : int

    Returns
    -------
    df with mean and sd values in LOG space

    Notes
    -----
    Assumptions -- That people under age 25 have the TMRED SBP
    Questions -- We have estimates starting in the age 25-29 age group. Should we be using
    the midpoint or age 25 as the starting point?
    TODO: Might want to change the TMRED. Need to catch up with Stan regarding calculating
    TMREDs + write a function that will allow us to calculate TMREDs for a given risk.
    """

    draws = pd.DataFrame()
    for sex_id in [1, 2]:
        for year_id in np.arange(year_start, year_end + 1, 5):
            one_year_file = gbd.get_data_from_auxiliary_file('Systolic Blood Pressure Distributions',
                                                             location_id=location_id,
                                                             year_id=year_id,
                                                             sex_id=sex_id)
            one_year_file['year_id'] = year_id
            one_year_file['sex_id'] = sex_id
            draws = draws.append(one_year_file)

    # TODO: Need to rethink setting ages for this function. Since sbp estimates start for the
    # age 25-29 group, it should start at age 25, not 27.5.
    draws = get_age_group_midpoint_from_age_group_id(draws)
    draws = expand_ages(draws)
    draws.set_index(['year_id', 'sex_id', 'age'], inplace=True)

    # Set nulls to be 1 to keep from messing up the math below. The nulls are the
    # younger age groups (simulants less than 27.5 years old) and they'll get an
    # sbp of 112 and an sd of .001 because we want them to be at the TMRED
    draws[['exp_mean_{}'.format(i) for i in range(1000)]] = draws[
        ['exp_mean_{}'.format(i) for i in range(1000)]].fillna(value=1)
    draws[['exp_sd_{}'.format(i) for i in range(1000)]] = draws[
        ['exp_sd_{}'.format(i) for i in range(1000)]].fillna(value=1)

    # FIXME: This process does produce a df that has null values for simulants
    # under 27.5 years old for the exp_mean and exp_sd cols.
    # Dont think this will affect anything but may be worth fixing
    exp_mean = draws[['exp_mean_{}'.format(i) for i in range(1000)]].values
    exp_sd = draws[['exp_sd_{}'.format(i) for i in range(1000)]].values

    mean_df = pd.DataFrame(np.log(exp_mean),
                           columns=['log_mean_{}'.format(i) for i in range(1000)],
                           index=draws.index)
    sd_df = pd.DataFrame(np.divide(exp_sd, exp_mean),
                         columns=['log_sd_{}'.format(i) for i in range(1000)],
                         index=draws.index)

    output_df = mean_df.join(sd_df)

    output_df = output_df.reset_index()
    mean_columns = ['log_mean_{}'.format(i) for i in range(1000)]
    sd_columns = ['log_sd_{}'.format(i) for i in range(1000)]
    young_idx = output_df.age <= 27.5
    output_df.loc[young_idx, mean_columns] = np.log(112)
    output_df.loc[young_idx, sd_columns] = 0.001

    validate_data(output_df, ['age', 'year_id', 'sex_id'])

    keep_columns = ['year_id', 'sex_id', 'age'] + mean_columns + sd_columns

    return output_df[keep_columns].sort_values(by=['year_id', 'age', 'sex_id'])


def get_angina_proportions():
    """Format the angina proportions so that we can use them in CEAM.
    This is messy. The proportions were produced by Catherine Johnson.
    The proportion differs by age, but not by sex, location, or time.
    This will likely change post GBD-2016.
    Returns
    -------
    df with year_id, sex_id, age and 1k draws

    Notes
    -----
    Assumptions -- The file does not have estimates for people under age 20. I've set the
    proportions for people under age 20 to be the same as the proportion for people that are 20 years old.
    This shouldn't have much of an impact on anything, since we don't expect for
    people under age 20 to have heart attacks.

    Questions -- Is it valid to assign the angina proportion for 20 year olds to be the
    angina proportions for people under the age of 20? Who should we talk to about
    having these proportions stored in a better place (e.g. the database)? Who should
    we talk to about ensuring that this file doesn't move? How can we ensure that the
    file is updated if need be?
    """
    ang = gbd.get_data_from_auxiliary_file('Angina Proportions')

    # Not sure why income is included in this file. estimates are the same for high and low income countries.
    # We'll filter on high income to get rid of the superfluous rows.
    ang = ang.query("income == 'high'")
    ang = get_age_group_midpoint_from_age_group_id(ang)

    # We don't have estimates under age 22.5, so I'm filling all
    # ages under 20 with the same proportion that we have for 20 year olds.
    # TODO: Should check this assumption w/ Abie
    # values are same for each sex, so we can grab the value for the lowest age from either
    # sex to apply to the younger age groups for which we do not have data.
    ang_copy = ang.set_index('age').copy()
    value_at_youngest_age_for_which_we_have_data = ang_copy.query("sex_id == 1").get_value(22.5, 'angina_prop')
    total_ang = pd.DataFrame()

    # the data is not year specific. we manually add year_id values here
    # TODO: Probably a more sophisticated way to do this
    for year in [1990, 1995, 2000, 2005, 2010, 2013, 2015]:
        one_year = ang.copy()
        one_year['year_id'] = year
        total_ang = total_ang.append(one_year)

    total_ang = total_ang[['year_id', 'sex_id', 'age', 'angina_prop']]
    total_ang = total_ang.apply(lambda x: x.fillna(value_at_youngest_age_for_which_we_have_data), axis=0)

    return total_ang


def get_disability_weight(cause, draw_number):
    """Returns a dataframe with disability weight draws for a given healthstate id

    Parameters
    ----------
    draw_number: int
        GBD draw to pull data for

    cause: ceam_inputs.gbd_mapping.Cause or ceam_inputs.gbd_mapping.Sequela
           or ceam_inputs.gbd_mapping.Etiology or ceam_inputs.gbd_mapping.SeveritySplit

    Returns
    -------
    df with disability weight draws

    Notes
    -----
    Assumptions -- None

    Questions -- How can IHME create a more systematic way for access this data?
    The current way (looking in one csv prepared by central comp and then checking another
    if the draws are not in the first csv) is pretty disorganized. Since many disability
    weights are going to be updated in 2016, these files may move. I would propose that we
    ask central comp to store the disability weights in the database.

    Unit test in place? -- Yes
    """
    if isinstance(cause.disability_weight, meid):
        healthstate_id = gbd.get_healthstate_id(cause.disability_weight)
        modelable_entity_id = cause.disability_weight
    elif isinstance(cause.disability_weight, hid):
        healthstate_id = cause.disability_weight
        modelable_entity_id = None
    else:
        raise ValueError('Unknown id type: {}'.format(type(cause.disability_weight)))

    df = None
    # TODO: Need to confirm with someone on central comp that all 'asymptomatic' sequela get this healthstate_id
    if healthstate_id == 799:
        df = pd.DataFrame({'healthstate_id': [799],
                           'healthstate': ['asymptomatic'],
                           'draw{}'.format(draw_number): [0]})
    else:
        for file_name in ['Disability Weights', 'Combined Disability Weights']:
            weights = gbd.get_data_from_auxiliary_file(file_name)
            if healthstate_id in weights.healthstate_id.tolist():
                df = weights[weights.healthstate_id == healthstate_id]
                break

    if df is None:
        raise ValueError("the modelable entity id {} ".format(modelable_entity_id)
                         + "has a healthstate_id of {}. There are no draws for".format(healthstate_id)
                         + "this healthstate_id in the csvs that get_healthstate_id_draws checked. "
                         + "Look in this folder for the draws for healthstate_id {}: ".format(healthstate_id)
                         + "/home/j/WORK/04_epi/03_outputs/01_code/02_dw/03_custom. "
                         + "if you can't find draws there, talk w/ central comp")

    df.loc[:, 'modelable_entity_id'] = modelable_entity_id

    return df['draw{}'.format(draw_number)].iloc[0]


def get_asympt_ihd_proportions(location_id, year_start, year_end, draw_number):
    """
    Gets the proportion of post-mi simulants that will get asymptomatic ihd.
    Proportion that will get asymptomatic ihd is equal to 1 - proportion of
    mi 1 month survivors that get angina + proportion of mi 1 month survivors
    that get heart failure

    Parameters
    ----------
    Feed in parameters required by get_post_mi_heart_failure_proportion_draws and get_angina_proportion_draws

    Returns
    -------
    df with post-mi asymptomatic ihd proportions

    Notes
    -----
    Assumptions -- That all people who survive a heart attack then get one of asymptomatic ihd, heart failure, or angina

    Questions -- None

    Unit test in place? -- Yes
    """

    hf_prop_df = get_post_mi_heart_failure_proportion_draws(location_id, year_start, year_end, draw_number,
                                                            gbd_round_id=config.simulation_parameters.gbd_round_id)
    angina_prop_df = get_angina_proportions()

    merged = pd.merge(hf_prop_df, angina_prop_df, on=['age', 'year_id', 'sex_id'])
    merged = merged.set_index(['year_id', 'sex_id', 'age'])
    hf_values = merged[['draw_{}'.format(i) for i in range(1000)]].values
    angina_values = merged[['angina_prop']].values

    asympt_prop_df = pd.DataFrame(1 - hf_values - angina_values,
                                  columns=['asympt_prop_{}'.format(i) for i in range(1000)],
                                  index=merged.index)

    keep_columns = ['year_id', 'sex_id', 'age']
    keep_columns.extend(('asympt_prop_{i}'.format(i=i) for i in range(1000)))

    return asympt_prop_df.reset_index()[keep_columns]


def get_age_specific_fertility_rates(location_id, year_start, year_end):
    # TODO: I'm loading this from disk because central comp doesn't have a good
    # tool for ingesting covariates from python and I don't feel like writing
    # any more stata. They say there should be something in a couple of weeks
    # and we should switch to it asap. -Alec 11/01/2016
    asfr = gbd.get_data_from_auxiliary_file('Age-Specific Fertility Rates')
    asfr = asfr.query('age_group_id in {} or age_group_id in {}'.format(gbd.ZERO_TO_EIGHTY, gbd.EIGHTY_PLUS))
    asfr = asfr.query(
        'location_id == {} and year_id >= {} and year_id <= {}'.format(location_id, year_start, year_end)).copy()
    asfr['sex'] = 'Female'
    asfr = asfr.rename(columns={'year_id': 'year', 'mean_value': 'rate'})
    asfr = get_age_group_midpoint_from_age_group_id(asfr)
    keep_columns = ['age', 'sex', 'year', 'rate']

    return asfr.reset_index(level=0)[keep_columns]


# TODO: Write a SQL query for get_covariate_estimates that returns a covariate id instead of
# covariate short name, because names are subject to change but ids should stay the same
# TODO: Also link that covariate id to a publication id, if possible
def get_covariate_estimates(covariate_name_short, location_id, year_id=None, sex_id=None):
    """Gets covariate estimates for a specified location.

    Processes data to put in correct format for CEAM (i.e. gets estimates for all years/ages/ and both sexes.)

    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD
    covariate_name_short: str
        the covariate_short_name for the covariate of interest.
        you can look up covariate_short_names here: http://cn307.ihme.washington.edu:9998/
        (check the covariate_metadata_tab in website above)
    year_id: int
    sex_id: int

    Returns
    -------
    A dataframe of covariate_estimates.
        Column are age, sex_id, year_id, and {etiology_name}_incidence_{draw} (1k draws)
    """
    covariate_estimates = gbd.get_covariate_estimates(covariate_name_short, location_id)
    if year_id:
        covariate_estimates = covariate_estimates[covariate_estimates.year_id == year_id]
    if sex_id:
        covariate_estimates = covariate_estimates[covariate_estimates.sex_id == sex_id]
    return covariate_estimates


def get_severity_splits(parent_meid, child_meid, draw_number):
    """
    Returns a severity split proportion for a given cause

    parent_meid: int, modelable_entity_id
        the modelable entity id for the severity split

    child_meid: int, modelable_entity_id
        the modelable entity id for the severity split

    draw_number: int
        specific draw number

    See also
    --------
    To determine parent and child meids, see here: http://dev-tomflem.ihme.washington.edu/sevsplits/editor
    If the severity splits that you are require are not in the file path below,
    email central comp to ask them to create the splits.
    """
    splits = gbd.get_data_from_auxiliary_file('Severity Splits', parent_meid=parent_meid)

    # the splits don't always add up exactly to one, so I get the sum of the splits and
    # then divide each split by the total to scale to 1
    total = splits[['draw_{}'.format(draw_number)]].sum()
    splits['scaled'] = splits['draw_{}'.format(draw_number)] / total.values
    splits = splits.query("child_meid == {}".format(child_meid))

    return splits[["scaled"]].values.item(0)


# TODO: Write a test for get_rota_vaccine_coverage.
# Make sure values make sense for year/age in test, similar to get_relative_risk tests
def get_rota_vaccine_coverage(location_id, year_start, year_end, gbd_round_id):
    draws = gbd.get_modelable_entity_draws(location_id, meid(10596), gbd_round_id)
    draws = draws.query('age_group_id < {}'.format(6))
    draws = draws.query('year_id>={ys} and year_id<={ye}'.format(ys=year_start, ye=year_end))
    draws = get_age_group_midpoint_from_age_group_id(draws)
    draws = expand_ages(draws)
    draws[['draw_{}'.format(i) for i in range(1000)]] = draws[
        ['draw_{}'.format(i) for i in range(1000)]].fillna(value=0)

    keep_columns = ['year_id', 'sex_id', 'age']
    keep_columns.extend(('draw_{i}'.format(i=i) for i in range(0, 1000)))

    draws = draws[keep_columns]

    validate_data(draws, ['age', 'year_id', 'sex_id'])

    return draws.sort_values(by=['year_id', 'age', 'sex_id'])


def get_ors_pafs(location_id, year_start, year_end):
    """
    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    year_end : int, end year
        year_end is the year in which you want to end the simulation

    draw_number: int
        current draw number (as specified in config.run_configuration.draw_number)
    """
    pafs = gbd.get_data_from_auxiliary_file('Ors Pafs', location_id=location_id)
    pafs = get_age_group_midpoint_from_age_group_id(pafs)
    pafs = pafs.query("year_id >= {} and year_id <= {}".format(year_start, year_end))
    return pafs


def get_ors_relative_risks(draw_number):
    """
    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    year_end : int, end year
        year_end is the year in which you want to end the simulation

    draw_number: int
        current draw number (as specified in config.run_configuration.draw_number)
    """
    rr = gbd.get_data_from_auxiliary_file('Ors Relative Risks')
    return float(rr[rr.parameter == 'cat1']['draw_{}'.format(draw_number)][0])


def get_ors_exposures(location_id, year_start, year_end, draw_number):
    """
    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    year_end : int, end year
        year_end is the year in which you want to end the simulation

    draw_number: int
        current draw number (as specified in config.run_configuration.draw_number)
    """
    ors_exp = gbd.get_data_from_auxiliary_file('Ors Exposure', location_id=location_id)

    exp = expand_ages_for_dfs_w_all_age_estimates(ors_exp)
    exp = exp.query("year_id >= {} and year_id <= {}".format(year_start, year_end))

    # Per Patrick Liu, the ors relative risk and exposure estimates are only valid
    # for children under 5 the input data only uses the all ages age group id since
    # the covariates database requires that covariates apply to all ages.
    exp = exp.query("age < 5")
    exp = expand_ages(exp)
    exp[['draw_{}'.format(i) for i in range(1000)]] = exp[['draw_{}'.format(i) for i in range(1000)]].fillna(value=0)

    keep_columns = ['year_id', 'sex_id', 'age', 'parameter', 'draw_{}'.format(draw_number)]
    return exp[keep_columns]


def get_diarrhea_visit_costs(location_id, year_start, year_end, draw_number):
    """
    Parameters
    ----------
    location_id : int
        location_id takes same location_id values as are used for GBD

    year_start : int, year
        year_start is the year in which you want to start the simulation

    year_end : int, end year
        year_end is the year in which you want to end the simulation

    draw_number: int
        current draw number (as specified in config.run_configuration.draw_number)
    """
    if location_id not in [179, 161, 214]:
        raise ValueError("We only currently have outpatient costs for Ethiopia, Bangladesh, and Nigeria")

    costs = gbd.get_data_from_auxiliary_file('Diarrhea Visit Costs')
    costs = costs.query("location_id == {}".format(location_id))
    costs = costs.query("variable == 'draw_{}'".format(draw_number))
    costs = costs.query("year_id >= {} and year_id <= {}".format(year_start, year_end))

    return costs


def get_mediation_factors(risk_id, cause_id, draw_number):
    mediation_factors = gbd.get_data_from_auxiliary_file('Mediation Factors')
    mediation_factors = mediation_factors.query("rei_id == {} and cause_id == {}".format(risk_id, cause_id))
    return 0 if mediation_factors.empty else np.prod(1 - mediation_factors['draw_{}'.format(draw_number)])


def validate_data(draws, duplicate_columns=None):
    # Use introspection to get the name of the calling function.
    current_frame = inspect.currentframe()
    caller_frame = inspect.getouterframes(current_frame, 2)
    caller_name = caller_frame[1][3]

    assert not np.any(draws.isnull()), ("There are nulls in the data that {} tried to output. ".format(caller_name)
                                        + "Check the cache to to make sure the data you're pulling is correct.")

    if duplicate_columns:
        assert not draws.duplicated(duplicate_columns).sum(), ("There are duplicates in the dataframe that "
                                                               + "{} tried to output. ".format(caller_name)
                                                               + "Check the cache to to make sure the data "
                                                               + "you're pulling is correct.")


def get_fpg_distribution_parameters(location_id, year_start, year_end, draw):
    columns = ['age_group_id', 'sex_id', 'year_id', 'sll_loc_{}'.format(draw),
               'sll_scale_{}'.format(draw), 'sll_error_{}'.format(draw)]

    sub_location_ids = gbd.get_subregions(location_id)
    if not sub_location_ids:
        sub_location_ids = [location_id]
    sex_ids = [1, 2]
    year_ids = range(year_start, year_end + 1, 5)

    dfs = [gbd.get_data_from_auxiliary_file('Fasting Plasma Glucose Distributions', location_id=sub_location_id,
                                            year_id=year_id, sex_id=sex_id)[columns].assign(location=sub_location_id)
           for sub_location_id, year_id, sex_id in product(sub_location_ids, year_ids, sex_ids)]

    parameters = pd.concat(dfs).drop_duplicates()
    parameters = get_age_group_midpoint_from_age_group_id(parameters).drop(['age_group_start', 'age_group_end'], axis=1)
    parameters = (parameters
                  .replace({'sex_id': 1}, 'Male').replace({'sex_id': 2}, 'Female')
                  .rename(columns={'sex_id': 'sex',
                                   'year_id': 'year',
                                   'sll_loc_{}'.format(draw): 'loc',
                                   'sll_scale_{}'.format(draw): 'scale',
                                   'sll_error_{}'.format(draw): 'error'}))
    if not config.simulation_parameters.use_subregions:
        return aggregate_location_data(parameters, location_id)
    return parameters


def aggregate_location_data(df, location_id):
    weights = get_subregion_weights(location_id)
    merge_cols = ['age', 'sex', 'year', 'location']
    param_cols = list(set(df.columns).difference(merge_cols))
    df = df.merge(weights, on=['age', 'sex', 'year', 'location'], how='inner')
    for c in param_cols:
        df[c] = df[c].mul[df.location_weight]

    df = df.drop(['location_weight', 'location'], axis=1)
    return df.groupby(['age', 'year', 'sex'], as_index=False).sum().assign(location=location_id)


def get_subregion_weights(location_id):
    sub_pops = pd.concat([get_populations(location_id=location) for location in gbd.get_subregions(location_id)],
                         ignore_index=True)
    sub_pops = sub_pops[['age', 'sex', 'year', 'location_id', 'pop_scaled']].rename(columns={'location_id': 'location'})
    sub_pops = sub_pops[sub_pops.sex != 'Both']
    sub_pops['location_weight'] = (
        sub_pops
            .groupby(['age', 'sex', 'year'], as_index=False)
            .apply(lambda sub_pop: sub_pop.pop_scaled / sub_pop.pop_scaled.sum())
            .reset_index(level=0).pop_scaled
    )
    return sub_pops.drop('pop_scaled', axis=1)



def get_bmi_distribution_parameters(location_id, year_start, year_end, draw):
    a = pd.DataFrame()
    b = pd.DataFrame()
    loc = pd.DataFrame()
    scale = pd.DataFrame()
    for sex_id in [1, 2]:
        for year_id in np.arange(year_start, year_end + 1, 5):
            a = a.append(gbd.get_data_from_auxiliary_file('Body Mass Index Distributions',
                                                          parameter='bshape1',
                                                          location_id=location_id,
                                                          year_id=year_id,
                                                          sex_id=sex_id))

            b = b.append(gbd.get_data_from_auxiliary_file('Body Mass Index Distributions',
                                                          parameter='bshape2',
                                                          location_id=location_id,
                                                          year_id=year_id,
                                                          sex_id=sex_id))
            loc = loc.append(gbd.get_data_from_auxiliary_file('Body Mass Index Distributions',
                                                              parameter='mm',
                                                              location_id=location_id,
                                                              year_id=year_id,
                                                              sex_id=sex_id))
            scale = scale.append(gbd.get_data_from_auxiliary_file('Body Mass Index Distributions',
                                                                  parameter='scale',
                                                                  location_id=location_id,
                                                                  year_id=year_id,
                                                                  sex_id=sex_id))
    a = a.set_index(['age_group_id', 'sex_id', 'year_id'])
    b = b.set_index(['age_group_id', 'sex_id', 'year_id'])
    loc = loc.set_index(['age_group_id', 'sex_id', 'year_id'])
    scale = scale.set_index(['age_group_id', 'sex_id', 'year_id'])

    parameters = pd.DataFrame()
    parameters['a'] = a['draw_{}'.format(draw)]
    parameters['b'] = b['draw_{}'.format(draw)]
    parameters['loc'] = loc['draw_{}'.format(draw)]
    parameters['scale'] = scale['draw_{}'.format(draw)]

    parameters = parameters.reset_index()
    parameters = get_age_group_midpoint_from_age_group_id(parameters)
    parameters['year'] = parameters.year_id
    parameters.loc[parameters.sex_id == 1, 'sex'] = 'Male'
    parameters.loc[parameters.sex_id == 2, 'sex'] = 'Female'

    return parameters[['age', 'year', 'sex', 'a', 'b', 'scale', 'loc']]


def get_dtp3_coverage(location_id, year_start, year_end, draw_number):
    if gbd.get_subregions(location_id):
        raise ValueError('DTP 3 coverage only available at the finest geographic level.  '
                         'Use the subregion ids {}'.format(gbd.get_subregions(location_id)))
    dtp3 = gbd.get_data_from_auxiliary_file('DTP3 Coverage', location_id=location_id)
    dtp3 = expand_ages_for_dfs_w_all_age_estimates(dtp3)

    # TODO: Confirm below assumption.
    # Per Patrick Liu, the ors relative risk and exposure estimates are only valid
    # for children under 5 the input data only uses the all ages age group id since
    # the covariates database requires that covariates apply to all ages
    dtp3 = dtp3.query("age < 5")
    dtp3 = expand_ages(dtp3)
    dtp3[['draw_{}'.format(i) for i in range(1000)]] = dtp3[['draw_{}'.format(i) for i in range(1000)]].fillna(value=0)
    dtp3 = dtp3.query("year_id >= {} and year_id <= {}".format(year_start, year_end))

    keep_columns = ['year_id', 'sex_id', 'age', 'draw_{}'.format(draw_number)]
    return dtp3[keep_columns]


def get_rota_vaccine_protection(location_id, draw_number):

    protection = gbd.get_data_from_auxiliary_file('Rota Vaccine Protection')
    assert location_id in protection.location_id.unique(), ("protection draws do not exist for the "
                                                            + "requested location id -- {}. ".format(location_id)
                                                            + "you may need to generate them")
    return protection.set_index(['location_id']).get_value(location_id, 'draw_{}'.format(draw_number))


def get_rota_vaccine_rrs(location_id, draw_number):

    rrs = gbd.get_data_from_auxiliary_file('Rota Vaccine RRs')
    assert location_id in rrs.location_id.unique(), ("rr draws do not exist for the "
                                                            + "requested location id -- {}. ".format(location_id)
                                                            + "you may need to generate them")
    return rrs.set_index(['location_id']).get_value(location_id, 'draw_{}'.format(draw_number))


def get_diarrhea_costs(location_id, year_start, year_end, draw_number):
    costs = gbd.get_data_from_auxiliary_file('Diarrhea Costs', location_id=location_id)
    costs = costs[(costs.year_id >= year_start) & (costs.year_id <= year_end)]
    costs = costs[[c for c in costs.columns if ('draw_{}'.format(draw_number) in c or 'draw' not in c)]]
    return costs.rename(columns={'year_id': 'year', 'draw_{}'.format(draw_number): 'cost'})


def get_ors_costs(location_id, year_start, year_end, draw_number):
    costs = gbd.get_data_from_auxiliary_file('ORS Costs', location_id=location_id)
    costs = costs[(costs.year_id >= year_start) & (costs.year_id <= year_end)]
    costs = costs[[c for c in costs.columns if ('draw_{}'.format(draw_number) in c or 'draw' not in c)]]
    return costs.rename(columns={'year_id': 'year', 'draw_{}'.format(draw_number): 'cost'})
