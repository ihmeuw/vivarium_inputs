GBD MS Functions Documentation
------------------------------


get_modelable_entity_draws
--------------------------
Purpose -- Pulls all available draws (e.g. prevalence, incidence, proportion, etc.) for a given modelable entity id. 

Functionality -- Uses central comp's get draws function.

Dependencies -- get_cause_level_prevalence, sum_up_csmrs_for_all_causes_in_microsim, get_post_mi_heart_failure_proportion_draws, get_excess_mortality, get_incidence, get_continuous, get_proportion, get_prevalence

Assumptions -- None

Questions -- None

Unit test in place? -- No. Have some code that produces graphs of the output of get_modelable_entity_draws, which we can manually compare to epi viz, but I would to figure out how to automatically compare the output of get_modelable_entity_draws to what's in epi viz.


generate_ceam_population
------------------------
Purpose -- Creates a population of simulants to be fed into the CEAM microsimulation 

Functionality -- Uses get_populations, create_age_column, and create_sex_id_column (all gbd ms auxiliary functions)

Dependencies -- None

Assumptions -- None

Questions -- None

Unit test in place? -- Not currently. Would be good to have a unit test here.

TODO -- Need to smooth out initial ages (JIRA ticket - CE-213)


get_cause_level_prevalence
--------------------------
Purpose -- Get cause-level prevalence gets the prevalence of a certain cause by aggregating the prevalences of the sequela associated with that cause

Functionality -- Appends dataframes of sequela prevalences and then sums up the prevalences using pd.groupby

Dependencies -- assign_cause_at_beginning_of_simulation

Assumptions -- That the sequela prevalences associated with a cause will add up to the cause level prevalence

Questions -- Would it be better to just pull cause level prevalence? I'm a bit worried that the sequela prevalences won't add up

Unit test in place? -- Yes


determine_if_sim_has_cause
--------------------------
Purpose -- Determines if the simulant has a cause at the beginning of the simulation

Functionality -- Merges the dataframe of simulants to a dataframe with cause-level prevalence on age and sex to ensure that the correct prevalence values are applied to the correct demographics. Then uses CEAM's choice function to determine who is a case. Prevalence and 1-prevalence are used as weights.

Dependencies -- assign_cause_at_beginning_of_simulation

Assumptions -- None

Questions -- None

Unit test in place? -- No. This function feeds into assign_cause_at_beginning_of_simulation, which is tested.


get_sequela_proportions
-----------------------
Purpose -- Gets the proportional prevalence of each sequela associated with a cause

Dependencies -- assign_cause_at_beginning_of_simulation

Assumptions -- That the prevalence of a sequela can be divided by the prevalence of the cause associated with that sequela to get the proportional prevalence.

Questions -- None

Unit test in place? -- No. This function feeds into assign_cause_at_beginning_of_simulation, which is tested.


determine_which_seq_diseased_sim_has
------------------------------------
Purpose -- At this point we know who is afflicted by a cause and the proportional incidence for the sequela associated with that cause. determine_which_seq_diseased_sim_has assigns sequelas according to the sequela proportions. 

Functionality -- Uses CEAM's choice function to determine who is a case. Prevalence and 1-prevalence are used as weights.

Dependencies -- assign_cause_at_beginning_of_simulation

Assumptions -- None

Questions -- None

Unit test in place? -- No. This function feeds into assign_cause_at_beginning_of_simulation, which is tested.


assign_cause_at_beginning_of_simulation
---------------------------------------
Purpose -- Assigns prevalence of modeled diseases to the starting population of simulants.

Dependencies -- get_disease_states

Assumptions -- None

Questions -- None

Unit test in place? -- I wrote code to produce graphs to make sure we're assigning prevalence correctly at the beginning of the simulation. I need to automate that code so that the graphs are produced each time CEAM is run.


sum_up_csmrs_for_all_causes_in_microsim
---------------------------------------
Purpose -- Sums up the cause-specific mortality rate for every cause in a simulation. The sums of all of the cause-specific mortality rates are then subtracted from the all-cause mortality rate to get the cause-deleted mortality rate.

Functionality -- Uses groupby to sum all of the csmrs together

Dependencies -- get_cause_deleted_mortality_rate

Assumptions -- That we can add together the csmrs for every cause in the microsim and then subtract from the all-cause mortality rate to get the cause-deleted mortality rate.

Questions -- None

Unit test in place? -- Yes


get_cause_deleted_mortality_rate
--------------------------------
Purpose -- Need to calculate the cause-deleted mortality rate so that simulants can die from causes not explicitly modeled in the microsimulation.

Functionality -- Uses sum_up_csmrs_for_all_causes_in_microsim to sum up the csmrs for all causes in the microsim. 

Dependencies -- Used in base_population.py

Assumptions -- That we can subtract the csmrs for the causes we care about to get the cause-deleted mortality rate

Questions -- None

Unit test in place? -- Yes

get_post_mi_heart_failure_proportion_draws
------------------------------------------
Purpose -- Estimating the incidence of heart failure after an MI. This solution definitely is not perfect. We take the incidence of all heart failure, then multiply it by the proportion of heart failure due to mi, to estimate the incidence of heart failure due to mi. The proportion is based on prevalence, which makes this method pretty imperfect. We also convert the rate to a probability within this function, using CEAM's rate to probability function (1-np.exp(-rate))

Functionality -- Uses get_modelable_entity_draws to get the envelope incidence and post-mi proportion, then divides proportion by the envelope. Uses rate_to_probability to convert the rate to a probability.

Dependencies -- Used in disease_models.py to determine how many people get heart failure following an mi.

Assumptions -- That the proportional prevalence is a good enough estimation of the proportional incidence.

Questions -- More of a general python question -- should I be using np.multiply for multiplication? Maybe it has to do with python's floating point issues, but I was getting different results when using A*B instead of np.multiply(A,B).

Unit test in place? --  No. The function uses get_modelable_entity draws (which is tested) twice and then just multiplies the values together. Should put in a test saying that it pulled the correct model version once I write the code to link the pull to the publication id.


get_relative_risks
------------------
Purpose -- Pulls relative risk draws from the database

Functionality -- Uses central comp's get_draws function and CEAM's get_age_group_midpoint_from_age_group_id auxiliary function

Dependencies -- Used to pull relative risks which are then multiplied by incidence rates in continuous_exposure_effect and categorical_exposure_effect

Assumptions -- Some risks in GBD (e.g. Zinc deficiency and high sbp) don't have estimates for all ages. I have set up the code so that each age group for which we don't have GBD estimates has an RR of 1 (i.e. no elevated risk). 

Questions -- Should we set the RR to 1 for age groups for which we do not have rr estimates?

Unit test in place? -- No. But should put one in place to make sure correct model numbers are being pulled.


get_pafs
--------
Purpose -- Pulls PAFs draws from the database

Functionality -- Uses central comp's get_draws function and CEAM's get_age_group_midpoint_from_age_group_id auxiliary function

Assumptions -- Some risks in GBD (e.g. Zinc deficiency and high sbp) don't have estimates for all ages. I have set up the code so that each age group for which we don't have GBD estimates has a PAF of 0

Questions -- Should we set the PAF to 0 for age groups for which we do not have rr estimates? Need to submit an epihelp ticket to determine whether we should use get_draws or transmogrifier.risk.risk_draws.

Unit test in place? -- No. But should put one in place to make sure correct model numbers are being pulled.


get_exposures
-------------
Purpose -- Pulls exposure draws from the database

Functionality -- Uses central comp's get_draws function and CEAM's get_age_group_midpoint_from_age_group_id auxiliary function

Assumptions -- Some risks in GBD (e.g. Zinc deficiency and high sbp) don't have estimates for all ages. I have set up the code so that each age group for which we don't have GBD estimates has an exposure of 0

Questions -- Should we set the exposure to 0 for age groups for which we do not have rr estimates? Need to submit an epihelp ticket to determine whether we should use get_draws or transmogrifier.risk.risk_draws.

Unit test in place? -- No. But should put one in place to make sure correct model numbers are being pulled.


get_sbp_mean_sd
---------------
Purpose -- Need to pull the mean and standard deviation of sbp for demographic groups

Functionality -- Manually pulls from csvs created by central comp

Assumptions -- 

Questions -- We have estimates starting in the age 25-29 age group. Should we be using the midpoint or age 25 as the starting point?

Unit test in place? -- 

TRMED -- Might want to change the TMRED. Need to catch up with Stan regarding calculating TMREDs + write a function that will allow us to calculate TMREDs for a given risk.


get_angina_proportions
----------------------
Purpose -- Returns the proportion of people who get angina (instead of heart failure or asymptomatic ihd) after a heart attack

Functionality -- We pull in an excel spreadsheet that was manually created by Catherine Johnson

Assumptions -- The file does not have estimates for people under age 20. I've set the proportions for people under age 20 to be the same as the proportion for people that are 20 years old. This shouldn't have much of an impact on anything, since we don't expect for people under age 20 to have heart attacks.

Questions -- Is it valid to assign the angina proportion for 20 year olds to be the angina proportions for people under the age of 20? Who should we talk to about having these proportions stored in a better place (e.g. the database)? Who should we talk to about ensuring that this file doesn't move? How can we ensure that the file is updated if need be?

Unit test in place? -- Yes


get_disability_weight
---------------------
Purpose -- Returns disability weights for a given modelable entity id

Functionality -- Uses CEAM's get_healthstate_id auxiliary function and then pulls disability weights from csvs created by central comp

Assumptions -- None

Questions -- How can IHME create a more systematic way for access this data? The current way (looking in one csv prepared by central comp and then checking another if the draws are not in the first csv) is pretty disorganized. Since many disability weights are going to be updated in 2016, these files may move. I would propose that we ask central comp to store the disability weights in the database.

Unit test in place? -- Not yet


get_asympt_ihd_proportions
--------------------------
Purpose -- Returns the proportion of people who survive a heart attack who should get asymptomatic ihd. We assign heart failure using get_post_mi_heart_failure_proportion_draws and angina using get_angina_proportions. The proportion of people with angina equals 1 - proportion of mi 1 month survivors that get angina + proportion of mi 1 month survivors that get heart failure


Functionality -- Uses get_post_mi_heart_failure_proportion_draws and get_angina_proportions to determine the proportion of simulants who should get angina.

Assumptions -- That all people who survive a heart attack then get one of asymptomatic ihd, heart failure, or angina

Questions -- None

Unit test in place? -- Not yet
