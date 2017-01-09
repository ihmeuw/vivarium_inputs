GBD MS Functions Documentation
------------------------------


get_modelable_entity_draws
--------------------------
Used by -- get_cause_level_prevalence, sum_up_csmrs_for_all_causes_in_microsim, get_post_mi_heart_failure_proportion_draws, get_excess_mortality, get_incidence, get_continuous, get_proportion, get_prevalence

Assumptions -- None

Questions -- None

Unit test in place? -- No. Don't think it's necessary, since this function merely pulls draws from the database and then filters a dataframe so that only one measure is included in the output and that only the years in b/w the simulation year start and year end are included in the df.


generate_ceam_population
------------------------
Used by -- None

Assumptions -- None

Questions -- None

Unit test in place? -- Yes

TODO -- Need to smooth out initial ages (JIRA ticket - CE-213)


get_cause_level_prevalence
--------------------------
Used by -- assign_cause_at_beginning_of_simulation

Assumptions -- That the sequela prevalences associated with a cause will add up to the cause level prevalence

Questions -- Would it be better to just pull cause level prevalence? I'm a bit worried that the sequela prevalences won't add up

Unit test in place? -- Yes


determine_if_sim_has_cause
--------------------------
Used by -- assign_cause_at_beginning_of_simulation

Assumptions -- None

Questions -- I sort the prevalence and simulants dataframes by simulant_id to make sure that the prevalence is being assigned correctly to each demographic group. Is there a better way to make sure that we're applying the correct prevalence rate to each simulant?

Unit test in place? -- Yes


get_sequela_proportions
-----------------------
Used -- That the prevalence of a sequela can be divided by the prevalence of the cause associated with that sequela to get the proportional prevalence.

Questions -- None

Unit test in place? -- Yes


determine_which_seq_diseased_sim_has
------------------------------------
Used by -- assign_cause_at_beginning_of_simulation

Assumptions -- None

Questions -- None

Unit test in place? -- Yes. 


assign_cause_at_beginning_of_simulation
---------------------------------------
Used by -- get_disease_states

Assumptions -- None

Questions -- None

Unit test in place? -- I wrote code to produce graphs to make sure we're assigning prevalence correctly at the beginning of the simulation. I need to automate that code so that the graphs are produced each time CEAM is run.


sum_up_csmrs_for_all_causes_in_microsim
---------------------------------------
Used by -- get_cause_deleted_mortality_rate

Assumptions -- That we can add together the csmrs for every cause in the microsim and then subtract from the all-cause mortality rate to get the cause-deleted mortality rate.

Questions -- None

Unit test in place? -- Yes


get_cause_deleted_mortality_rate
--------------------------------
Used by -- Used in base_population.py

Assumptions -- That we can subtract the csmrs for the causes we care about to get the cause-deleted mortality rate

Questions -- None

Unit test in place? -- Yes


get_post_mi_heart_failure_proportion_draws
------------------------------------------
Used by -- Used in disease_models.py to determine how many people get heart failure following an mi.

Assumptions -- That the proportional prevalence is a good enough estimation of the proportional incidence.

Questions -- More of a general python question -- should I be using np.multiply for multiplication? Maybe it has to do with python's floating point issues, but I was getting different results when using A*B instead of np.multiply(A,B).

Unit test in place? --  Yes


get_relative_risks
------------------
Used by -- Used to pull relative risks which are then multiplied by incidence rates in continuous_exposure_effect and categorical_exposure_effect

Assumptions -- Some risks in GBD (e.g. Zinc deficiency and high sbp) don't have estimates for all ages. I have set up the code so that each age group for which we don't have GBD estimates has an RR of 1 (i.e. no elevated risk). 

Questions -- Should we set the RR to 1 for age groups for which we do not have rr estimates?

Unit test in place? -- No. Just pulls relative risks from the database and then does some light processing (e.g. gets age group midpoints)


get_pafs
--------
Used by -- Some risks in GBD (e.g. Zinc deficiency and high sbp) don't have estimates for all ages. I have set up the code so that each age group for which we don't have GBD estimates has a PAF of 0

Assumptions -- We should use PAFs for DALYs, since we use PAFs to affect incidence in CEAM

Questions -- Should we be using PAFs for Deaths or DALYs? Should we set the PAF to 0 for age groups for which we do not have rr estimates? Need to submit an epihelp ticket to determine whether we should use get_draws or transmogrifier.risk.risk_draws. 

Unit test in place? -- No. Just pulls pafs from the database and then does some light processing (e.g. gets age group midpoints)


get_exposures
-------------
Assumptions -- Some risks in GBD (e.g. Zinc deficiency and high sbp) don't have estimates for all ages. I have set up the code so that each age group for which we don't have GBD estimates has an exposure of 0

Questions -- Should we set the exposure to 0 for age groups for which we do not have rr estimates? Need to submit an epihelp ticket to determine whether we should use get_draws or transmogrifier.risk.risk_draws.

Unit test in place? -- No. Just pulls exposures from the database and then does some light processing (e.g. gets age group midpoints)



get_sbp_mean_sd
---------------
Assumptions -- That people under age 25 have the TMRED SBP 

Questions -- We have estimates starting in the age 25-29 age group. Should we be using the midpoint or age 25 as the starting point?

Unit test in place? -- Yes

TRMED -- Might want to change the TMRED. Need to catch up with Stan regarding calculating TMREDs + write a function that will allow us to calculate TMREDs for a given risk.


get_angina_proportions
----------------------
Assumptions -- The file does not have estimates for people under age 20. I've set the proportions for people under age 20 to be the same as the proportion for people that are 20 years old. This shouldn't have much of an impact on anything, since we don't expect for people under age 20 to have heart attacks.

Questions -- Is it valid to assign the angina proportion for 20 year olds to be the angina proportions for people under the age of 20? Who should we talk to about having these proportions stored in a better place (e.g. the database)? Who should we talk to about ensuring that this file doesn't move? How can we ensure that the file is updated if need be?

Unit test in place? -- Yes


get_disability_weight
---------------------
Assumptions -- None

Questions -- How can IHME create a more systematic way for access this data? The current way (looking in one csv prepared by central comp and then checking another if the draws are not in the first csv) is pretty disorganized. Since many disability weights are going to be updated in 2016, these files may move. I would propose that we ask central comp to store the disability weights in the database.

Unit test in place? -- Not yet


get_asympt_ihd_proportions
--------------------------
Assumptions -- That all people who survive a heart attack then get one of asymptomatic ihd, heart failure, or angina

Questions -- None

Unit test in place? -- Not yet
