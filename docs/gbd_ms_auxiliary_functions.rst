GBD-MS Auxiliary Functions Documentation
----------------------------------------


create_age_column
-----------------
Purpose -- Creates a new column called 'age' which proportionally assigns ages to a population of simulants.

Functionality -- Uses numpy random choice to create a column of ages. The proportions are determined by the number of simulants in each age category in the simulants_df input

Inputs -- A population of simulants with an age column that represents the number of people in each 5 year age group. when input, the values in the age column are age group midpoints

Dependencies -- generate_ceam_population

Assumptions -- None

Questions - None

Unit test in place? -- Yes


normalize_for_simulation
------------------------
Purpose -- Normalize age and sex values and remove 'id' from year_id col

Functionality -- Uses pd.map to remap sex_id values 1 and 2 to sex values Male and Female. Uses pd.rename to rename year_id column

Inputs -- Any dataframe with sex_id and year_id columns

Dependencies -- load_data_from_cache

Assumptions -- None

Questions -- None

Unit test in place? -- Yes


get_age_group_midpoint_from_age_group_id
-------------------------
Purpose -- Get the age group midpoints for each age group id. GBD data is estimated for 5 year age groups. We want to get the midpoints so that we can use mipoints as knots for our interpolation functions.

Functionality -- Uses db_tools.ez.funcs (central comp function) to access the database to pull the age group start/age group end age for each age group. Then uses pd.mean to get the midpoint of the age group start/age group end. For the oldest age group (age 80+, we manually set the midpoint to be 82.5)

Dependencies -- get_modelable_entity_draws, get_relative_risks, get_pafs, get_exposures, get_sbp_mean_sd, get_bmi_distributions, get_age_specific_fertility_rates, get_populations

Assumptions -- We assume that using a midpoint of age 82.5 for the 80+ year old age group is ok for the purposes of CEAM. Everett proposed that we could get the life expectancy at age 80 for each location and use that as the midpoint for the 80+ group, but Abie suggested that we keep the midpoint as 82.5 for now. GBD populations have data for each age group up until the age 95+ age group, at which point I'm assuming we can use 97.5 as the midpoint.

Questions -- None

Unit test in place? -- Yes


get_populations
---------------
Purpose -- Get the GBD populations for a specific location

Functionality -- Uses central comp's get_population function. returns population estimates for every age group up through age group 90-94. also includes population estimates for age group 95+.

Dependencies -- generate_ceam_population, create_sex_id_column

Assumptions --  None

Questions -- None

Unit test in place? -- No. Don't think one is needed. We just use the central comp get_population function to get the population data and then select a specific year, specific sex, and use the get_age_group_midpoint_from_age_group_id function to get the age group midpoints.

Uncertainty draws -- Need to be cognizant of the fact that there are not currently uncertainty estimates for populations in GBD, but that these estimates will be produced for GBD 2017, and maybe even GBD 2016. Hopefully, when the draws are ready, we will be able to continue using central comp's get_populations function.

TODO -- There is a python function for get_population, but I am not currently able to import anything from db_queries 


assign_sex_id
-------------
Purpose -- Assigns sex to a population of simulants so that age and sex are correlated

Functionality -- Uses np.random.choice to assing sex_id values. Each simulant has an age assigned. We know how many men/women there are in each age group from the GBD populations (we use get_populations to pull the GBD populations) so we can assign a sex to each simulant and ensure that age/sex are correlated.

Dependencies -- create_sex_id_column

Assumptions -- That we can assign ages/sexes at different times while still ensuring correlation.

Questions -- Currently, when we generate a population of simulants, we assign age and then assign sex. Should we be assigning age and sex at the same time?

Unit test in place? -- Yes


create_sex_id_column
--------------------
Purpose -- Creates a sex_id column for a population of simulants

Functionality -- Uses get_populations to get proportions of men/women in each age group for a given location/year. Then uses assign_sex_id to assign sex to simulants while ensuring correlation between age/sex. Then ensures that the values in sex_id are integers.

Dependencies -- generate_ceam_population

Assumptions -- That we can assign ages/sexes at different times while still ensuring correlation.

Questions -- Currently, when we generate a population of simulants, we assign age and then assign sex. Should we be assigning age and sex at the same time?

Unit test in place? -- No. Don't think it's needed for this function, since this function just utilizes two of our other functions (get_populations and assign_sex_id) which are already tested.


get_all_cause_mortality_rate
----------------------------
Purpose -- We need the all cause mortality rate as an input for cause-deleted mortality rate

Functionality -- Pulls in the number of deaths in a location, age, sex, year group and divides the number of deaths by the population in each location, age, sex, year group to get the mortality rate. 

Dependencies -- get_cause_deleted_mortality_rate 

Assumptions -- None

Questions -- Is the dalynator the correct source for pulling the all-cause mortality rate? In some of the developing countries, the mortality rate is higher than 1 for the very young age groups. Should we be doing anything about this in the simulation?

Unit test in place? -- Not currently, but one does need to be put in place


get_healthstate_id
------------------
Purpose -- Gets the health state id for a given modelable entity id. Disability weights are assigned to health states, so we need to get the health state id to determine the disability weight for a given modelable entity id.

Functionality -- Uses central comp's db_tools.ezfuncs to connect to the epi database to get the healthstate_id that corresponds with a specific modelable entity id

Dependencies -- get_disability_weight

Assumptions -- None

Questions -- None

Unit test in place? -- Yes 
