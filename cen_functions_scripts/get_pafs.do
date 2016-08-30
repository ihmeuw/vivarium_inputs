// clear current data set and set more off
clear all
set more off

// Set output directory
local outpath `3'

// Set country of interest
local location_id `1'

// Set sexes of interest
local sexes 1 2

// Set cause of interest
local cause_id `2'

// Get every GBD age group (5 year intervals... age_group_id = 2-21, 30-33)
local ages
forvalues i = 2/21 {
local ages "`ages' `i' "
}
foreach i in 30 31 32 33 {
local ages "`ages' `i' "
}

// Use get_outputs function to generate results
get_draws, gbd_id_field(cause_id) gbd_id(`cause_id') age_group_ids(`ages') location_ids(`location_id') sex_ids(`sexes') source(dalynator) measure_ids(1) status(best) include_risks clear


// Output results to a csv file
outsheet using `outpath', comma replace

exit, STATA clear
