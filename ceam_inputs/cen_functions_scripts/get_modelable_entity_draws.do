// Connect to J Drive for shared function
adopath + "/home/j/temp/central_comp/libraries/current/stata/"

// clear current data set and set more off
clear all
set more off

// Set output directory
local outpath `4'

// Set country of interest (using iso3 code)
local location_id `1'

// Set modelable_id of interest
local me_id `2'

// Set status
local status `3'

// Get every GBD age group (5 year intervals... age_group_id = 2-21, 30-33)
local ages
forvalues i = 2/21 {
local ages "`ages' `i' "
}
// foreach i in 30 31 32 33 {
// local ages "`ages' `i' "
// }

// Use get_draws function to generate results
get_draws, gbd_id_field(modelable_entity_id) gbd_id(`me_id') location_ids(`location_id') source(epi) age_group_ids(`ages') status(`status') clear

// Output results to a csv file
outsheet using `outpath', comma replace

exit, STATA clear
