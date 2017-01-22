// Connect to J Drive for shared function
adopath + "/home/j/temp/central_comp/libraries/current/stata/"

// clear current data set and set more off
clear all
set more off

// Set output directory
local outpath `4'

// Set country of interest
local location_id `1'

// Set rei of interest
local rei_id `2'

<<<<<<< HEAD
// Get every GBD age group (5 year intervals... age_group_id = 2-21, 30-33)
local ages
forvalues i = 2/21 {
local ages "`ages' `i' "
}
foreach i in 30 31 32 33 {
local ages "`ages' `i' "
}

// Use get_draws function to generate results
get_draws, gbd_id_field("rei_id") gbd_id(`rei_id') location_ids(`location_id') kwargs(draw_type:exposure) age_group_ids(`ages') status(best) source(risk) clear

if _rc != 0 {   // if the above command returned no results
    get_draws, gbd_id_field(rei_id) gbd_id(`rei_id') location_ids(`location_id') kwargs(draw_type:exposre gbd_round_id:2) age_group_ids(`ages') status(best) source(risk) clear
=======
// Set gbd_round_id of interest
local gbd_round `3'

// Use get_draws function to generate results
// Use get_draws function to generate results
cap get_draws, gbd_id_field(rei_id) gbd_id(`rei_id') location_ids(`location_id') source(risk) kwargs(draw_type:exposure) status(best) gbd_round_id(`gbd_round') clear

if _rc != 0 {   // if the above command returned no results
    get_draws, gbd_id_field(rei_id) gbd_id(`rei_id')location_ids(`location_id') source(risk) kwargs(draw_type:exposure) status(best) gbd_round_id(2) clear
>>>>>>> develop
} // will try to pull 2013 results

// Output results to a csv file
outsheet using `outpath', comma replace

exit, STATA clear
