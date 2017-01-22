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


// Set gbd round of interest
local gbd_round `3'

// Use get_draws function to generate results
// Use get_draws function to generate results
cap get_draws, gbd_id_field(rei_id) gbd_id(`rei_id') location_ids(`location_id') source(risk) gbd_round_id(`gbd_round') kwargs(draw_type:rr) clear   // this will default to pulling 2015 results

if _rc != 0 {   // if the above command returned no results
    get_draws, gbd_id_field(rei_id) gbd_id(`rei_id') location_ids(`location_id') source(risk) gbd_round_id(2) kwargs(draw_type:rr) clear
} // will try to pull 2013 results

// Output results to a csv file
outsheet using `outpath', comma replace

exit, STATA clear
