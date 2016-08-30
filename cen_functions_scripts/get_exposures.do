// clear current data set and set more off
clear all
set more off

// Set output directory
local outpath `3'

// Set country of interest
local location_id `1'

// Set rei of interest
local rei_id `2'

// Use get_draws function to generate results
get_draws, gbd_id_field(rei_id) gbd_id(`rei_id') location_ids(`location_id') kwargs(draw_type:exposure) status(best) source(risk) clear

// Output results to a csv file
outsheet using `outpath', comma replace

exit, STATA clear
