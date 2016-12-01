// Use home/J if running on the cluster, J: if running locally
if c(os) == "Unix" {
                global prefix "/home/j"
                set odbcmgr unixodbc
        }
        else if c(os) == "Windows" {
                global prefix "J:"
        }
// Connect to J Drive for shared function
adopath + "$prefix/WORK/10_gbd/00_library/functions"

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
cap get_draws, gbd_id_field(rei_id) gbd_id(`rei_id') location_ids(`location_id') source(risk) status(best) kwargs(draw_type:rr) clear   // this will default to pulling 2015 results

if _rc != 0 {   // if the above command returned no results
    get_draws, gbd_id_field(rei_id) gbd_id(`rei_id') location_ids(`location_id') source(risk) status(best) kwargs(draw_type:rr gbd_round_id:2) clear
} // will try to pull 2013 results

// Output results to a csv file
outsheet using `outpath', comma replace

exit, STATA clear
