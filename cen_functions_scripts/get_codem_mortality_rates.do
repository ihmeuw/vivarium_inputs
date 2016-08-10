// Run this script on the DEV cluster
// qlogin -pe multi_slot 2 -now no
// stata do "/share/costeffectiveness/CEAM/gbd_to_microsim_code/get_codem_mortality_rates.do"

// Use home/J if running on the cluster, J: if running locally
if c(os) == "Unix" {
                global prefix "/home/j"
                set odbcmgr unixodbc
        }
        else if c(os) == "Windows" {
                global prefix "J:"
        }

// clear current data set and set more off
clear all
set more off

// Connect to J Drive for get_populations function
adopath + "$prefix/WORK/10_gbd/00_library/functions"

// Set output directory
local outdir "/share/costeffectiveness/CEAM/gbd_to_microsim_unprocessed_data"

// Set country of interest (using iso3 code)
local location_id `1'

// Set modelable_id of interest
local cause_id `2'

// Use get_draws function to generate results
get_draws, gbd_id_field(cause_id) gbd_id(`cause_id') location_ids(`location_id') source(codem) clear

// Output results to a csv file
outsheet using "`outdir'/codem_draws_for_location`location_id'_for_causeid`cause_id'.csv", comma replace


