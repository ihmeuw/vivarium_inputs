// Run this script on the DEV cluster
// 3 Commands:
// qlogin -pe multi_slot 2 -now no
// stata
// do "/share/costeffectiveness/CEAM/gbd_to_microsim_code/get_all_cause_mortality.do"

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

// Set GBD years of interest
local years 1990 1995 2000 2005 2010 2013

// Set country of interest (using iso3 code)
local location_id `1'

// Set gbd_round
local gbd_round `2'

// Set sexes of interest
local sex 1 2 3

// Set cause of interest
local cause 294

// Get every GBD Age Group (5 year intervals... age_group_id = 2-21)
set obs 20
egen age_group_ids = fill(2 3)
levelsof(age_group_ids), local(ages)

// Use get_outputs function to generate results
get_outputs, topic(cause) measure_id(1) metric_id(3) location_id (`location_id') sex_id(`sex') cause_id(`cause') age_group_id(`ages') gbd_round(`2') year_id(`years') clear

// Output results to a csv file. Columns are age_group id, year_id, location_id, sex_id, and pop_scaled
outsheet using "`outdir'/all_cause_mortality_causeid`cause'_in_country`location_id'.csv", comma replace


