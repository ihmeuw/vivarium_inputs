// Run this script on the DEV cluster
// 3 Commands:
// qlogin -pe multi_slot 2 -now no
// stata
// do "/share/code/costeffectiveness/microsim/gbd_data_to_microsim/get_populations.do"

// Use home/J if running on the cluster, J: if running locally
if c(os) == "Unix" {
                global prefix "/home/j"
                set odbcmgr unixodbc
        }
        else if c(os) == "Windows" {
                global prefix "J:"
        }

// clear current data and set more off
clear all
set more off

// Connect to J Drive for get_populations function
adopath + "$prefix/WORK/10_gbd/00_library/functions"

// Set output directory
local outdir "/share/costeffectiveness/CEAM/gbd_to_microsim_unprocessed_data"

// Set country of interest (using iso3 code)
local location_id `1'

// Set GBD years of interest
local years 1990 1995 2000 2005 2010 2013 2015

// Set sexes of interest
local sex 1 2 3

// Get every GBD age group (5 year intervals... age_group_id = 2-21)
set obs 20
egen age_group_ids = fill(2 3)
levelsof(age_group_ids), local(ages)

// Use get_populations function to generate results
get_populations , year_id(`years') location_id(`location_id') sex_id(`sex') age_group_id(`ages') include_names clear

// Output results to a csv file. Columns are age_group id, year_id, location_id, sex_id, and pop_scaled
outsheet using "`outdir'/pop_`location_id'.csv", comma replace

