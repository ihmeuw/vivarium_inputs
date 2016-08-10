// Run this script on the DEV cluster
// qlogin -pe multi_slot 2 -now no
// stata do "/homes/emumford/notebooks/hypertension_icer/Hello_World/get_pafs.do"

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
local dir "/ihme/scratch/users/emumford/emumford/microsim/random"

// Set GBD years of interest
local years 1990 1995 2000 2005 2010 2013

// Set country of interest (using iso3 code)
local iso3 180

// Set sexes of interest
local sex 1 2 3

// Set rei of interest
local rei 107

// Set cause of interest
local cause 493

// Get every GBD Age Group (5 year intervals... age_group_id = 2-21)
set obs 20
egen age_group_ids = fill(2 3)
levelsof(age_group_ids), local(ages)

// Use get_outputs function to generate results
get_outputs, topic(rei) metric_id(2) rei_id(`rei') cause_id(`cause') gbd_round(2013) age_group_id(`ages') year_id(`years') location_id(`iso3') sex_id(`sex') clear


// Output results to a csv file
outsheet using "`dir'/PAFs_of_risk_`rei'_for_`cause'_in_`iso3'.csv", comma replace

