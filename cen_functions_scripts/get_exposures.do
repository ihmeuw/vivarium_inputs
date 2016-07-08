// Run this script on the DEV cluster
// qlogin -pe multi_slot 2 -now no
// stata do "/homes/emumford/notebooks/hypertension_icer/Hello_World/get_exposures.do"

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
local dir "/share/costeffectiveness/CEAM/gbd_to_microsim_unprocessed_data"

// Set country of interest (using iso3 code)
local iso3 `1'

// Set rei of interest
local rei `2'

// Set years of interest
local year 2010

// Use get_draws function to generate results
get_draws, gbd_id_field(rei_id) gbd_id(`rei') year_ids(`year') location_ids(`iso3') status(latest) kwargs(draw_type:exposure) source(risk) clear

// Output results to a csv file
outsheet using "`dir'/Exposure_of_risk`rei'_in_location`iso3'_inyear`year'.csv", comma replace

