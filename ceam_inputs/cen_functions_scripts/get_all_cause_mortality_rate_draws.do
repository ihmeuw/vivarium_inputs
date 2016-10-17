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
local outpath `2'

// Set country of interest (using iso3 code)
local location_id `1'

// Use get_draws function to generate results
get_draws, gbd_id_field("cause_id") gbd_id(294) location_ids(`location_id') measure_ids(1) source(dalynator) status(best) clear

// Output results to a csv file
outsheet using `outpath', comma replace

exit, STATA clear
