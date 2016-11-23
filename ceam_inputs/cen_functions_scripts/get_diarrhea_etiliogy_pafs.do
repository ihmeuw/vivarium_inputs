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

// Set country of interest
local location_id `1'

// Set reii_id of interest
local rei_id `2'

// Set output directory
local outpath `3'

import delimited "/home/j/temp/ctroeger/GEMS/eti_rr_me_ids.csv", clear

keep if rei_id == `1'

levelsof modelable_entity, local(m) c
 
get_draws, gbd_id_field(rei_id) source(dalynator) gbd_id(`1') sex_ids(1 2) location_ids(`location_id') measure_ids(3) // measure_id 3 is YLDs, which provides the non-fatal PAFs. Since were calculating incidence, we want non-fatal pafs

clear

keep if metric_id==2 // metric_id 2 is percent

save “", replace

