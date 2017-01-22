// Connect to J Drive for shared function
adopath + "/home/j/temp/central_comp/libraries/current/stata/"

// clear current data set and set more off
clear all
set more off

// Set output directory
local outpath `5'

// Set country of interest
local location_id `1'

// Set sexes of interest
local sexes 1 2

// Set cause of interest
local cause_id `2'

// Set gbd round of interest
local gbd_round `3'

// Get every GBD age group (5 year intervals... age_group_id = 2-21, 30-33)
local ages
forvalues i = 2/21 {
local ages "`ages' `i' "
}
foreach i in 30 31 32 33 {
local ages "`ages' `i' "
}

<<<<<<< HEAD
// If cause is diarrhea, we want to use YLDs, not deaths as the measure
if `cause_id' == 302 {
    local measure_id = 3
}
else {
    local measure_id = 1
}


// Use get_outputs function to generate results
get_draws, gbd_id_field(cause_id) gbd_id(`cause_id') age_group_ids(`ages') location_ids(`location_id') sex_ids(`sexes') source(dalynator) measure_ids(`measure_id') status(best) include_risks clear
// TODO: THINK WE WANT TO BE USING YLDS (measure_id=3) FOR PAFS SINCE WE'RE AFFECTING INCIDENCE (JIRA Ticket CE-343)

keep if metric_id==2
=======
// Set measure_id of interest
local measure `4'

// Use get_outputs function to generate results
get_draws, gbd_id_field(cause_id) gbd_id(`cause_id') age_group_ids(`ages') location_ids(`location_id') sex_ids(`sexes') source(dalynator) measure_ids(`measure') status(best) gbd_round_id(`gbd_round') include_risks(1) clear
// TODO: THINK WE WANT TO BE USING YLDS (measure_id=3) FOR PAFS SINCE WE'RE AFFECTING INCIDENCE (JIRA Ticket CE-343)
>>>>>>> develop

// Output results to a csv file
outsheet using `outpath', comma replace

exit, STATA clear
