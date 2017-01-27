// Two steps here //
// 1) Pull proportion estimates for each etiology
// 2) Convert proportion to qPCR definition

clear all
set more off

qui do "/home/j/WORK/10_gbd/00_library/functions/get_draws.ado"
qui do "/home/j/WORK/10_gbd/00_library/functions/get_location_metadata.ado"


// cache results
local outpath `2'


get_location_metadata, location_set_id(9) clear
levelsof location_id, local(locs)


import delimited "/home/j/temp/ctroeger/GEMS/eti_rr_me_ids.csv", clear
keep if modelable_entity_id == `1'
levelsof modelable_entity, local(m) c


// Part 1:
get_draws, gbd_id_field(modelable_entity_id) source(dismod) gbd_id(`1') age_group_ids(2 3 4 5) sex_ids(1 2) ///
	year_ids(1990 1995 2000 2005 2010 2015) location_ids(`locs') clear

	
// Part 2:
merge m:1 modelable_entity_id using "/home/j/temp/ctroeger/GEMS/adjustment_matrix_bimodal.dta", keep(3) nogen


forval i = 1/1000{
	local j = `i'-1
	gen proportion_`j' = (draw_`j'+ specificity_`i' - 1)/(sensitivity_`i' + specificity_`i' - 1)
	replace proportion_`j' = 1 if proportion_`j' > 1
	replace proportion_`j' = 0.001 if proportion_`j' < 0
}

drop sensitivity_* specificity_* draw_*
rename proportion_* draw_*
gen modelable_entity_name = "`m'"

keep year_id age_group_id cause_id modelable_entity_id modelable_entity_name rei_id sex_id rei_name location_id draw_*

// Part 3: Save!	
save "`outpath'", replace

