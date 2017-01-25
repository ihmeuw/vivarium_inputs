run "/home/j/temp/central_comp/libraries/current/stata/get_covariate_estimates.ado"

// Set covariate id
local cov_id `1'

// Set outpath
local outpath `2'

get_covariate_estimates, covariate_name_short(`cov_id') clear

outsheet using `outpath', comma replace

exit, clear
