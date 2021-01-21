**4.0.2 - 01/21/21**

 - Bugfix release, special case for LBWSG data

**4.0.1 - 01/6/21**

 - Bugfix release, fix deployment

**4.0.0 - 01/5/21**

 - Update for GBD 2019
 - Added tests for GBD version changes (not run under CI)

**3.1.1 - 01/14/20**

 - Fix bug in hierarchical data sorting

**3.1.0 - 11/18/19**

 - Update information about pulling data from gbd.
 - Move data artifact from vivarium public health to vivarium.
 - Data exclusion for moderate wasting with edema.
 - Change 'age_group_start' and 'age_group_end' to 'age_start' and 'age_end'.
 - Consistent naming of rate data.
 - Get rid of old multi-build artifact code.
 - Add the core-maths package as a dependency.

**3.0.1 - 07/16/19**

 - Allow for wider YLL age ranges than YLD with warning instead of error.
 - Add special cases for a handful of causes with very high excess mortality.
 - Bugfix for validating deaths data for sex-specific causes.

**3.0.0 - 06/18/19**

 - Use indexes for artifact data.
 - Bugfixes for lbwsg.
 - Expand tmrel code to pull data when available.
 - Memory optimizations.
 - CI script to properly handle dependencies.
 - Update metadata.
 - Shift data produce to be wide on draws.
 - Update component manager usage.
 - Correct import guard issue.
 - Special case for iron deficiency exposure sd.

**2.0.3 - 03/29/19**

 - Update memory requirements for building artifacts.
 - Fix multi-build artifact.
 - Add integration tests.
 - Make reshape faster, idempotent, and move to get_data function.
 - Various bugfixes.

**2.0.2 - 02/24/19**

 - Make sure demographic dimensions are in every artifact.
 - Expose public API
 - Remove forecast vestiges.
 - Add top level function to pull raw GBD data.
 - Update memory limits on artifact builder.

**2.0.1 - 02/14/19**

 - Update dependencies

**2.0.0 - 02/14/19**

 - Rewrite vivarium inputs for GBD 2017.

**1.0.18 - 01/04/19**

 - Add support for multi-location artifacts.
 - Add CI branch synchronization.

**1.0.17 - 12/20/18**

 - Bugfix to normalize disability weight data.

**1.0.16 - 12/20/18**

 - Build in functionality for pulling forecasting data for demography and diarrhea etiologies.
 - Add functions for coverage and effects for health technologies.
 - Allow pulling disability weight by cause.
 - Update PAF and RR data format to support excess mortality effects.

**1.0.15 - 11/16/18**

 - Dependency mismatch

**1.0.14 - 11/15/18**

 - Update documentation dependencies

**1.0.13 - 11/15/18**

 - Add tool to locally build artifacts
 - Setup artifact filter terms in configuration.
 - Remove mean columns for age and year.
 - Allow paf of 1 data.
 - Allow artifact building on the new cluster.

**1.0.12 - 11/08/18**

 - Include a new version of vivarium_gbd_access version

**1.0.11 - 11/07/18**

 - Pull pafs by risk instead of by cause.

**1.0.10 - 11/05/18**

 - Make entity/location/measure arguments to functions in core singular.
 - Fix artifact builder bug causing disappearing builds.

**1.0.9 - 10/26/18**

 - Added kind attribute to mapping objects

**1.0.8 - 10/26/18**

 - Fix bug in artifact builder to properly store processed keys.
 - Switch artifact builder to only load needed data.
 - Return age_group_start and age_group_end to support order0 interpolation
 - Bugfix in locations with apostrophes

**1.0.7 - 10/16/18**

 - Add support for arbitrary covariates.
 - PAF naming bugfix

**1.0.6 - 10/09/18**

 - Update dependencies

**1.0.5 - 10/09/18**

 - Switch from true incidence to incidence hazard
 - Fix in artifact builder to get correct causes
 - Add affected_risk_factors to risks

**1.0.4 - 09/28/18**

 - Added measles covariates

**1.0.3 - 09/25/18**

 - Update for coverage gap exposure/rr/paf
 - Add support for building artifacts for locations with spaces.
 - Bugfixes around measles

**1.0.2 - 09/12/18**

 - Add logging to artifact builder.
 - Various bugfixes in artifact builder.
 - Bugfix for smoking prevalence w/r/t tuberculosis

**1.0.1 - 8/22/18**

 - Move all file handling to artifact in public health
 - Rewrite loaders to be usable for individual measures
 - Make vivarium_gbd_access mockable
 - Rewrite artifact build script to support multiple locations
 - Build ArtifactPassthrough
 - Switch to new aux data source

**1.0.0 - 7/31/18**

 - Initial Release
