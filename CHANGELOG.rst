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
