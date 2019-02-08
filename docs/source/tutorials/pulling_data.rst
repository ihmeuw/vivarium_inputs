============
Pulling Data
============

:mod:`vivarium_inputs` provides an interface to pull data from GBD + auxiliary data
that will perform the same validations, transformations, and standardizations
as occur for data that is used in a simulation.

You can use this interface to examine data that you want to use in a model to
ensure it passes all validations and looks as you expect.

.. contents::
    :depth: 1
    :local:
    :backlinks: none

:mod:`vivarium_inputs` provides interface methods to pull entity-measure data (e.g.,
prevalence data for a cause or exposure for a risk factor) as well as
population structure, life expectancy, and data about the extents of certain
demographic variables (i.e., specifically age bins as well as all demographic
dimensions combined).

Entity-Measure Data
--------------------
The interface provides :func:`vivarium_inputs.interface.get_measure` for pulling
specific measure data for an entity for a single location.
`entity` should be a :class:`gbd_mapping.base_template.ModelableEntity` (e.g.,
a cause or coverage_gap from ``gbd_mapping``), while `measure` should be a string
describing the measure for which you want to retrieve data (e.g., 'prevalence'
or 'relative_risk'). A list of possible measures for each entity
kind is included in the table below. Finally, `location` should be the string
location for which you want to pull data (e.g., 'Ethiopia'), in the form used by
GBD (e.g., 'United States' instead of 'USA').

For example, to pull prevalence data for diarrheal diseases in Kenya, we would
do the following:

.. code-block:: python

    from gbd_mapping import causes
    from vivarium_inputs.interface import get_measure

    prev = get_measure(causes.diarrheal_diseases, 'prevalence', 'Kenya')
    print(prev.head())

::

      draw location     sex  age_group_start  age_group_end  year_start  year_end     value
    0    0    Kenya  Female              0.0       0.019178        1990      1991  0.032557
    1    0    Kenya  Female              0.0       0.019178        1991      1992  0.031751
    2    0    Kenya  Female              0.0       0.019178        1992      1993  0.031039
    3    0    Kenya  Female              0.0       0.019178        1993      1994  0.030458
    4    0    Kenya  Female              0.0       0.019178        1994      1995  0.030039

The following table lists the measures available for each entity kind:

.. list-table:: Available Entity-Measure Pairs
    :header-rows: 1
    :widths: 30, 40

    *   - Entity Kind
        - Measures
    *   - sequela
        - | incidence
          | prevalence
          | birth_prevalence
          | disability_weight
    *   - cause
        - | incidence
          | prevalence
          | birth_prevalence
          | disability_weight
          | remission
          | cause_specific_mortality
          | excess_mortality
    *   - coverage_gap
        - | exposure
          | exposure_standard_deviation
          | exposure_distribution_weights
          | relative_risk
    *   - risk_factor
        - | exposure
          | exposure_standard_deviation
          | exposure_distribution_weights
          | relative_risk
          | population_attributable_fraction
          | mediation_factors
    *   - alternative_risk_factor
        - | exposure
          | exposure_standard_deviation
          | exposure_distribution_weights
    *   - etiology
        - | population_attributable_fraction
    *   - covariate
        - | estimate
    *   - healthcare_entity
        - | cost
          | utilization
    *   - health_technology
        - | cost

Population Structure Data
-------------------------
To pull population data for a specific location, :mod:`vivarium_inputs.interface`
provides :func:`vivarium_inputs.interface.get_population_structure`, which returns
population data in the input format expected by a simulation.

For example, to pull population data for Kenya, we would do the following:

.. code-block:: python

    from vivarium_inputs.interface import get_population_structure

    pop = get_population_structure('Kenya')
    print(pop.head())

::

      location     sex  age_group_start  age_group_end  year_start  year_end        value
    0    Kenya  Female              0.0       0.019178        1990      1991  9251.406428
    1    Kenya  Female              0.0       0.019178        1991      1992  9371.524292
    2    Kenya  Female              0.0       0.019178        1992      1993  9488.631659
    3    Kenya  Female              0.0       0.019178        1993      1994  9592.689862
    4    Kenya  Female              0.0       0.019178        1994      1995  9701.918801

Life Expectancy Data
--------------------
To pull life expectancy data, :mod:`vivarium_inputs.interface`
provides :func:`vivarium_inputs.interface.get_theoretical_minimum_risk_life_expectancy`,
which returns life expectancy data in the input format expected by a simulation.
Because life expectancy is not location specific, the function takes no arguments.

To use:

.. code-block:: python

    from vivarium_inputs.interface import get_theoretical_minimum_risk_life_expectancy

    life_exp = get_theoretical_minimum_risk_life_expectancy()
    print(life_exp.head())

::

       age_group_start  age_group_end      value
    0             0.00           0.01  87.885872
    1             0.01           0.02  87.877086
    2             0.02           0.03  87.868299
    3             0.03           0.04  87.859513
    4             0.04           0.05  87.850727


Age Bin Data
------------
To see what age bins GBD uses that are used in age-specific data, :mod:`vivarium_inputs`
provides :func:`vivarium_inputs.interface.get_age_bins`, which returns the start,
end, and name of each GBD age bin expected to appear in age-specific data (with
the exception of life expectancy, which uses its own age ranges).

.. code-block:: python

    from vivarium_inputs.interface import get_age_bins

    age_bins = get_age_bins()
    print(age_bins.head())

::

       age_group_start  age_group_end  age_group_name
    0         0.000000       0.019178  Early Neonatal
    1         0.019178       0.076712   Late Neonatal
    2         0.076712       1.000000   Post Neonatal
    3         1.000000       5.000000          1 to 4
    4         5.000000      10.000000          5 to 9


Demographic Dimensions Data
---------------------------
Finally, to view the full extent of all demographic dimensions that is expected
in input data to the simulation,  :mod:`vivarium_inputs` provides
:func:`vivarium_inputs.interface.get_demographic_dimensions`, which expects a `location`
argument to fill the location dimension.

.. code-block:: python

    from vivarium_inputs.interface import get_demographic_dimensions

    dem_dims = get_demographic_dimensions('Kenya')
    print(dem_dims.head())

::

      location     sex  age_group_start  age_group_end  year_start  year_end
    0    Kenya  Female              0.0       0.019178        1990      1991
    1    Kenya  Female              0.0       0.019178        1991      1992
    2    Kenya  Female              0.0       0.019178        1992      1993
    3    Kenya  Female              0.0       0.019178        1993      1994
    4    Kenya  Female              0.0       0.019178        1994      1995


.. testcode::
    :hide:

    import inspect
    import pandas as pd

    from vivarium_inputs.interface import (get_measure, get_population_structure,
                                           get_theoretical_minimum_risk_life_expectancy,
                                           get_age_bins, get_demographic_dimensions)
    from gbd_mapping import ModelableEntity

    funcs = {get_measure: {'entity': ModelableEntity, 'measure': str, 'location': str},
             get_population_structure: {'location': str},
             get_theoretical_minimum_risk_life_expectancy: {},
             get_age_bins: {},
             get_demographic_dimensions: {'location': str}
    }
    for func, params in funcs.items():
        sig = inspect.signature(func)
        assert len(sig.parameters) == len(params)
        for name, annotation in params.items():
            assert name in sig.parameters
            assert sig.parameters[name].annotation == annotation
        assert sig.return_annotation == pd.DataFrame

