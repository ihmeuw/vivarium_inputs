============
Pulling Data
============

:mod:`vivarium_inputs` provides an interface to pull data from GBD + auxiliary
data. Use this interface to examine data that you want to use in a model to
ensure it passes all validations and looks as you expect. You have 2 choices
for pulling data:

  - :func:`vivarium_inputs.interface.get_measure` will pull data through the
    same process as simulations do, which will perform the same validations,
    transformations, and standardizations as occur for data that is used in a
    simulation.

  - :func:`vivarium_inputs.interface.get_raw_data` will pull raw data from GBD,
    skipping all validations in order to explore and investigate.

Both of the above methods can retrieve entity-measure data (e.g.,
prevalence data for a cause or exposure for a risk factor), population structure,
and life expectancy. Functions to retrieve data about the extents of certain
demographic variables --  :func:`vivarium_inputs.interface.get_age_bins` and
:func:`vivarium_inputs.interface.get_demographic_dimensions` are somewhat
orthogonal and imply the same data modifications inherent in
calling :func:`vivarium_inputs.interface.get_measure`.

.. contents::
    :depth: 2
    :local:
    :backlinks: none


Which should I use... get_measure() versus get_raw_data()
----------------------------------------------------------
Prefer :func:`vivarium_inputs.interface.get_measure` over
:func:`vivarium_inputs.interface.get_raw_data`. :func:`vivarium_inputs.interface.get_measure`
will produce simulation-prepped data. If :func:`vivarium_inputs.interface.get_measure`
fails, or the data it returns doesn't match your expectations, then
:func:`vivarium_inputs.interface.get_raw_data` might provide some insight
into what is happening.

Pulling Simulation-Prepped Data
-------------------------------
For simulation-prepped data, the interface provides separate methods to pull
entity-measure data and population structure and life expectancy data. Additionally,
methods to pull age bin data and demographic dimensions are provided. Simulation-
prepped data has had GBD IDs replaced with meaningful values or ranges and
expansion over all demographic dimensions has been performed.  We'll walk
through how to pull data using each of these functions.

Entity-Measure Data
+++++++++++++++++++
The interface provides :func:`vivarium_inputs.interface.get_measure` for pulling
specific measure data for an entity for a single location.
`entity` should be a :class:`gbd_mapping.base_template.ModelableEntity` (e.g.,
a cause from ``gbd_mapping``), while `measure` should be a string
describing the measure for which you want to retrieve data (e.g., 'prevalence'
or 'relative_risk'). A list of possible measures for each entity
kind is included in the table below. Finally, `location` should be the string
location for which you want to pull data (e.g., 'Ethiopia'), in the form used by
GBD (e.g., 'United States' instead of 'USA').

To pull simulation-prepped entity-measure data, you must have plenty of available
memory. Please have at least 50GB on a qlogin.

For example, to pull prevalence data for diarrheal diseases in Kenya, we would
do the following:

.. code-block:: python

    from gbd_mapping import causes
    from vivarium_inputs import get_measure

    prev = get_measure(causes.diarrheal_diseases, 'prevalence', 'Kenya')
    print(prev.head())

::

      draw location     sex        age_start        age_end  year_start  year_end     value
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
+++++++++++++++++++++++++
To pull population data for a specific location, :mod:`vivarium_inputs.interface`
provides :func:`vivarium_inputs.interface.get_population_structure`, which returns
population data in the input format expected by a simulation.

For example, to pull population data for Kenya, we would do the following:

.. code-block:: python

    from vivarium_inputs import get_population_structure

    pop = get_population_structure('Kenya')
    print(pop.head())

::

      location     sex        age_start        age_end  year_start  year_end        value
    0    Kenya  Female              0.0       0.019178        1990      1991  9251.406428
    1    Kenya  Female              0.0       0.019178        1991      1992  9371.524292
    2    Kenya  Female              0.0       0.019178        1992      1993  9488.631659
    3    Kenya  Female              0.0       0.019178        1993      1994  9592.689862
    4    Kenya  Female              0.0       0.019178        1994      1995  9701.918801

Life Expectancy Data
++++++++++++++++++++
To pull life expectancy data, :mod:`vivarium_inputs.interface`
provides :func:`vivarium_inputs.interface.get_theoretical_minimum_risk_life_expectancy`,
which returns life expectancy data in the input format expected by a simulation.
Because life expectancy is not location specific, the function takes no arguments.

To use:

.. code-block:: python

    from vivarium_inputs import get_theoretical_minimum_risk_life_expectancy

    life_exp = get_theoretical_minimum_risk_life_expectancy()
    print(life_exp.head())

::

             age_start        age_end      value
    0             0.00           0.01  87.885872
    1             0.01           0.02  87.877086
    2             0.02           0.03  87.868299
    3             0.03           0.04  87.859513
    4             0.04           0.05  87.850727


Age Bin Data
++++++++++++
To see what age bins GBD uses that are used in age-specific data, :mod:`vivarium_inputs`
provides :func:`vivarium_inputs.interface.get_age_bins`, which returns the start,
end, and name of each GBD age bin expected to appear in age-specific data (with
the exception of life expectancy, which uses its own age ranges).

.. code-block:: python

    from vivarium_inputs import get_age_bins

    age_bins = get_age_bins()
    print(age_bins.head())

::

             age_start        age_end  age_group_name
    0         0.000000       0.019178  Early Neonatal
    1         0.019178       0.076712   Late Neonatal
    2         0.076712       1.000000   Post Neonatal
    3         1.000000       5.000000          1 to 4
    4         5.000000      10.000000          5 to 9


Demographic Dimensions Data
+++++++++++++++++++++++++++
Finally, to view the full extent of all demographic dimensions that is expected
in input data to the simulation,  :mod:`vivarium_inputs` provides
:func:`vivarium_inputs.interface.get_demographic_dimensions`, which expects a `location`
argument to fill the location dimension.

.. code-block:: python

    from vivarium_inputs import get_demographic_dimensions

    dem_dims = get_demographic_dimensions('Kenya')
    print(dem_dims.head())

::

      location     sex        age_start        age_end  year_start  year_end
    0    Kenya  Female              0.0       0.019178        1990      1991
    1    Kenya  Female              0.0       0.019178        1991      1992
    2    Kenya  Female              0.0       0.019178        1992      1993
    3    Kenya  Female              0.0       0.019178        1993      1994
    4    Kenya  Female              0.0       0.019178        1994      1995


Pulling Raw GBD Data
--------------------
The interface provides :func:`vivarium_inputs.interface.get_raw_data`, which can
be used to pull entity-measure data as well as population structure and life
expectancy. Raw validation checks are not performed to return data that can
be investigated for oddities. The only filtering that occurs is by applicable
measure id, metric id, or to most detailed causes where relevant. No formatting
or reshaping of the data is done. The following sections detail how to pull each
type of data.


Entity-Measure Data
+++++++++++++++++++
The interface provides :func:`vivarium_inputs.interface.get_raw_data` for pulling
specific raw measure data for an entity for a single location from GBD, without
the prep work that occurs on data for a simulation.

``entity`` should be a :class:`gbd_mapping.base_template.ModelableEntity` (e.g.,
a cause from ``gbd_mapping``), while ``measure`` should be a string
describing the measure for which you want to retrieve data (e.g., 'prevalence'
or 'relative_risk'). A list of possible measures for each entity
kind is included in the table below. Finally, ``location`` should be the string
location for which you want to pull data (e.g., 'Ethiopia'), in the form used by
GBD (e.g., 'United States' instead of 'USA').

For example, to pull raw prevalence data for diarrheal diseases in Kenya, we would
do the following:

.. code-block:: python

    from gbd_mapping import causes
    from vivarium_inputs import get_raw_data

    prev = get_raw_data(causes.diarrheal_diseases, 'prevalence', 'Kenya')
    print(prev.head())

::

          year_id  age_group_id  sex_id  measure_id  cause_id    draw_0    ...      draw_999  location_id  metric_id
    1288     1990             2       1           5       302  0.030940    ...      0.029214          180          3
    1289     1990             3       1           5       302  0.063305    ...      0.059538          180          3
    1290     1990             4       1           5       302  0.056916    ...      0.058788          180          3
    1291     1990             5       1           5       302  0.026376    ...      0.035843          180          3
    1292     1990             6       1           5       302  0.011728    ...      0.011231          180          3


The following table lists the measures available for each entity kind for pulling raw data:

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
          | deaths
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
+++++++++++++++++++++++++
To pull raw population data for a specific location, we will actually use the same
:func:`vivarium_inputs.interface.get_raw_data` function we used for pulling
entity-measure data, with a special Population entity.

For example, to pull population data for Kenya, we would do the following:

.. code-block:: python

    from vivarium_inputs import get_raw_data
    from vivarium_inputs.globals import Population

    pop = get_raw_data(Population(), 'structure', 'Kenya')
    print(pop.head())

::

       age_group_id  location_id  year_id  sex_id   population  run_id
    0             2          180     1950       1  2747.467163     117
    1             2          180     1950       2  2484.512754     117
    2             2          180     1950       3  5231.979917     117
    3             2          180     1951       1  3146.320799     117
    4             2          180     1951       2  3038.538221     117


Life Expectancy Data
++++++++++++++++++++
Similarly to pull life expectancy data, we will use the same
:func:`vivarium_inputs.interface.get_raw_data` function with the special Population
entity. Life expectancy data is not location-specific, so we'll just use the
'Global' location.

To use:

.. code-block:: python

    from vivarium_inputs import get_raw_data
    from vivarium_inputs.globals import Population

    life_exp = get_raw_data(Population(), 'theoretical_minimum_risk_life_expectancy', 'Global')
    print(life_exp.head())

::

        age  life_expectancy
    0  0.00        87.885872
    1  0.01        87.877086
    2  0.02        87.868299
    3  0.03        87.859513
    4  0.04        87.850727


.. testcode::
    :hide:

    import inspect
    from typing import Union

    import pandas as pd

    from vivarium_inputs import (get_measure, get_population_structure,
                                 get_theoretical_minimum_risk_life_expectancy,
                                 get_age_bins, get_demographic_dimensions, get_raw_data)
    from gbd_mapping import ModelableEntity

    funcs = {get_measure: {
                'parameters': {'entity': ModelableEntity, 'measure': str, 'location': str},
                'return': pd.DataFrame, },
             get_population_structure: {
                 'parameters': {'location': str},
                 'return': pd.DataFrame, },
             get_theoretical_minimum_risk_life_expectancy: {
                 'parameters': {},
                 'return': pd.DataFrame, },
             get_age_bins: {
                 'parameters': {},
                 'return': pd.DataFrame, },
             get_demographic_dimensions: {
                 'parameters': {'location': str},
                 'return': pd.DataFrame, },
             get_raw_data: {
                 'parameters': {'entity': ModelableEntity, 'measure': str, 'location': str},
                'return': Union[pd.DataFrame, pd.Series], },
             }
    for func, spec in funcs.items():
        sig = inspect.signature(func)
        assert len(sig.parameters) == len(spec['parameters'])
        for name, annotation in spec['parameters'].items():
            assert name in sig.parameters
            assert sig.parameters[name].annotation == annotation
        assert sig.return_annotation == spec['return']


