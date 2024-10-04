============
Pulling Data
============


.. testsetup::

    import inspect

    from vivarium_inputs import (
        get_measure,
        get_population_structure,
        get_theoretical_minimum_risk_life_expectancy,
        get_age_bins,
        get_demographic_dimensions,
        get_raw_data,
    )

    def check_args(func, used_args):
        """Test that the above code-block is valid given that we don't guaranteed 
        access to vivarium_gbd_access
        """

        # TODO: Test that the argument and return types are as expected for each
        # function. We can't easily do this now because we are supporting py3.9
        # which doesn't allow for type hints using | (instead of Union) and
        # we've implemented a bit of that.
        
        actual_args = inspect.signature(func).parameters
        
        # Check that all arguments passed in actually exist in the signature
        assert not set(used_args).difference(set(actual_args))

        # Check that any args not passed in actually have a default in the signature
        check_for_default = [arg for arg in actual_args if arg not in used_args]
        assert all(actual_args[arg].default != inspect._empty for arg in check_for_default)


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

.. note::

    The data returned by the interface may occasionally change format as 
    :mod:`vivarium_inputs` is updated and/or the actual underlying GBD data 
    changes. Therefore, the examples we provide below may not exactly match 
    what you see.

Which should I use... get_measure() versus get_raw_data()
---------------------------------------------------------
Typically, you should prefer :func:`get_measure <vivarium_inputs.interface.get_measure>` over
:func:`get_raw_data <vivarium_inputs.interface.get_raw_data>`. 
:func:`get_measure <vivarium_inputs.interface.get_measure>` will produce 
simulation-prepped data. If :func:`get_measure <vivarium_inputs.interface.get_measure>`
fails, or the data it returns doesn't match your expectations, then
:func:`get_raw_data <vivarium_inputs.interface.get_raw_data>` might provide some 
insight into what is happening.

Pulling Simulation-Prepped Data
-------------------------------
For simulation-prepped data, the interface provides separate methods to pull
entity-measure data, population structure, and life expectancy data. Additionally,
methods to pull age bin data and demographic dimensions are provided. 
Simulation-prepped data has had GBD IDs replaced with meaningful values or ranges and
expansion over all demographic dimensions has been performed. We'll walk
through how to pull data using each of these functions.

Entity-Measure Data
+++++++++++++++++++
The interface provides :func:`get_measure <vivarium_inputs.interface.get_measure>` 
for pulling location-specific measure data for an entity (e.g. a cause from 
``gbd_mapping``). The `measure` is the descriptor of the data you want to pull 
(e.g., 'prevalence' or 'relative_risk') - a list of possible measures for each entity
type is included in the table below.

.. note::

    To pull simulation-prepped entity-measure data, you must have plenty of 
    available memory - please request at least 50GB.

.. note::

    The simulation-prepped data returned by :func:`get_measure <vivarium_inputs.interface.get_measure>`
    has all demographic and year values set as the index with only draw-level
    data as columns.

For example, to pull prevalence data for diarrheal diseases in Kenya, we would
do the following:

.. code-block:: python

    from gbd_mapping import causes
    from vivarium_inputs import get_measure

    prev = get_measure(
        entity=causes.diarrheal_diseases,
        measure='prevalence',
        location='Kenya',
        data_type="draws",
    )
    print(prev.head())

::

                                                            draw_0  ...  draw_499
    location sex    age_start age_end  year_start year_end            ...          
    Kenya    Female 0.000000  0.019178 2021       2022      0.018762  ...  0.018243
                    0.019178  0.076712 2021       2022      0.041142  ...  0.041379
                    0.076712  0.500000 2021       2022      0.040640  ...  0.042404
                    0.500000  1.000000 2021       2022      0.026530  ...  0.029795
                    1.000000  2.000000 2021       2022      0.011624  ...  0.014232

.. testcode::
    :hide:
    
    check_args(get_measure, ["entity", "measure", "location", "data_type"])


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
provides :func:`get_population_structure <vivarium_inputs.interface.get_population_structure>`, 
which returns population data in the input format expected by a simulation.

For example, to pull population data for Kenya, we would do the following:

.. code-block:: python

    from vivarium_inputs import get_population_structure

    pop = get_population_structure(location='Kenya')
    print(pop.head())

::

                                                                    value
    location sex    age_start age_end  year_start year_end               
    Kenya    Female 0.000000  0.019178 2021       2022       10995.345135
                    0.019178  0.076712 2021       2022       32740.129897
                    0.076712  0.500000 2021       2022      241157.325386
                    0.500000  1.000000 2021       2022      283195.389282
                    1.000000  2.000000 2021       2022      575233.481802

.. testcode::
    :hide:

    check_args(get_population_structure, ["location"])


Life Expectancy Data
++++++++++++++++++++
To pull life expectancy data, :mod:`vivarium_inputs.interface` provides 
:func:`get_theoretical_minimum_risk_life_expectancy <vivarium_inputs.interface.get_theoretical_minimum_risk_life_expectancy>`,
which returns life expectancy data in the input format expected by a simulation.
Because life expectancy is not location specific, the function takes no arguments.

To use:

.. code-block:: python

    from vivarium_inputs import get_theoretical_minimum_risk_life_expectancy

    life_exp = get_theoretical_minimum_risk_life_expectancy()
    print(life_exp.head())

::

                           value
    age_start age_end           
    0.00      0.01     89.958040
    0.01      0.02     89.975474
    0.02      0.03     89.990990
    0.03      0.04     89.985077
    0.04      0.05     89.979164

.. testcode::
    :hide:

    check_args(get_theoretical_minimum_risk_life_expectancy, [])


Age Bin Data
++++++++++++
To see what age bins GBD uses that are used in age-specific data, :mod:`vivarium_inputs`
provides :func:`get_age_bins <vivarium_inputs.interface.get_age_bins>`, which returns 
the start, end, and name of each GBD age bin expected to appear in age-specific data 
(with the exception of life expectancy, which uses its own age ranges).

.. code-block:: python

    from vivarium_inputs import get_age_bins

    age_bins = get_age_bins()
    print(age_bins.reset_index().head())

::

       age_start   age_end   age_group_name
    0   0.000000  0.019178   Early Neonatal
    1   0.019178  0.076712    Late Neonatal
    2   0.076712  0.500000       1-5 months
    3   0.500000  1.000000      6-11 months
    4   1.000000  2.000000  12 to 23 months

.. testcode::
    :hide:

    check_args(get_age_bins, [])


Demographic Dimensions Data
+++++++++++++++++++++++++++
Finally, to view the full extent of all demographic dimensions that is expected
in input data to the simulation, :mod:`vivarium_inputs` provides
:func:`get_demographic_dimensions <vivarium_inputs.interface.get_demographic_dimensions>`, 
which expects a `location` argument to fill the location dimension.

.. code-block:: python

    from vivarium_inputs import get_demographic_dimensions

    dem_dims = get_demographic_dimensions(location='Kenya')
    print(dem_dims.reset_index().head())

::

      location     sex  age_start   age_end  year_start  year_end
    0    Kenya  Female   0.000000  0.019178        2021      2022
    1    Kenya  Female   0.019178  0.076712        2021      2022
    2    Kenya  Female   0.076712  0.500000        2021      2022
    3    Kenya  Female   0.500000  1.000000        2021      2022
    4    Kenya  Female   1.000000  2.000000        2021      2022

.. testcode::
    :hide:

    check_args(get_demographic_dimensions, ["location"])


Pulling Raw GBD Data
--------------------
The interface provides :func:`get_raw_data <vivarium_inputs.interface.get_raw_data>`, 
which can be used to pull entity-measure data as well as population structure and life
expectancy. Raw validation checks are not performed to return data that can
be investigated for oddities. The only filtering that occurs is by applicable
measure id, metric id, or to most detailed causes where relevant. No formatting
or reshaping of the data is done. The following sections detail how to pull each
type of data.


Entity-Measure Data
+++++++++++++++++++
The interface provides :func:`get_raw_data <vivarium_inputs.interface.get_raw_data>` 
for pulling specific raw measure data for an entity for a single location from GBD, 
without the prep work that occurs on data for a simulation.

``entity`` should be a :class:`gbd_mapping.base_template.ModelableEntity` (e.g.,
a cause from ``gbd_mapping``), while ``measure`` should be a string
describing the measure for which you want to retrieve data (e.g., 'prevalence'
or 'relative_risk'). A list of possible measures for each entity
kind is included in the table below. Finally, ``location`` should be the string
location for which you want to pull data (e.g., 'Ethiopia'), in the form used by
GBD (e.g., 'United States' instead of 'USA').

For example, to pull draw-level raw prevalence data for diarrheal diseases in Kenya, we would
do the following:

.. code-block:: python

    from gbd_mapping import causes
    from vivarium_inputs import get_raw_data

    prev = get_raw_data(
        entity=causes.diarrheal_diseases,
        measure='prevalence',
        location='Kenya',
        data_type="draws",
    )
    print(prev.head())

::

        age_group_id  cause_id    draw_0  ...  year_id  metric_id  version_id
    50             2       302  0.018762  ...     2021          3        1471
    51             3       302  0.041142  ...     2021          3        1471
    52             6       302  0.014616  ...     2021          3        1471
    53             7       302  0.023237  ...     2021          3        1471
    54             8       302  0.024702  ...     2021          3        1471

.. testcode::
    :hide:

    check_args(get_raw_data, ["entity", "measure", "location", "data_type"])

    
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
:func:`get_raw_data <vivarium_inputs.interface.get_raw_data>` function we used for 
pulling entity-measure data, with a special Population entity.

For example, to pull population data for Kenya, we would do the following:

.. code-block:: python

    from vivarium_inputs import get_raw_data
    from vivarium_inputs.globals import Population

    pop = get_raw_data(entity=Population(), measure='structure', location='Kenya')
    print(pop.head())

::

       age_group_id  location_id  year_id  sex_id    population  run_id
    0             2          180     2021       1  1.145138e+04     359
    1             3          180     2021       1  3.402961e+04     359
    2             6          180     2021       1  3.187225e+06     359
    3             7          180     2021       1  3.264795e+06     359
    4             8          180     2021       1  2.997167e+06     359

.. testcode::
    :hide:

    check_args(get_raw_data, ["entity", "measure", "location"])


Life Expectancy Data
++++++++++++++++++++
Similarly to pull life expectancy data, we will use the same
:func:`get_raw_data <vivarium_inputs.interface.get_raw_data>` function with the 
special Population entity. Life expectancy data is not location-specific, so we'll 
just use the 'Global' location.

To use:

.. code-block:: python

    from vivarium_inputs import get_raw_data
    from vivarium_inputs.globals import Population

    life_exp = get_raw_data(Population(), 'theoretical_minimum_risk_life_expectancy', 'Global')
    print(life_exp.head())

::

        age  life_expectancy
    0  0.00        89.958040
    1  0.01        89.975474
    2  0.02        89.990990
    3  0.03        89.985077
    4  0.04        89.979164

.. testcode::
    :hide:

    check_args(get_raw_data, ["entity", "measure", "location"])

