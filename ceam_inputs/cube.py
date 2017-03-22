from warnings import warn

import pandas as pd

from ceam import config

import ceam_inputs as inp
import ceam_inputs.gbd_ms_functions as gfuncs
from ceam_inputs.util import gbd_year_range
from ceam_inputs.gbd_mapping import causes

function_map = {
        'excess_mortality': inp.get_excess_mortality,
        'prevalence': inp.get_prevalence,
        'mortality': inp.get_cause_specific_mortality,
        'disability_weight': gfuncs.get_disability_weight,
}

def make_input_cube_from_gbd(year_start, year_end, locations, draws, measures):
    # TODO: This fiddling of the config is awkward but it's necessary
    # unless we re-architect the existing ceam_input functions.
    old_year_start = config.get('simulation_parameters', 'year_start')
    old_year_end = config.get('simulation_parameters', 'year_end')
    old_location = config.get('simulation_parameters', 'location_id')
    old_draw = config.get('run_configuration', 'draw_number')
    config.set('simulation_parameters', 'year_start', str(year_start))
    config.set('simulation_parameters', 'year_end', str(year_end))
    cube = pd.DataFrame(columns=['year', 'age', 'sex', 'measure', 'cause', 'draw', 'value'])
    for location in locations:
        config.set('simulation_parameters', 'location_id', str(location))
        for draw in draws:
            config.set('run_configuration', 'draw_number', str(draw))
            for cause, measure in measures:
                if cause == 'all' and measure == 'mortality':
                    data = inp.get_cause_deleted_mortality_rate({})
                else:
                    if cause in causes and measure in causes[cause]:
                        data = function_map[measure](causes[cause][measure])
                    else:
                        warn("Trying to load input for {}.{} but no mapping was present".format(cause, measure))
                        continue
                value_column = [c for c in data.columns if c not in ['age', 'sex', 'year']]
                assert len(value_column) == 1
                value_column = value_column[0]
                data = data.rename(columns={value_column: 'value'})
                data['draw'] = draw
                data['measure'] = measure
                data['cause'] = cause
                data['location'] = location
                cube = cube.append(data)

    config.set('simulation_parameters', 'year_start', old_year_start)
    config.set('simulation_parameters', 'year_end', old_year_end)
    config.set('simulation_parameters', 'location_id', old_location)
    config.set('run_configuration', 'draw_number', old_draw)

    return cube.set_index(['year', 'age', 'sex', 'measure', 'cause', 'draw', 'location'])

if __name__ == '__main__':
    print(make_input_cube_from_gbd(1990, 1990, [180], [0], [('heart_attack', 'excess_mortality'), ('mild_angina', 'prevalence'), ('all', 'mortality')]))
