from joblib import Memory

import numpy as np
import pandas as pd

from ceam_inputs.util import get_cache_directory
from ceam_inputs.auxiliary_files import open_auxiliary_file

memory = Memory(cachedir=get_cache_directory(), verbose=1)

MALE = [1]
FEMALE = [2]
COMBINED = [3]

ZERO_TO_EIGHTY = list(range(2, 21))
EIGHTY_PLUS = [21]


@memory.cache
def get_model_versions(publication_ids):
    from db_tools import ezfuncs

    mapping = ezfuncs.query("""
        SELECT modelable_entity_id, 
               model_version_id
        FROM epi.publication_model_version
        JOIN epi.model_version USING (model_version_id)
        JOIN shared.publication USING (publication_id)
        WHERE publication_id in ({})
        """.format(','.join([str(pid) for pid in publication_ids])), conn_def='epi')

    return dict(mapping[['modelable_entity_id', 'model_version_id']].values)


@memory.cache
def get_age_bins():
    from db_tools import ezfuncs
    return ezfuncs.query("""
        SELECT age_group_id, 
               age_group_years_start, 
               age_group_years_end, 
               age_group_name 
        FROM age_group
        """, conn_def='shared')


@memory.cache
def get_healthstate_id(me_id):
    from db_tools import ezfuncs

    healthstate_id_df = ezfuncs.query("""    
    SELECT modelable_entity_id, 
           healthstate_id
    FROM epi.sequela_hierarchy_history
    WHERE modelable_entity_id = {}
    """.format(int(me_id)), conn_def='epi')

    if healthstate_id_df.empty:
        raise ValueError("Modelable entity id {} does not have a healthstate id.".format(me_id)
                         + "There is not a disability weight associated with this sequela,"
                         + "so you should not try to pull draws for it")

    healthstate_id = healthstate_id_df.at[0, 'healthstate_id']

    return healthstate_id


@memory.cache
def get_subregions(location_id):
    from hierarchies import dbtrees
    return [c.id for c in dbtrees.loctree(None, location_set_id=2).get_node_by_id(location_id).children]


@memory.cache
def get_modelable_entity_draws(location_id, me_id, publication_ids=None, gbd_round_id=None):
    from transmogrifier.draw_ops import get_draws
    model_version = get_model_versions(publication_ids)[me_id] if publication_ids else None
    gbd_round_id = gbd_round_id if gbd_round_id else 4

    return get_draws(gbd_id_field='modelable_entity_id',
                     gbd_id=me_id,
                     source="dismod",
                     location_ids=location_id,
                     sex_ids=MALE + FEMALE,
                     age_group_ids=ZERO_TO_EIGHTY + EIGHTY_PLUS,
                     model_version_id=model_version,
                     gbd_round_id=gbd_round_id)


@memory.cache
def get_covariate_estimates(covariate_name_short, location_id):
    # This import is not at global scope because I only want the dependency if cached data is unavailable
    from db_queries import get_covariate_estimates

    covariate_estimates = get_covariate_estimates(covariate_name_short=covariate_name_short,
                                                  covariate_id=None,
                                                  location_id=location_id,
                                                  sex_id=MALE + FEMALE + COMBINED,
                                                  age_group_id=-1,
                                                  model_version_id=None)
    return covariate_estimates


@memory.cache
def _get_risk_draws(location_id, risk_id, draw_type, gbd_round_id):
    from transmogrifier.draw_ops import get_draws
    draws = get_draws(gbd_id_field='rei_id',
                      gbd_id=risk_id,
                      source='risk',
                      location_ids=location_id,
                      sex_ids=MALE + FEMALE,
                      age_group_ids=ZERO_TO_EIGHTY + EIGHTY_PLUS,
                      draw_type=draw_type,
                      gbd_round_id=gbd_round_id)
    if np.all(draws.values == "error"):
        draws = get_draws(gbd_id_field='rei_id',
                          gbd_id=risk_id,
                          source='risk',
                          location_ids=location_id,
                          sex_ids=MALE + FEMALE,
                          age_group_ids=ZERO_TO_EIGHTY + EIGHTY_PLUS,
                          draw_type=draw_type,
                          gbd_round_id=gbd_round_id-1)
    elif np.any(draws.values == "error"):
        raise ValueError("Get draws failed for some rows but not all. It is unclear how to proceed so stopping")
    return draws


def get_relative_risks(location_id, risk_id, gbd_round_id):
    return _get_risk_draws(location_id=location_id,
                           risk_id=risk_id,
                           draw_type='err',
                           gbd_round_id=gbd_round_id)


def get_exposures(location_id, risk_id, gbd_round_id):
    return _get_risk_draws(location_id=location_id,
                           risk_id=risk_id,
                           draw_type='exposure',
                           gbd_round_id=gbd_round_id)


@memory.cache
def get_pafs(location_id, cause_id, gbd_round_id):
    from transmogrifier.draw_ops import get_draws
    return get_draws(gbd_id_field='cause_id',
                     gbd_id=cause_id,
                     source='dalynator',
                     location_ids=location_id,
                     sex_ids=MALE + FEMALE,
                     age_group_ids=ZERO_TO_EIGHTY + EIGHTY_PLUS,
                     include_risks=True,
                     gbd_round_id=gbd_round_id)


@memory.cache
def get_populations(location_id, gbd_round_id):
    from db_queries import get_population

    return get_population(age_group_id=ZERO_TO_EIGHTY + EIGHTY_PLUS,
                          location_id=location_id,
                          year_id=[-1],
                          sex_id=MALE + FEMALE + COMBINED,
                          gbd_round_id=gbd_round_id)


@memory.cache
def get_deaths(location_id, gbd_round_id):
    from transmogrifier.draw_ops import get_draws
    return get_draws(gbd_id_field="cause_id",
                     gbd_id=294,
                     source="dalynator",
                     age_group_ids=ZERO_TO_EIGHTY + EIGHTY_PLUS,
                     location_ids=location_id,
                     measure_ids=1,
                     gbd_round_id=gbd_round_id)


@memory.cache
def get_data_from_auxiliary_file(file_name, **kwargs):
    with open_auxiliary_file(file_name, **kwargs) as f:
        file_type = file_name.split('.')[-1]
        if file_type == 'csv':
            data = pd.read_csv(f)
        elif file_type == 'h5':
            data = pd.read_hdf(f)
    return data





