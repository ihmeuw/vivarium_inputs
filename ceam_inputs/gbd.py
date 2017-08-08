from multiprocessing.process import current_process
import warnings

from joblib import Memory
import pandas as pd

from ceam_inputs.util import get_cache_directory
from ceam_inputs.auxiliary_files import auxiliary_file_path
from ceam_inputs.gbd_mapping import cid

memory = Memory(cachedir=get_cache_directory(), verbose=1)

MALE = [1]
FEMALE = [2]
COMBINED = [3]

ZERO_TO_EIGHTY = list(range(2, 21))
EIGHTY_PLUS = [21]


class CentralCompError(Exception):
    """Error for failures in central-comp tooling."""
    pass


class DataNotFoundError(CentralCompError):
    """Raised when the set of parameters passed to central comp functions fails due to lack of data."""
    pass


def _get_draws_safely(draw_function, draw_options, *args, **kwargs):
    measure_draws = None
    for location_id, round_id in draw_options:
        try:
            measure_draws = draw_function(*args, location_ids=location_id, gbd_round_id=round_id, **kwargs)
            break
        except:
            pass
    if measure_draws is None:
        raise DataNotFoundError("Couldn't find draws for your requirements\n"
                                + "function : {}\ndraw_options :  {}\n".format(draw_function.__name__, draw_options)
                                + "args : {}\nkwargs : {}.".format(args, kwargs))
    return measure_draws

@memory.cache
def get_gbd_tool_versions(publication_ids, source):
    from db_tools import ezfuncs
    #NOTE: this mapping comes from the gbd.metadata_type table but in that
    #      database it isn't in a form that's convenient to query and these
    #      ids should be stable so I'm sticking it here -Alec
    metadata_type_id = {
            'codcorrect': 1,
            'como': 4
    }[source]

    como_ids = ezfuncs.query("""
        SELECT distinct val
        FROM gbd.gbd_process_version_metadata
        JOIN gbd.gbd_process_version_publication USING (gbd_process_version_id)
        WHERE metadata_type_id = {} and publication_id in ({})
        """.format(metadata_type_id, ','.join([str(pid) for pid in publication_ids])), conn_def='gbd')
    if como_ids.empty:
        warnings.warn('No version id found for {} with publications {}. This likely indicates missing entries in the GBD database.'.format(source, publication_ids))
        return None
    return como_ids['val'].astype('int').tolist()

@memory.cache
def get_dismod_model_versions(publication_ids):
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
        WHERE age_group_id IN ({})
        """.format(','.join([str(a) for a in ZERO_TO_EIGHTY + EIGHTY_PLUS])), conn_def='shared')


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


def get_gbd_draws(location_id, gbd_id, publication_ids=None, gbd_round_id=4):
    if isinstance(gbd_id, cid):
        return get_como_draws(location_id, gbd_id, publication_ids, gbd_round_id)
    else:
        return get_modelable_entity_draws(location_id, gbd_id, publication_ids, gbd_round_id)

@memory.cache
def get_modelable_entity_draws(location_id, me_id, publication_ids=None, gbd_round_id=4):
    from transmogrifier.draw_ops import get_draws
    model_version = get_dismod_model_versions(publication_ids)[me_id] if publication_ids else None
    gbd_round_id = gbd_round_id if gbd_round_id else 4

    return get_draws(gbd_id_field='modelable_entity_id',
                     gbd_id=me_id,
                     source="dismod",
                     location_ids=location_id,
                     sex_ids=MALE + FEMALE,
                     age_group_ids=ZERO_TO_EIGHTY + EIGHTY_PLUS,
                     version_id=model_version,
                     gbd_round_id=gbd_round_id)


@memory.cache
def get_codcorrect_draws(location_id, cause_id, publication_ids=None, gbd_round_id=4):
    from transmogrifier.draw_ops import get_draws
    versions = get_gbd_tool_versions(publication_ids, 'codcorrect') if publication_ids else None

    # FIXME: Should submit a ticket to IT to determine if we need to specify an
    # output_version_id or a model_version_id to ensure we're getting the correct results
    return get_draws(gbd_id_field='cause_id',
                     gbd_id=cause_id,
                     source="codcorrect",
                     location_ids=location_id,
                     sex_ids=MALE + FEMALE,
                     age_group_ids=ZERO_TO_EIGHTY + EIGHTY_PLUS,
                     version_id=versions,
                     gbd_round_id=gbd_round_id)

@memory.cache
def get_como_draws(location_id, cause_id, publication_ids=None, gbd_round_id=4):
    from transmogrifier.draw_ops import get_draws
    versions = get_gbd_tool_versions(publication_ids, 'como') if publication_ids else None

    # FIXME: Should submit a ticket to IT to determine if we need to specify an
    # output_version_id or a model_version_id to ensure we're getting the correct results
    return get_draws(gbd_id_field='cause_id',
                     gbd_id=cause_id,
                     source="como",
                     location_ids=location_id,
                     sex_ids=MALE + FEMALE,
                     age_group_ids=ZERO_TO_EIGHTY + EIGHTY_PLUS,
                     version_id=versions,
                     gbd_round_id=gbd_round_id)


@memory.cache
def get_covariate_estimates(covariate_name_short, location_id):
    from db_queries import get_covariate_estimates

    covariate_estimates = get_covariate_estimates(covariate_name_short=covariate_name_short,
                                                  covariate_id=None,
                                                  location_id=location_id,
                                                  sex_id=MALE + FEMALE + COMBINED,
                                                  age_group_id=-1,
                                                  model_version_id=None)
    return covariate_estimates


@memory.cache
def get_relative_risks(location_id, risk_id, gbd_round_id):
    from transmogrifier.draw_ops import get_draws

    # Some RRs are only reported at the global level (1). Also, sometimes draws from a previous gbd round are reused.
    global_location_id = 1
    draw_options = [[location_id, gbd_round_id], [global_location_id, gbd_round_id],
                    [location_id, gbd_round_id-1], [global_location_id, gbd_round_id-1]]
    return _get_draws_safely(get_draws, draw_options,
                             gbd_id_field='rei_id',
                             gbd_id=risk_id,
                             source='risk',
                             sex_ids=MALE + FEMALE,
                             age_group_ids=ZERO_TO_EIGHTY + EIGHTY_PLUS,
                             draw_type='rr')

@memory.cache
def get_exposures(location_id, risk_id, gbd_round_id):
    from transmogrifier.draw_ops import get_draws

    # Sometimes draws from a previous gbd round are reused.
    draw_options = [[location_id, gbd_round_id], [location_id, gbd_round_id - 1]]
    return _get_draws_safely(get_draws, draw_options,
                             gbd_id_field='rei_id',
                             gbd_id=risk_id,
                             source='risk',
                             sex_ids=MALE + FEMALE,
                             age_group_ids=ZERO_TO_EIGHTY + EIGHTY_PLUS,
                             draw_type='exposure')


@memory.cache
def get_pafs(location_id, cause_id, gbd_round_id):
    from transmogrifier.draw_ops import get_draws

    # I'm cargo culting here. When the simulation is hosted by a dask worker,
    # we can't spawn sub-processes in the way that get_draws wants to
    # There are better ways of solving this but they involve understanding dask
    # better or working on shared function code, neither of
    # which I'm going to do right now. -Alec
    worker_count = 0 if current_process().daemon else 6  # One worker per 5-year dalynator file (1990 - 2015)

    # Sometimes draws from a previous gbd round are reused.
    draw_options = [[location_id, gbd_round_id], [location_id, gbd_round_id - 1]]
    return _get_draws_safely(get_draws, draw_options,
                             gbd_id_field='cause_id',
                             gbd_id=cause_id,
                             source='dalynator',
                             sex_ids=MALE + FEMALE,
                             age_group_ids=ZERO_TO_EIGHTY + EIGHTY_PLUS,
                             include_risks=True,
                             num_workers=worker_count)

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

    # I'm cargo culting here. When the simulation is hosted by a dask worker,
    # we can't spawn sub-processes in the way that get_draws wants to
    # There are better ways of solving this but they involve understanding dask
    # better or working on shared function code, neither of
    # which I'm going to do right now. -Alec
    worker_count = 0 if current_process().daemon else 6  # One worker per 5-year dalynator file (1990 - 2015)

    return get_draws(gbd_id_field="cause_id",
                     gbd_id=294,
                     source="dalynator",
                     age_group_ids=ZERO_TO_EIGHTY + EIGHTY_PLUS,
                     location_ids=location_id,
                     measure_ids=1,
                     gbd_round_id=gbd_round_id,
                     num_workers=worker_count)


@memory.cache
def get_data_from_auxiliary_file(file_name, **kwargs):
    path, encoding = auxiliary_file_path(file_name, **kwargs)
    file_type = path.split('.')[-1]
    if file_type == 'csv':
        data = pd.read_csv(path, encoding=encoding)
    elif file_type == 'h5':
        data = pd.read_hdf(path, encoding=encoding)
    elif file_type == 'dta':
        data = pd.read_stata(path, encoding=encoding)
    else:
        raise NotImplementedError("File type {} is not supported".format(file_type))
    return data
