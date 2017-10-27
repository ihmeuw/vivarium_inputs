from multiprocessing.process import current_process
import warnings
from typing import Iterable, Union, List, Dict, Callable, Any, Mapping

from joblib import Memory
import pandas as pd

from ceam_inputs.util import get_cache_directory, get_input_config
from ceam_inputs.auxiliary_files import auxiliary_file_path

memory = Memory(cachedir=get_cache_directory(get_input_config()), verbose=1)

# Define GBD sex ids for usage with central comp tools.
MALE = [1]
FEMALE = [2]
COMBINED = [3]





class CentralCompError(Exception):
    """Error for failures in central-comp tooling."""
    pass


class DataNotFoundError(CentralCompError):
    """Raised when the set of parameters passed to central comp functions fails due to lack of data."""
    pass


#######################################################
# Tools for pulling version ids for particular models #
#######################################################

@memory.cache
def get_gbd_tool_version(publication_ids: Iterable[int], source: str) -> Union[int, None]:
    """Grabs the version id for codcorrect and como draws."""
    from db_tools import ezfuncs
    # NOTE: this mapping comes from the gbd.metadata_type table but in that
    #       database it isn't in a form that's convenient to query and these
    #       ids should be stable so I'm sticking it here -Alec
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
        warnings.warn(f'No version id found for {source} with publications {publication_ids}. '
                      'This likely indicates missing entries in the GBD database.')
        return None
    return como_ids['val'].astype('int')[0]


@memory.cache
def get_dismod_model_version(me_id: int, publication_ids: Union[Iterable[int], None]) -> Union[int, None]:
    """Grabs the model version ids for dismod draws."""
    try:
        from db_tools import ezfuncs
        mapping = ezfuncs.query(f"""
            SELECT modelable_entity_id, 
                   model_version_id
            FROM epi.publication_model_version
            JOIN epi.model_version USING (model_version_id)
            JOIN shared.publication USING (publication_id)
            WHERE publication_id in ({','.join([str(pid) for pid in publication_ids])})
            """, conn_def='epi')
        return dict(mapping[['modelable_entity_id', 'model_version_id']].values)[me_id]
    except (TypeError, KeyError):
        pass

    warnings.warn(f'publication_ids supplied to get_modelable_entity_draws but me_id {me_id} does map to any version '
                  'associated with those publications. That likely means there is a mapping missing in the database')
    return None


@memory.cache
def get_sequela_set_version_id(gbd_round_id: int) -> int:
    """Grabs the sequela set version associated with a particular gbd round."""
    from db_tools import ezfuncs
    q = """
        SELECT gbd_round_id,
               sequela_set_version_id
        FROM epi.sequela_set_version_active
        """
    return ezfuncs.query(q, conn_def='epi').set_index('gbd_round_id').at[gbd_round_id, 'sequela_set_version_id']


@memory.cache
def get_cause_risk_set_version_id(gbd_round_id: int) -> int:
    """Grabs the version id associated with a cause risk mapping for a particular gbd round."""
    from db_tools import ezfuncs
    q = f"""
         SELECT cause_risk_set_version_id
         FROM shared.cause_risk_set_version_active
         WHERE gbd_round_id = {gbd_round_id}
         """
    return ezfuncs.query(q, conn_def='epi').at[0, 'cause_risk_set_version_id']


####################################
# Mappings between various gbd ids #
####################################

@memory.cache
def get_sequela_id_mapping(model_version: int) -> pd.DataFrame:
    """Grabs a mapping between sequelae ids and their associated names, me_ids, cause_ids, and healthstate_ids."""
    from db_tools import ezfuncs

    q = f"""
        SELECT sequela_id,
               sequela_name,
               modelable_entity_id,
               cause_id, 
               healthstate_id
        FROM epi.sequela_hierarchy_history
        WHERE sequela_set_version_id = {model_version}"""
    return ezfuncs.query(q, conn_def='epi')


@memory.cache
def get_cause_etiology_mapping(gbd_round_id: int) -> pd.DataFrame:
    """Get a mapping between the diarrhea and lri cause ids and the rei_ids associated with their etiologies."""
    from db_tools import ezfuncs
    # FIXME: This table has not been updated with the round 4 mapping, but Joe Wagner assures
    # me that the mapping did not change from round 3.  He's going to update the table, then we
    # should remove this.
    if gbd_round_id == 4:
        gbd_round_id = 3

    q = f""" 
        SELECT rei_id,
               cause_id
        FROM shared.cause_etiology
        WHERE gbd_round_id = {gbd_round_id}
        AND cause_id in (302, 322)
        """
    return ezfuncs.query(q, conn_def='epi')


@memory.cache
def get_cause_risk_mapping(cause_risk_set_version_id: int) -> pd.DataFrame:
    """Get a mapping between risk ids and cause ids for a particular gbd round."""
    from db_tools import ezfuncs

    q = f"""
         SELECT cause_id,
                rei_id
         FROM shared.cause_risk_hierarchy_history
         WHERE cause_risk_set_version_id = {cause_risk_set_version_id}
         """
    return ezfuncs.query(q, conn_def='epi')

@memory.cache
def get_age_group_ids(gbd_round_id: int, mortality: bool = False) -> pd.DataFrame:
    """Get the age group ids associated with a particular gbd round and team."""
    from db_queries.get_demographics import get_demographics
    if mortality:
        team = 'mort'
    else:
        team = 'epi'

    return get_demographics(team, gbd_round_id)['age_group_id']


@memory.cache
def get_age_bins(gbd_round_id: int) -> pd.DataFrame:
    """Get the age group bin edges, ids, and names associated with a particular gbd round."""
    from db_tools import ezfuncs
    return ezfuncs.query(f"""
            SELECT age_group_id,
                   age_group_years_start,
                   age_group_years_end,
                   age_group_name
            FROM age_group
            WHERE age_group_id IN ({','.join([str(a) for a in get_age_group_ids(gbd_round_id)])})
            """, conn_def='shared')


@memory.cache
def get_healthstate_mapping() -> pd.DataFrame:
    """Get a mapping between healthstate ids and the healthstate names.

    Notes
    -----
    This mapping is stable between gbd rounds.
    """
    from db_tools import ezfuncs

    q = """
        SELECT healthstate_id,
               healthstate_name
        FROM epi.healthstate
        """
    return ezfuncs.query(q, conn_def='epi')


# FIXME: This function is probably unnecessary with the update to 2016 data.
@memory.cache
def get_healthstate_id(me_id: int) -> int:
    """Get's the healthstate id associate with a particular modelable entity id."""
    from db_tools import ezfuncs

    healthstate_id_df = ezfuncs.query(f"""    
        SELECT modelable_entity_id, 
               healthstate_id
        FROM epi.sequela_hierarchy_history
        WHERE modelable_entity_id = {me_id}
        """, conn_def='epi')

    if healthstate_id_df.empty:
        raise ValueError(f"Modelable entity id {me_id} does not have a healthstate id. "
                         "There is not a disability weight associated with this sequela, "
                         "so you should not try to pull draws for it")

    healthstate_id = healthstate_id_df.at[0, 'healthstate_id']

    return healthstate_id


@memory.cache
def get_subregions(location_id: int) -> List[int]:
    """Get the subregion location ids associated with a particular location id."""
    from hierarchies import dbtrees
    return [c.id for c in dbtrees.loctree(None, location_set_id=2).get_node_by_id(location_id).children]


#####################################
# Tools for pulling draw level data #
#####################################


def _get_draws_safely(draw_function: Callable, draw_options: Iterable[Iterable[int]],
                      *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Allows for pulling draws with multiple draw options to overcome some common errors in central comp tools."""
    measure_draws = None
    for location_id, round_id in draw_options:
        try:
            measure_draws = draw_function(*args, location_ids=location_id, gbd_round_id=round_id, **kwargs)
            break
        except:  # FIXME: Figure out the pattern of errors we get back here and replace the bare except clause.
            pass
    if measure_draws is None:
        raise DataNotFoundError("Couldn't find draws for your requirements\n"
                                f"function : {draw_function.__name__}\ndraw_options :  {draw_options}\n"
                                f"args : {args}\nkwargs : {kwargs}.")
    return measure_draws


@memory.cache
def get_modelable_entity_draws(location_id: int, me_id: int, gbd_round_id: int,
                               publication_ids: Union[Iterable[int], None] = None) -> pd.DataFrame:
    """Gets draw level epi parameters for a particular dismod model, location, and gbd round."""
    from transmogrifier.draw_ops import get_draws

    model_version = get_dismod_model_version(me_id, publication_ids)
    return get_draws(gbd_id_field='modelable_entity_id',
                     gbd_id=me_id,
                     source="dismod",
                     location_ids=location_id,
                     sex_ids=MALE + FEMALE,
                     age_group_ids=get_age_group_ids(gbd_round_id),
                     version_id=model_version,
                     gbd_round_id=gbd_round_id)


@memory.cache
def get_codcorrect_draws(location_id: int, cause_id: int, gbd_round_id: int,
                         publication_ids: Union[Iterable[int], None] = None) -> pd.DataFrame:
    """Gets draw level deaths for a particular cause, location, and gbd round."""
    from transmogrifier.draw_ops import get_draws
    # FIXME: Should submit a ticket to IT to determine if we need to specify an
    # output_version_id or a model_version_id to ensure we're getting the correct results
    return get_draws(gbd_id_field='cause_id',
                     gbd_id=cause_id,
                     source="codcorrect",
                     location_ids=location_id,
                     sex_ids=MALE + FEMALE,
                     age_group_ids=get_age_group_ids(gbd_round_id),
                     gbd_round_id=gbd_round_id)


@memory.cache
def get_como_draws(location_id: int, cause_id: int, gbd_round_id: int,
                   publication_ids: Union[Iterable[int], None] = None) -> pd.DataFrame:
    """Gets draw level epi parameters for a particular cause, location, and gbd round."""
    from transmogrifier.draw_ops import get_draws
    # FIXME: Should submit a ticket to IT to determine if we need to specify an
    # output_version_id or a model_version_id to ensure we're getting the correct results
    return get_draws(gbd_id_field='cause_id',
                     gbd_id=cause_id,
                     source="como",
                     location_ids=location_id,
                     sex_ids=MALE + FEMALE,
                     age_group_ids=get_age_group_ids(gbd_round_id),
                     gbd_round_id=gbd_round_id)


@memory.cache
def get_relative_risks(location_id: int, risk_id: int, gbd_round_id: int) -> pd.DataFrame:
    """Gets draw level relative risks for a particular risk, location, and gbd round."""
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
                             age_group_ids=get_age_group_ids(gbd_round_id),
                             draw_type='rr')


@memory.cache
def get_exposures(location_id: int, risk_id: int, gbd_round_id: int) -> pd.DataFrame:
    """Gets draw level exposure means for a particular risk, location, and gbd round."""
    from transmogrifier.draw_ops import get_draws

    # Sometimes draws from a previous gbd round are reused.
    draw_options = [[location_id, gbd_round_id], [location_id, gbd_round_id - 1]]
    return _get_draws_safely(get_draws, draw_options,
                             gbd_id_field='rei_id',
                             gbd_id=risk_id,
                             source='risk',
                             sex_ids=MALE + FEMALE,
                             age_group_ids=get_age_group_ids(gbd_round_id),
                             draw_type='exposure')


@memory.cache
def get_pafs(location_id: int, cause_id: int, gbd_round_id: int) -> pd.DataFrame:
    """Gets draw level pafs for all risks associated with a particular cause, location, and gbd round."""
    from transmogrifier.draw_ops import get_draws

    # I'm cargo culting here. When the simulation is hosted by a dask worker,
    # we can't spawn sub-processes in the way that get_draws wants to
    # There are better ways of solving this but they involve understanding dask
    # better or working on shared function code, neither of
    # which I'm going to do right now. -Alec
    # TODO: Find out if the dalynator files are still structured the same for the 2016 round.
    worker_count = 0 if current_process().daemon else 6  # One worker per 5-year dalynator file (1990 - 2015)

    if gbd_round_id >= 4:
        # The risk-cause data moved from dalynator to burdenator as of round 4 so must be retrieved from there.
        draw_options = [[location_id, gbd_round_id]]
        return _get_draws_safely(get_draws, draw_options,
                                 gbd_id_field='cause_id',
                                 gbd_id=cause_id,
                                 source='burdenator',
                                 sex_ids=MALE + FEMALE,
                                 age_group_ids=get_age_group_ids(gbd_round_id),
                                 num_workers=worker_count)

    # FIXME: Probably don't need this branch anymore post 2016 update.
    # Sometimes draws from a previous gbd round are reused.
    draw_options = [[location_id, gbd_round_id], [location_id, gbd_round_id - 1]]
    return _get_draws_safely(get_draws, draw_options,
                             gbd_id_field='cause_id',
                             gbd_id=cause_id,
                             source='dalynator',
                             sex_ids=MALE + FEMALE,
                             age_group_ids=get_age_group_ids(gbd_round_id),
                             include_risks=True,
                             num_workers=worker_count)


####################################
# Miscellaneous data pulling tools #
####################################

@memory.cache
def get_covariate_estimates(covariate_id: int, location_id: int) -> pd.DataFrame:
    """Pulls covariate data for a particular covariate and location."""
    from db_queries import get_covariate_estimates
    covariate_estimates = get_covariate_estimates(covariate_id=covariate_id,
                                                  location_id=location_id,
                                                  sex_id=MALE + FEMALE + COMBINED,
                                                  age_group_id=-1)
    return covariate_estimates


@memory.cache
def get_populations(location_id: int, gbd_round_id: int) -> pd.DataFrame:
    """Gets all population levels for a particular location and gbd round."""
    from db_queries import get_population

    return get_population(age_group_id=get_age_group_ids(gbd_round_id),
                          location_id=location_id,
                          year_id=[-1],
                          sex_id=MALE + FEMALE + COMBINED,
                          gbd_round_id=gbd_round_id)


@memory.cache
def get_data_from_auxiliary_file(file_name: str, **kwargs: Any) -> pd.DataFrame:
    """Gets data from our auxiliary files, i.e. data not accessible from the gbd databases."""
    path, encoding = auxiliary_file_path(file_name, **kwargs)
    file_type = path.split('.')[-1]
    if file_type == 'csv':
        data = pd.read_csv(path, encoding=encoding)
    elif file_type == 'h5':
        data = pd.read_hdf(path, encoding=encoding)
    elif file_type == 'dta':
        data = pd.read_stata(path, encoding=encoding)
    elif file_type == 'xlsx':
        data = pd.read_excel(path, encoding=encoding)
    else:
        raise NotImplementedError("File type {} is not supported".format(file_type))
    return data


@memory.cache
def get_rei_metadata(rei_set_id: int, gbd_round_id: int) -> pd.DataFrame:
    """Gets a whole host of metadata associated with a particular rei set and gbd round"""
    import db_queries
    return db_queries.get_rei_metadata(rei_set_id=rei_set_id, gbd_round_id=gbd_round_id)


@memory.cache
def get_cause_metadata(cause_set_id: int, gbd_round_id: int) -> pd.DataFrame:
    """Gets a whole host of metadata associated with a particular cause set and gbd round"""
    import db_queries
    return db_queries.get_cause_metadata(cause_set_id=cause_set_id, gbd_round_id=gbd_round_id)


@memory.cache
def get_risk(risk_id: int, gbd_round_id: int):  # FIXME: I don't know how to properly annotate the return type
    """Gets a risk object containing info about the exposure distribution type and names of exposure categories."""
    from risk_utils.classes import risk
    return risk(risk_id=risk_id, gbd_round_id=gbd_round_id)
