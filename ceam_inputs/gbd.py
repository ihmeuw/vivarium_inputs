from multiprocessing.process import current_process
import warnings
from typing import Iterable, Union, List, Any, Mapping

from joblib import Memory
import pandas as pd
import numpy as np

from ceam_inputs.util import get_cache_directory, get_input_config
from ceam_inputs.gbd_mapping import cid, sid, rid
from ceam_inputs.auxiliary_files import auxiliary_file_path

memory = Memory(cachedir=get_cache_directory(get_input_config()), verbose=1)

# Define GBD sex ids for usage with central comp tools.
MALE = [1]
FEMALE = [2]
COMBINED = [3]

GBD_ROUND_ID = 4


class CentralCompError(Exception):
    """Error for failures in central-comp tooling."""
    pass


class DataNotFoundError(CentralCompError):
    """Raised when the set of parameters passed to central comp functions fails due to lack of data."""
    pass


#####################################################################################
# Tools for pulling version ids for particular models.  These should not be cached. #
#####################################################################################

def get_publication_ids_for_round(gbd_round_id: int) -> Iterable[int]:
    """Gets the Lancet publication ids associated with a particular gbd round."""
    from db_tools import ezfuncs
    round_year = {3: 2015, 4: 2016}[gbd_round_id]

    return ezfuncs.query(f'select publication_id from shared.publication where gbd_round = {round_year}',
                         conn_def='epi').publication_id.values


def get_gbd_tool_version(publication_ids: Iterable[int], source: str) -> Union[int, None]:
    """Grabs the version id for codcorrect, burdenator, and como draws."""
    from db_tools import ezfuncs
    # NOTE: this mapping comes from the gbd.metadata_type table but in that
    #       database it isn't in a form that's convenient to query and these
    #       ids should be stable so I'm sticking it here -Alec
    metadata_type_id = {
            'codcorrect': 1,
            'burdenator': 11,
            'como': 4
    }[source]

    version_ids = ezfuncs.query(f"""
        SELECT distinct val
        FROM gbd.gbd_process_version_metadata
        JOIN gbd.gbd_process_version_publication USING (gbd_process_version_id)
        WHERE metadata_type_id = {metadata_type_id} 
              and publication_id in ({','.join([str(pid) for pid in publication_ids])})
        """, conn_def='gbd')
    if version_ids.empty:
        warnings.warn(f'No version id found for {source} with publications {publication_ids}. '
                      'This likely indicates missing entries in the GBD database.')
        return None
    return version_ids['val'].astype('int')[0]


def get_dismod_model_versions(me_ids: Iterable[int],
                              publication_ids: Union[Iterable[int], None]) -> Mapping[int, Union[int, str]]:
    """Grabs the model version ids for dismod draws."""
    from db_tools import ezfuncs

    mapping = ezfuncs.query(f"""
        SELECT modelable_entity_id, 
               model_version_id
        FROM epi.publication_model_version
        JOIN epi.model_version USING (model_version_id)
        JOIN shared.publication USING (publication_id)
        WHERE publication_id in ({','.join([str(pid) for pid in publication_ids])})
        """, conn_def='epi')

    version_dict = dict(mapping[['modelable_entity_id', 'model_version_id']].values)
    versions = {}
    for me_id in me_ids:
        versions[me_id] = version_dict[me_id] if me_id in version_dict else 'best'

    return versions


def get_sequela_set_version_id(gbd_round_id: int) -> int:
    """Grabs the sequela set version associated with a particular gbd round."""
    from db_tools import ezfuncs
    q = """
        SELECT gbd_round_id,
               sequela_set_version_id
        FROM epi.sequela_set_version_active
        """
    return ezfuncs.query(q, conn_def='epi').set_index('gbd_round_id').at[gbd_round_id, 'sequela_set_version_id']


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
def get_cause_me_id_mapping() -> pd.DataFrame:
    """Get a mapping between causes and epi/dismod models"""
    from db_tools import ezfuncs

    q = """SELECT modelable_entity_id,
                  cause_id, 
                  modelable_entity_name
           FROM epi.modelable_entity_cause
           JOIN epi.modelable_entity USING (modelable_entity_id)"""
    return ezfuncs.query(q, conn_def='epi')


@memory.cache
def get_age_group_ids(mortality: bool = False) -> pd.DataFrame:
    """Get the age group ids associated with a particular gbd round and team."""
    from db_queries.get_demographics import get_demographics
    if mortality:
        team = 'mort'
    else:
        team = 'epi'

    return get_demographics(team, GBD_ROUND_ID)['age_group_id']


@memory.cache
def get_age_bins() -> pd.DataFrame:
    """Get the age group bin edges, ids, and names associated with a particular gbd round."""
    from db_tools import ezfuncs
    return ezfuncs.query(f"""
            SELECT age_group_id,
                   age_group_years_start,
                   age_group_years_end,
                   age_group_name
            FROM age_group
            WHERE age_group_id IN ({','.join([str(a) for a in get_age_group_ids()])})
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


@memory.cache
def get_subregions(location_ids: Iterable[int]) -> List[int]:
    """Get the subregion location ids associated with a particular location id."""
    from hierarchies import dbtrees
    return {location_id : [c.id for c in dbtrees.loctree(location_set_id=2).get_node_by_id(location_id).children] for location_id in location_ids}


#####################################
# Tools for pulling draw level data #
#####################################


# TODO: Either this function will be necessary with the 2016 update, or it should be removed.  I'm super hoping
# for the latter.
# def _get_draws_safely(draw_function: Callable, draw_options: Iterable[Iterable[int]],
#                       *args: Any, **kwargs: Any) -> pd.DataFrame:
#     """Allows for pulling draws with multiple draw options to overcome some common errors in central comp tools."""
#     measure_draws = None
#     for location_id, round_id in draw_options:
#         try:
#             measure_draws = draw_function(*args, location_ids=location_id, gbd_round_id=round_id, **kwargs)
#             break
#         except:  # FIXME: Figure out the pattern of errors we get back here and replace the bare except clause.
#             pass
#     if measure_draws is None:
#         raise DataNotFoundError("Couldn't find draws for your requirements\n"
#                                 f"function : {draw_function.__name__}\ndraw_options :  {draw_options}\n"
#                                 f"args : {args}\nkwargs : {kwargs}.")
#     return measure_draws


@memory.cache
def get_modelable_entity_draws(me_ids: Iterable[int], location_ids: Iterable[int]) -> pd.DataFrame:
    """Gets draw level epi parameters for a particular dismod model, location, and gbd round."""
    from transmogrifier.draw_ops import get_draws
    publication_ids = get_publication_ids_for_round(GBD_ROUND_ID)
    model_versions = get_dismod_model_versions(me_ids, publication_ids)
    return pd.concat([get_draws(gbd_id_field='modelable_entity_id',
                                gbd_id=me_id,
                                source="dismod",
                                location_ids=location_ids,
                                sex_ids=MALE + FEMALE + COMBINED,
                                age_group_ids=get_age_group_ids(),
                                version_id=version,
                                gbd_round_id=GBD_ROUND_ID) for me_id, version in model_versions.items()])


@memory.cache
def get_codcorrect_draws(cause_ids: List[cid], location_ids: Iterable[int]) -> pd.DataFrame:
    """Gets draw level deaths for a particular cause, location, and gbd round."""
    from transmogrifier.draw_ops import get_draws
    # FIXME: Should submit a ticket to IT to determine if we need to specify an
    # output_version_id or a model_version_id to ensure we're getting the correct results
    # publication_ids = get_publication_ids_for_round(GBD_ROUND_ID)
    # version_id = get_gbd_tool_version(publication_ids, source='codcorrect')
    return get_draws(gbd_id_field=['cause_id']*len(cause_ids),
                     gbd_id=cause_ids,
                     source="codcorrect",
                     location_ids=location_ids,
                     sex_ids=MALE + FEMALE + COMBINED,
                     age_group_ids=get_age_group_ids(),
                     gbd_round_id=GBD_ROUND_ID)


@memory.cache
def get_como_draws(entity_ids: List[Union[cid, sid]], location_ids: Iterable[int]) -> pd.DataFrame:
    """Gets draw level epi parameters for a particular cause, location, and gbd round."""
    from transmogrifier.draw_ops import get_draws
    # FIXME: Should submit a ticket to IT to determine if we need to specify an
    # output_version_id or a model_version_id to ensure we're getting the correct results
    # publication_ids = get_publication_ids_for_round(GBD_ROUND_ID)
    # version_id = get_gbd_tool_version(publication_ids, source='codcorrect')
    entity_types = ['cause_id' if isinstance(entity_id, cid) else 'sequela_id' for entity_id in entity_ids]
    publication_ids = get_publication_ids_for_round(GBD_ROUND_ID)
    # NOTE: Currently this doesn't pull any thing because the tables haven't been built yet,
    # but get_draws doesn't mind and this will automatically update once the DB tables are in place - J.C 11/20
    model_version = get_gbd_tool_version(publication_ids, 'como')

    return get_draws(gbd_id_field=entity_types,
                     gbd_id=entity_ids,
                     source="como",
                     location_ids=location_ids,
                     sex_ids=MALE + FEMALE + COMBINED,
                     age_group_ids=get_age_group_ids(),
                     version_id=model_version,
                     gbd_round_id=GBD_ROUND_ID)


@memory.cache
def get_relative_risks(risk_ids: Iterable[rid], location_ids: Iterable[int]) -> pd.DataFrame:
    """Gets draw level relative risks for a particular risk, location, and gbd round."""
    from transmogrifier.draw_ops import get_draws
    results = []
    for risk_id in risk_ids:
        # FIXME: We should not need this loop but the current version of get_draws does not return
        # a risk_id/gbd_id column so it's impossible to disambiguate the data for the different
        # risks without doing additional complicated lookups to associate the meid (which it does
        # return) with the risk. So instead we loop and wait for central comp to fix the issue.
        # Help desk ticket: HELP-4746
        data = get_draws(gbd_id_field='rei_id',
                     gbd_id=risk_id,
                     source='risk',
                     location_ids=location_ids,
                     sex_ids=MALE + FEMALE + COMBINED,
                     age_group_ids=get_age_group_ids(),
                     draw_type='rr',
                     gbd_round_id=GBD_ROUND_ID)
        data['risk_id'] = risk_id
        results.append(data)
    return pd.concat(results)


@memory.cache
def get_exposures(risk_ids: Iterable[rid], location_ids: Iterable[int]) -> pd.DataFrame:
    """Gets draw level exposure means for a particular risk, location, and gbd round."""
    from transmogrifier.draw_ops import get_draws
    return get_draws(gbd_id_field='rei_id',
                     gbd_id=risk_ids,
                     source='risk',
                     location_ids=location_ids,
                     sex_ids=MALE + FEMALE + COMBINED,
                     age_group_ids=get_age_group_ids(),
                     draw_type='exposure',
                     gbd_round_id=GBD_ROUND_ID)


@memory.cache
def get_pafs(risk_ids: Iterable[cid], location_ids: Iterable[int]) -> pd.DataFrame:
    """Gets draw level pafs for all risks associated with a particular cause, location, and gbd round."""
    from transmogrifier.draw_ops import get_draws

    # I'm cargo culting here. When the simulation is hosted by a dask worker,
    # we can't spawn sub-processes in the way that get_draws wants to
    # There are better ways of solving this but they involve understanding dask
    # better or working on shared function code, neither of
    # which I'm going to do right now. -Alec
    # FIXME: This is not reflective of the actual file structure according to Joe. -James
    worker_count = 0 if current_process().daemon else 6  # One worker per 5-year burdenator file (1990 - 2015)
    results = []
    for risk_id in risk_ids:
        data = get_draws(gbd_id_field='rei_id',
                         gbd_id=risk_id,
                         source='burdenator',
                         location_ids=location_ids,
                         sex_ids=MALE + FEMALE + COMBINED,
                         age_group_ids=get_age_group_ids(),
                         num_workers=worker_count,
                         gbd_round_id=GBD_ROUND_ID)

        data = data.query('measure_id in (1, 3) and metric_id == 2')
        data['mortality'] = np.where(data.measure_id == 1, 1, 0)
        data['morbidity'] = np.where(data.measure_id == 3, 1, 0)
        del data['measure_id']

        data = data.rename(columns={'rei_id': 'risk_id'})

        results.append(data)
    return pd.concat(results)


####################################
# Miscellaneous data pulling tools #
####################################

@memory.cache
def get_covariate_estimates(covariate_id: int, location_id: int) -> pd.DataFrame:
    """Pulls covariate data for a particular covariate and location."""
    from db_queries import get_covariate_estimates
    # FIXME: Make sure we don't need a version id here.
    covariate_estimates = get_covariate_estimates(covariate_id=covariate_id,
                                                  location_id=location_id,
                                                  sex_id=MALE + FEMALE + COMBINED,
                                                  age_group_id=-1,
                                                  gbd_round_id=GBD_ROUND_ID)
    return covariate_estimates


@memory.cache
def get_populations(location_id: int) -> pd.DataFrame:
    """Gets all population levels for a particular location and gbd round."""
    from db_queries import get_population

    return get_population(age_group_id=get_age_group_ids(),
                          location_id=location_id,
                          year_id=[-1],
                          sex_id=MALE + FEMALE + COMBINED,
                          gbd_round_id=GBD_ROUND_ID)


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
def get_rei_metadata(rei_set_id: int) -> pd.DataFrame:
    """Gets a whole host of metadata associated with a particular rei set and gbd round"""
    import db_queries
    return db_queries.get_rei_metadata(rei_set_id=rei_set_id, gbd_round_id=GBD_ROUND_ID)


@memory.cache
def get_cause_metadata(cause_set_id: int) -> pd.DataFrame:
    """Gets a whole host of metadata associated with a particular cause set and gbd round"""
    import db_queries
    return db_queries.get_cause_metadata(cause_set_id=cause_set_id, gbd_round_id=GBD_ROUND_ID)


@memory.cache
def get_risk(risk_id: int):  # FIXME: I don't know how to properly annotate the return type
    """Gets a risk object containing info about the exposure distribution type and names of exposure categories."""
    from risk_utils.classes import risk
    return risk(risk_id=risk_id, gbd_round_id=GBD_ROUND_ID)


@memory.cache
def get_estimation_years(gbd_round_id: int) -> pd.Series:
    """Gets the estimation years for a particular gbd round."""
    from db_queries.get_demographics import get_demographics
    return get_demographics('epi', gbd_round_id=gbd_round_id)['year_id']


@memory.cache
def get_life_table(location_id: int) -> pd.DataFrame:
    """Gets the table of life expectancies for a particular location and gbd round."""
    import db_queries
    # FIXME: Make sure we don't need version info here.
    return db_queries.get_life_table(location_id=location_id, gbd_round_id=GBD_ROUND_ID)
