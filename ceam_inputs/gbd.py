"""
Wrappers for the lower level GBD and auxiliary data access tools.

NOTE on warnings. This module explicitly make deprecation warnings from modules
it depends directly on visible but only if they actually used in the current
process in order to reduce noise.
"""
from multiprocessing.process import current_process
import warnings
from typing import Iterable, Union, List, Any, Mapping

from joblib import Memory
import pandas as pd

from ceam_inputs.util import get_cache_directory, get_input_config
from ceam_inputs.gbd_mapping import cid, sid, rid
from ceam_inputs.auxiliary_files import auxiliary_file_path


memory = Memory(cachedir=get_cache_directory(get_input_config()), verbose=1)

# Define GBD sex ids for usage with central comp tools.
MALE = [1]
FEMALE = [2]
COMBINED = [3]

GBD_ROUND_ID = 4


class GbdError(Exception):
    """Error for failures accessing GBD data."""
    pass


class DataNotFoundError(GbdError):
    """Raised when the set of parameters passed to central comp functions fails due to lack of data."""
    pass


#####################################################################################
# Tools for pulling version ids for particular models.  These should not be cached. #
#####################################################################################

def get_publication_ids_for_round(gbd_round_id: int) -> Iterable[int]:
    """Gets the Lancet publication ids associated with a particular gbd round."""
    from db_tools import ezfuncs
    warnings.filterwarnings("default", module="db_tools")

    round_year = {3: 2015, 4: 2016}[gbd_round_id]

    return ezfuncs.query(f"""SELECT publication_id FROM
                             shared.publication
                             WHERE gbd_round = {round_year}
                             AND shared.publication.publication_type_id = 1""",
                         conn_def='epi').publication_id.values


def get_gbd_tool_version(publication_ids: Iterable[int], source: str) -> Union[int, None]:
    """Grabs the version id for codcorrect, burdenator, and como draws."""
    from db_tools import ezfuncs
    warnings.filterwarnings("default", module="db_tools")

    metadata_type_name = {
            'codcorrect': 'CoDCorrect Version',
            'burdenator': 'Burdenator Version',
            'como': 'Como Version',
    }[source]

    version_ids = ezfuncs.query(f"""
    SELECT distinct val
        FROM gbd.gbd_process_version_metadata
        JOIN gbd.gbd_process_version_publication USING (gbd_process_version_id)
        JOIN gbd.gbd_process_version using (gbd_process_version_id)
        JOIN gbd.metadata_type using (metadata_type_id)
        WHERE metadata_type = '{metadata_type_name}'
        AND publication_id in ({','.join([str(pid) for pid in publication_ids])})
        AND gbd.gbd_process_version.gbd_process_id = 1
    """, conn_def='gbd')

    if version_ids.empty:
        warnings.warn(f'No version id found for {source} with publications {publication_ids}. '
                      'This likely indicates missing entries in the GBD database.')
        return None
    else:
        assert len(version_ids) == 1, "Returned multiple tool version ids for round {GBD_ROUND}. This probably means there is a mistake in the publications mapping in the database."
    return version_ids['val'].astype('int')[0]


def get_dismod_model_versions(me_ids: Iterable[int],
                              publication_ids: Union[Iterable[int], None]) -> Mapping[int, Union[int, str]]:
    """Grabs the model version ids for dismod draws."""
    from db_tools import ezfuncs
    warnings.filterwarnings("default", module="db_tools")

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
        versions[me_id] = version_dict[me_id] if me_id in version_dict else None

    return versions


def get_sequela_set_version_id(gbd_round_id: int) -> int:
    """Grabs the sequela set version associated with a particular gbd round."""
    from db_tools import ezfuncs
    warnings.filterwarnings("default", module="db_tools")

    q = """
        SELECT gbd_round_id,
               sequela_set_version_id
        FROM epi.sequela_set_version_active
        """
    return ezfuncs.query(q, conn_def='epi').set_index('gbd_round_id').at[gbd_round_id, 'sequela_set_version_id']


def get_cause_risk_set_version_id(gbd_round_id: int) -> int:
    """Grabs the version id associated with a cause risk mapping for a particular gbd round."""
    from db_tools import ezfuncs
    warnings.filterwarnings("default", module="db_tools")

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
    warnings.filterwarnings("default", module="db_tools")

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
    warnings.filterwarnings("default", module="db_tools")

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
    warnings.filterwarnings("default", module="db_tools")

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
    warnings.filterwarnings("default", module="db_tools")

    q = """SELECT modelable_entity_id,
                  cause_id, 
                  modelable_entity_name
           FROM epi.modelable_entity_cause
           JOIN epi.modelable_entity USING (modelable_entity_id)"""
    return ezfuncs.query(q, conn_def='epi')


@memory.cache
def get_age_group_id(mortality: bool = False) -> pd.DataFrame:
    """Get the age group ids associated with a particular gbd round and team."""
    from db_queries.get_demographics import get_demographics
    warnings.filterwarnings("default", module="db_queries")

    if mortality:
        team = 'mort'
    else:
        team = 'epi'

    return get_demographics(team, GBD_ROUND_ID)['age_group_id']


@memory.cache
def get_age_bins() -> pd.DataFrame:
    """Get the age group bin edges, ids, and names associated with a particular gbd round."""
    from db_tools import ezfuncs
    warnings.filterwarnings("default", module="db_tools")

    return ezfuncs.query(f"""
            SELECT age_group_id,
                   age_group_years_start,
                   age_group_years_end,
                   age_group_name
            FROM age_group
            WHERE age_group_id IN ({','.join([str(a) for a in get_age_group_id()])})
            """, conn_def='shared')


@memory.cache
def get_healthstate_mapping() -> pd.DataFrame:
    """Get a mapping between healthstate ids and the healthstate names.

    Notes
    -----
    This mapping is stable between gbd rounds.
    """
    from db_tools import ezfuncs
    warnings.filterwarnings("default", module="db_tools")

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
    warnings.filterwarnings("default", module="hierarchies")

    return {location_id : [c.id for c in dbtrees.loctree(location_set_id=2).get_node_by_id(location_id).children] for location_id in location_ids}


#####################################
# Tools for pulling draw level data #
#####################################


@memory.cache
def get_modelable_entity_draws(me_ids: Iterable[int], location_ids: Iterable[int]) -> pd.DataFrame:
    """Gets draw level epi parameters for a particular dismod model, location, and gbd round."""
    from get_draws.api import get_draws
    warnings.filterwarnings("default", module="get_draws")

    publication_ids = get_publication_ids_for_round(GBD_ROUND_ID)
    model_versions = get_dismod_model_versions(me_ids, publication_ids)
    return pd.concat([get_draws(gbd_id_type='modelable_entity_id',
                                gbd_id=me_id,
                                source="epi",
                                location_id=location_ids,
                                sex_id=MALE + FEMALE + COMBINED,
                                age_group_id=get_age_group_id(),
                                version_id=version,
                                gbd_round_id=GBD_ROUND_ID) for me_id, version in model_versions.items()])


@memory.cache
def get_codcorrect_draws(cause_ids: List[cid], location_ids: Iterable[int]) -> pd.DataFrame:
    """Gets draw level deaths for a particular cause, location, and gbd round."""
    from get_draws.api import get_draws
    warnings.filterwarnings("default", module="get_draws")

    # FIXME: Should submit a ticket to IT to determine if we need to specify an
    # output_version_id or a model_version_id to ensure we're getting the correct results
    # publication_ids = get_publication_ids_for_round(GBD_ROUND_ID)
    # version_id = get_gbd_tool_version(publication_ids, source='codcorrect')
    return get_draws(gbd_id_type=['cause_id']*len(cause_ids),
                     gbd_id=cause_ids,
                     source="codcorrect",
                     location_id=location_ids,
                     sex_id=MALE + FEMALE + COMBINED,
                     age_group_id=get_age_group_id(),
                     gbd_round_id=GBD_ROUND_ID)


@memory.cache
def get_como_draws(entity_ids: List[Union[cid, sid]], location_ids: Iterable[int]) -> pd.DataFrame:
    """Gets draw level epi parameters for a particular cause, location, and gbd round."""
    from get_draws.api import get_draws
    warnings.filterwarnings("default", module="get_draws")

    # FIXME: Should submit a ticket to IT to determine if we need to specify an
    # output_version_id or a model_version_id to ensure we're getting the correct results
    # publication_ids = get_publication_ids_for_round(GBD_ROUND_ID)
    # version_id = get_gbd_tool_version(publication_ids, source='codcorrect')
    entity_types = ['cause_id' if isinstance(entity_id, cid) else 'sequela_id' for entity_id in entity_ids]
    publication_ids = get_publication_ids_for_round(GBD_ROUND_ID)
    # NOTE: Currently this doesn't pull any thing because the tables haven't been built yet,
    # but get_draws doesn't mind and this will automatically update once the DB tables are in place - J.C 11/20
    model_version = get_gbd_tool_version(publication_ids, 'como')

    return get_draws(gbd_id_type=entity_types,
                     gbd_id=entity_ids,
                     source="como",
                     location_id=location_ids,
                     sex_id=MALE + FEMALE + COMBINED,
                     age_group_id=get_age_group_id(),
                     version_id=model_version,
                     year_id=get_estimation_years(GBD_ROUND_ID),
                     gbd_round_id=GBD_ROUND_ID)



@memory.cache
def get_relative_risks(risk_ids: Iterable[rid], location_ids: Iterable[int]) -> pd.DataFrame:
    """Gets draw level relative risks for a particular risk, location, and gbd round."""
    from get_draws.api import get_draws
    warnings.filterwarnings("default", module="get_draws")

    results = []
    for risk_id in risk_ids:
        # FIXME: We should not need this loop but the current version of get_draws does not return
        # a risk_id/gbd_id column so it's impossible to disambiguate the data for the different
        # risks without doing additional complicated lookups to associate the meid (which it does
        # return) with the risk. So instead we loop and wait for central comp to fix the issue.
        # Help desk ticket: HELP-4746
        data = get_draws(gbd_id_type='rei_id',
                         gbd_id=risk_id,
                         source='rr',
                         location_id=location_ids,
                         sex_id=MALE + FEMALE + COMBINED,
                         age_group_id=get_age_group_id(),
                         gbd_round_id=GBD_ROUND_ID)
        data['risk_id'] = risk_id
        results.append(data)
    return pd.concat(results)


@memory.cache
def get_exposures(risk_ids: Iterable[rid], location_ids: Iterable[int]) -> pd.DataFrame:
    """Gets draw level exposure means for a particular risk, location, and gbd round."""
    from get_draws.api import get_draws
    warnings.filterwarnings("default", module="get_draws")

    results = []
    for risk_id in risk_ids:
        # FIXME: We should not need this loop but the current version of get_draws does not return
        # a risk_id/gbd_id column so it's impossible to disambiguate the data for the different
        # risks without doing additional complicated lookups to associate the meid (which it does
        # return) with the risk. So instead we loop and wait for central comp to fix the issue.
        # Help desk ticket: HELP-4746
        data = get_draws(gbd_id_type='rei_id',
                gbd_id=risk_id,
                source='exposure',
                location_id=location_ids,
                sex_id=MALE + FEMALE + COMBINED,
                age_group_id=get_age_group_id(),
                gbd_round_id=GBD_ROUND_ID)
        data['risk_id'] = risk_id
        results.append(data)
    return pd.concat(results)


@memory.cache
def get_pafs(entity_ids: Iterable[rid], location_ids: Iterable[int]) -> pd.DataFrame:
    """Gets draw level pafs for all risks associated with a particular cause, location, and gbd round."""
    from get_draws.api import get_draws
    warnings.filterwarnings("default", module="get_draws")

    # I'm cargo culting here. When the simulation is hosted by a dask worker,
    # we can't spawn sub-processes in the way that get_draws wants to
    # There are better ways of solving this but they involve understanding dask
    # better or working on shared function code, neither of
    # which I'm going to do right now. -Alec
    # FIXME: This is not reflective of the actual file structure according to Joe. -James
    worker_count = 0 if current_process().daemon else 6  # One worker per 5-year burdenator file (1990 - 2015)
    results = []
    for entity_id in entity_ids:
        if isinstance(entity_id, rid):
            id_type = 'rei_id'
        else:
            id_type = 'cause_id'
        data = get_draws(gbd_id_type=id_type,
                gbd_id=entity_id,
                source='burdenator',
                location_id=location_ids,
                sex_id=MALE + FEMALE,
                age_group_id=get_age_group_id(),
                num_workers=worker_count,
                gbd_round_id=GBD_ROUND_ID)

        data = data.query('metric_id == 2')
        del data['metric_id']

        results.append(data)
    return pd.concat(results)


####################################
# Miscellaneous data pulling tools #
####################################

@memory.cache
def get_covariate_estimates(covariate_ids: Iterable[int], location_ids: Iterable[int]) -> pd.DataFrame:
    """Pulls covariate data for a particular covariate and location."""
    from db_queries import get_covariate_estimates
    warnings.filterwarnings("default", module="db_queries")

    # FIXME: Make sure we don't need a version id here.
    covariate_estimates = pd.concat([get_covariate_estimates(covariate_id=covariate_id,
        location_id=location_ids,
        sex_id=MALE + FEMALE + COMBINED,
        age_group_id=-1,
        gbd_round_id=GBD_ROUND_ID)
        for covariate_id in covariate_ids])
    return covariate_estimates


@memory.cache
def get_populations(location_id: int) -> pd.DataFrame:
    """Gets all population levels for a particular location and gbd round."""
    from db_queries import get_population
    warnings.filterwarnings("default", module="db_queries")


    return get_population(age_group_id=get_age_group_id(),
            location_id=location_id,
            year_id=[-1],
                          sex_id=MALE + FEMALE + COMBINED,
                          gbd_round_id=GBD_ROUND_ID)


@memory.cache
def get_data_from_auxiliary_file(file_name: str, **kwargs: Any) -> pd.DataFrame:
    """Gets data from our auxiliary files, i.e. data not accessible from the gbd databases."""
    kwargs = dict(kwargs)
    kwargs['gbd_round'] = {4: 'GBD_2016'}[GBD_ROUND_ID]
    path, encoding = auxiliary_file_path(file_name, **kwargs)
    file_type = path.split('.')[-1]
    if file_type == 'csv':
        data = pd.read_csv(path, encoding=encoding)
    elif file_type in ('h5', 'hdf'):
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
    warnings.filterwarnings("default", module="db_queries")

    return db_queries.get_rei_metadata(rei_set_id=rei_set_id, gbd_round_id=GBD_ROUND_ID)


@memory.cache
def get_cause_metadata(cause_set_id: int) -> pd.DataFrame:
    """Gets a whole host of metadata associated with a particular cause set and gbd round"""
    import db_queries
    warnings.filterwarnings("default", module="db_queries")

    return db_queries.get_cause_metadata(cause_set_id=cause_set_id, gbd_round_id=GBD_ROUND_ID)


@memory.cache
def get_risk(risk_id: int):  # FIXME: I don't know how to properly annotate the return type
    """Gets a risk object containing info about the exposure distribution type and names of exposure categories."""
    from risk_utils.classes import risk
    warnings.filterwarnings("default", module="risk_utils")

    return risk(risk_id=risk_id, gbd_round_id=GBD_ROUND_ID)


@memory.cache
def get_estimation_years(gbd_round_id: int) -> pd.Series:
    """Gets the estimation years for a particular gbd round."""
    from db_queries.get_demographics import get_demographics
    warnings.filterwarnings("default", module="db_queries")

    return get_demographics('epi', gbd_round_id=gbd_round_id)['year_id']


@memory.cache
def get_theoretical_minimum_risk_life_expectancy() -> pd.DataFrame:
    from db_tools import ezfuncs
    warnings.filterwarnings("default", module="db_tools")

    query = f'''
    select precise_age as age, mean as life_expectancy
    from upload_theoretical_minimum_risk_life_table_estimate
    inner join process_version on process_version.run_id = upload_theoretical_minimum_risk_life_table_estimate.run_id
    where life_table_parameter_id = 5
    and estimate_stage_id = 6
    and gbd_round_id = {GBD_ROUND_ID}
    and status_id = 5
    and process_id = 30
    '''
    return ezfuncs.query(query, conn_def='mortality')
