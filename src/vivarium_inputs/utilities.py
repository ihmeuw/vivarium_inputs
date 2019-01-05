
try:
    from vivarium_gbd_access import gbd
except ModuleNotFoundError:
    class GbdDummy:
        """Mock class to wrap internal dependency."""
        def __getattr__(self, item):
            raise ModuleNotFoundError("Required package vivarium_gbd_access not found.")
    gbd = GbdDummy()


class VivariumInputsError(Exception):
    """Base exception for errors in vivarium_inputs."""
    pass


class GbdDataError(VivariumInputsError):
    """Base exception for errors related to GBD data."""
    pass


class DataNotExistError(VivariumInputsError, FileNotFoundError):
    """Exception raised when the gbd data for the entity-measure pair do not exist"""
    pass


class DataAbnormalError(VivariumInputsError, ValueError):
    """Exception raised when data has extra columns or values that we do not expect to have"""
    pass


class EtlError(VivariumInputsError):
    """Base exception for errors related to vivarium_inputs improperly handling data."""
    pass


class DataFormattingError(EtlError, ValueError):
    """Exception raised when data has been improperly formatted."""
    pass


class UserError(VivariumInputsError):
    """Base exception for user errors."""
    pass


class InvalidQueryError(VivariumInputsError, ValueError):
    """Exception raised when the user makes an invalid query."""
    pass


def get_location_id(location_name):
    return {r.location_name: r.location_id for _, r in gbd.get_location_ids().iterrows()}[location_name]


def scrub_gbd_conventions(data, location):
    scrub_location(data, location)
    scrub_sex(data)
    scrub_age(data)
    scrub_year(data)


def scrub_location(data, location):
    if 'location_id' in data.columns:
        data = data.drop('location_id', 'columns')
    data['location'] = location
    return data


def scrub_sex(data):
    if 'sex_id' in data.columns:
        data['sex'] = data['sex_id'].map({1: 'Male', 2: 'Female'})
        data = data.drop('sex_id', 'columns')
    return data


def scrub_age(data):
    if 'age_group_id' in data.columns:
        age_bins = (
            gbd.get_age_bins()[['age_group_id', 'age_group_years_start', 'age_group_years_end']]
                .rename(columns={'age_group_years_start': 'age_group_start',
                                 'age_group_years_end': 'age_group_end'})
        )
        data = data.merge(age_bins, on='age_group_id').drop('age_group_id', 'columns')
    return data


def scrub_year(data):
    if 'year_id' in data.columns:
        data = data.rename(columns={'year_id': 'year_start'})
        data['year_end'] = data['year_start'] + 1
    return data
