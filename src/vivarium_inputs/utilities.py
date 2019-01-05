
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








