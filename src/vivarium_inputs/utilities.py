
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


class DataNotExistError(VivariumInputsError, FileNotFoundError):
    """Exception raised when the gbd data for the entity-measure pair do not exist"""
    pass


class DataAbnormalError(VivariumInputsError, ValueError):
    """Exception raised when data has extra columns or values that we do not expect to have"""
    pass


class DataFormattingError(VivariumInputsError, ValueError):
    """Exception raised when data has been improperly formatted."""
    pass
