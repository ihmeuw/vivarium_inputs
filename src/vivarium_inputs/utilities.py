
class DataError(Exception):
    """Base exception for errors in data loading."""
    pass


class DataNotExistError(DataError):
    """Exception raised when the gbd data for the entity-measure pair do not exist"""
    pass


class DataAbnormalError(DataError):
    """Exception raised when data has extra columns or values that we do not expect to have"""
