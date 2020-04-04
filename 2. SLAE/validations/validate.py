def validlist(lis, throw=False):
    try:
        if lis == [] or lis is None:
            if throw:
                raise ValueError("Matrix is empty")
            else:
                return None
        else:
            return lis
    except Exception as err:
        if throw:
            raise err
        return False


def exactdimension(lis, d0, d1, throw=False):
    if len(lis) != d0:
        if throw:
            from .errors.errors import DimensionError
            raise DimensionError(f"Expected {d0} rows, got {len(lis)}")
        return False
    if not all([len(inner) == d1 for inner in lis]):
        if throw:
            from . import DimensionError
            raise DimensionError(f"Rows in the list should have {d1} columns")
        return False
    return True
