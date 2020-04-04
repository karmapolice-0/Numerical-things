class MatrixError(Exception):
    def __init__(self, msg=""):
        self.message = msg

    def __str__(self):
        return self.message


class DimensionError(MatrixError):
    def __init__(self, err, *args):
        self.message = err


class NotListOrTuple(MatrixError):
    def __init__(self, err, *args):
        self.message = f"Given value should be a list or a tuple, not '{type(err).__name__}'"+". ".join(args)


class EmptyMatrix(MatrixError):
    def __init__(self, err, *args):
        self.message = str(err).join(args)


class InvalidIndex(MatrixError):
    def __init__(self, err, *args):
        self.message = f"'{type(err).__name__}' type index '{err}' can't be used as a row index. "+". ".join(args)


class InvalidColumn(MatrixError):
    def __init__(self, err, *args):
        self.message = f"'{type(err).__name__}' type index '{err}' can't be used as a column index. "+". ".join(args)


class FillError(MatrixError):
    def __init__(self, err, *args):
        self.message = f"'{type(err).__name__}' type '{err}' can't be used to fill matrices. "+". ".join(args)


class OutOfRangeList(MatrixError):
    def __init__(self, lis, r, *args):
        self.message = f"Given {lis} should have values in range {r} \n"+". ".join(args)


class ParameterError(MatrixError):
    def __init__(self, err, params, *args):
        self.message = f"'{err}' isn't a valid parameter name. \nAvailable parameter names:\n\t{params}. "+". ".join(args)

