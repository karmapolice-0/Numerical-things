"""def setMatrix(self,
                  dim: Union[int, list, tuple, None] = None,
                  ranged: Union[List[Any], Tuple[Any], None] = None,
                  lis: Union[List[List[Any]], List[Any]] = [],
                  fill: Any = uniform,
                  fmat: bool = True):
"""
def _setMatrix(mat, d, r, lis, fill, fmat, uniform, seed):
    """

    :param mat: it's the Matrix object, obviously
    :param d: int or list or tuple or none; dimensions
    :param r: list[any] or tuple[any], none; range
    :param lis: list[list[any]] or list[any]; data
    :param fill: NOT STR; shows with which distribution you should fill
    :param fmat: bool; is it float or not?
    :param uniform:
    :param seed:
    """
    if isinstance(d, int):
        mat._setDim(d)  # or setDim
    if not isinstance(lis, (str, list)):
        lis = []
    # Empty list given
    if len(lis) == 0:
        if fill is None:
            fill = uniform  # null if dfMat
            mat._Matrix__fill = fill

    isMethod = bool(type(fill).__name__ in ["method", "function", "builtin_function_or_method"])
    d0, d1 = d

    if lis in [None, "", {}]:
        lis = []
    if not isinstance(lis, (list, str)):
        raise TypeError("'data' parameter only accepts lists, strings and dictionaries")

    # Set new range
    if r is None:
        r = mat.initRange
    else:
        mat._Matrix_initRange = r

    # Save the seed for reproduction
    if mat.seed == None and len(lis) == 0 and isMethod:
        randseed = int(uniform(-2**24, 2**24))
        mat._Matrix__seed = randseed
    elif isMethod and len(lis) == 0:
        seed(mat.seed)
    else:
        mat._Matrix__seed = None

    # Set the new matrix
    if isinstance(lis, str):
        mat._matrix = mat._listify(lis)
        if mat.dim == [0, 0]:
            mat._Matrix__dim = mat._declareDim()
    else:
        if len(lis) > 0:
