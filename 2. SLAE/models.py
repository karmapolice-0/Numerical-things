from . errors import *

import copy
from numbers import Number
from math import ceil, log
from typing import Union, Tuple, List, Any, Dict, Optional
from random import random, randint, uniform, triangular, \
    gauss, gammavariate, betavariate, \
    expovariate, lognormvariate, seed


class Vector:
    """
    Column vector
    """

    def __init__(self, data):
        self.__data = []

        # Data check
        if isinstance(data, (list, tuple)):
            self.__data = [[d] if not isinstance(d, list) else d for d in data]
        elif isinstance(data, str):
            from .setup.listify import _listify
            self.__data = _listify(self, data, True)
        else:
            raise TypeError(f"Can't use type {type(data).__name__} for 'data' param")

    @property
    def norm(self):
        return sum([i[0] ** 2 for i in self.__data]) ** 0.5 if self.d1 == 1 else None

    @property
    def dim(self):
        return [self.d0, 1]

    @property
    def d0(self):
        return len(self.__data)

    @property
    def d1(self):
        return 1

    def __getitem__(self, idx):
        return self.__data[idx][0]

    def __setitem__(self, idx, val):
        self.__data[idx][0] = val

    def __repr__(self):
        return "\n" + "\n".join([str(val[0]) for val in self.__data]) + f"\n\nsize: [{self.d0}x1]"


class Matrix(Vector):
    """
    dim: int or list/tuple of 2 ints; dimensions of the matrix. Giving an int creates square matrix

    data: str/list[list[values]]/list[values]; Elements of the matrix.

    fill: Any; Fills the matrix with chosen distribution or the value, default is uniform distribution
        Available distributions:
          ->uniform, gauss, lognormvariate, triangular, gammavariate, betavariate, expovariate.

    ranged: list/tuple of numbers or dict{}
        Usage:
          ->To apply all the elements give a list/tuple
          ->To apply every column individually give a dict as {"Column_name":[*args], ...}
          ->Arguments should follow one of the following rules:
                1)If 'fill' is uniform --> [minimum, maximum];
                2)If 'fill' is gauss or lognormvariate --> [mean, standard_deviation];
                3)If 'fill' is triangular --> [minimum, maximum, mode];
                4)If 'fill' is gammavariate or betavariate --> [alpha, beta]
                5)If 'fill' is expovariate --> [lambda]

    seed: int; seed to use while generating rand numbers, not useful when 'fill' isn't a special distribution

    dtype: int/float; type of values the matrix will hold

    implicit: bool; skip matrix setting operations if dimensions and elements are given
    """

    def __init__(self,
                 dim: Union[int, List[int], Tuple[int], None] = None,
                 data: Union[List[List[Any]], List[Any], Any] = None,
                 fill: Any = None,
                 ranged: Union[List[Any], Tuple[Any], Dict[str, Union[List[Any], Tuple[Any]]], None] = [0, 1],
                 seed: int = None,
                 dtype: Union[int, float] = int,
                 **kwargs):
        # Constants
        if data is None:
            data = []
        self.ROW_LIMIT = 30  # Upper limit for amount of rows to be printed
        # self.DEFAULT_NULL = 0  # Num to use as invalid value indicator
        self.DISPLAY_OPTIONS = {  # Options and symbols used while displaying
            "allow_label_dupes": False,
            "dupe_placeholder": "//",
            "left_top_corner": "+",
            "left_separator": "|",
            "top_separator": "-",
            "col_placeholder": "...",
            "row_placeholder": "...",
        }
        # Basic attributes
        self.__dim = dim  # Dimensions
        self._matrix = data  # Values
        self.__fill = fill  # Filling method for the matrix
        self.__initRange = ranged  # Given range for 'fill'
        self.__seed = seed  # Seed to pick values
        self.__dtype = dtype  # Type of the matrix
        self._fMat = False  # If it's int matrix

        # Overwrite attributes
        if kwargs != {}:
            overwrite_attributes(self, kwargs)
        # Save default arguments for __call__
        self.defaults = {
            "dim": dim,
            "data": data,
            "fill": fill,
            "ranged": ranged,
            "seed": seed,
            "dtype": dtype,
            "ROW_LIMITS": self.ROW_LIMIT,
            "DISPLAY_OPTIONS": self.DISPLAY_OPTIONS
        }
        # Set/fix attributes
        self._setDim(self.__dim)  # Fix dimensions
        # Set fMat
        self._fMat = True if self.__dtype is float else False
        # Setup the matrix, types e t.c.
        self.setMatrix(self.__dim, self.__initRange, self._matrix, self.__fill, self._fMat)

    # ...
    # ==========================================================================
    """Attribute formatting and settings methods"""
    # ==========================================================================
    # CORRECT
    def _setDim(self, d: Union[int, list, tuple]):
        """
        Set the dimension to be a list if it's an integer
        """
        valid = False
        if isinstance(d, int):
            if d >= 1:
                self.__dim = [d, d]
                valid = True
        elif isinstance(d, (list, tuple)):
            if len(d) == 2:
                if isinstance(d[0], int) and isinstance(d[1], int):
                    if d[0] > 0 and d[1] > 0:
                        self.__dim = list(d)
                        valid = True
        if not valid:
            self.__dim = [0, 0]

    def setMatrix(self,
                  dim: Union[int, list, tuple, None] = None,
                  ranged: Union[List[Any], Tuple[Any], None] = None,
                  lis: Union[List[List[Any]], List[Any]] = [],
                  fill: Any = uniform,
                  fmat: bool = True):
        """
        Set the matrix based on the arguments given
        :param dim: dimensions
        :param ranged: given range for 'fill'
        :param lis:
        :param fill:
        :param fmat:
        """
        from .setup.matfill import _setMatrix
        _setMatrix(self, dim, ranged, lis, fill, fmat, uniform=uniform, seed=seed)

    # ==========================================================================
    """Attribute recalculation methods"""
    # ==========================================================================
    # CORRECT
    def _declareDim(self):
        """
        Set new dimension
        """
        from .setup.declare import declareDim
        return declareDim(self)

    def _declareRange(self, lis: Union[List[List[Any]], List[Any]]):
        """
        Finds and returns the range of the elements in a given list
        """
        from setup.declare import declareRange
        return declareRange(self, lis)

# ==========================================================================
    """Element setting methods"""
# ==========================================================================
    # CORRECT
    def _listify(self, stringold: str):
        from .setup.listify import _listify
        return _listify(self, stringold, False)

    # def _stringfy(self, ):

# ==========================================================================
    """Row/Column methods"""
# ==========================================================================
    # CORRECT
    def head(self, rows: int = 5):
        """
        First 'rows' amount of rows of the matrix
        :param rows: int > 0; how many rows to return
        :return: matrix
        """
        if not isinstance(rows, int):
            raise InvalidIndex(rows, "Rows should be a positive integer number")
        if rows <= 0:
            raise InvalidIndex(rows, "Rows can't be less than or equal to 0")
        if self.d0 >= rows:
            return self[:rows]
        return self[:, :]

    # CORRECT
    def tail(self, rows: int = 5):
        """
        Last 'rows' amount of rows of the matrix
        :param rows: int>0; how many rows to return
        :return: matrix
        """
        if not isinstance(rows, int):
            raise InvalidIndex(rows, "Rows should be a positive integer number")
        if rows <= 0:
            raise InvalidIndex(rows, "Rows can't be less than or equal to 0")
        if self.d0 >= rows:
            return self[self.d0 - rows:]
        return self[:, :]

    # CORRECT
    def row(self, row: Union[List[int], int] = None, as_matrix: bool = True):
        """
        Get a specific row of the matrix
        :param row: 0 < int <= row_amount or list of ints; row number(s)
        :param as_matrix: False to get the row as a list, True - as a matrix (default)
        :return: matrix or list
        """
        if isinstance(row, int):
            if not (0 < row <= self.d0):
                raise InvalidIndex(row, "Row index out of range")
            row -= 1
        elif isinstance(row, (tuple, list)):
            if not all([1 if isinstance(num, int) else 0 for num in row]):
                raise TypeError("Given list should only contain ints")
        else:
            raise InvalidIndex(row)
        if as_matrix:
            return self[row]
        return self._matrix[row][:]

    # CORRECT
    def col(self, column: Union[List[int], int],
            as_matrix: bool = True):
        """
        Get a specific column of the matrix
        :param column: 0 < int <= column_amount or list of ints; col number(s)
        :param as_matrix: False to get the column as a list, True to get a column matrix (default)
        :return: matrix or list
        """
        column = column - 1 if isinstance(column, int) else column
        return self[:, column] if as_matrix else [row[0] for row in self[:,column].matrix]

    # CORRECT
    def add(self, lis: List[Any],
            row: Union[int, None] = None,
            col: Union[int, None] = None,
            dtype: Any = None,
            returnmat: bool = False,
            fillnull=True):
        """
        Add a row or a column of numbers

        To append a row, use row = self.d0
        To append a column, use col = self.d1
        :param lis: list; list of objects desired to be added to the matrix
        :param row: int >= 1; row index
        :param col: int >= 1; column index
        :param returnmat: bool; whether or not to return self
        :param fillnull: bool; whether or not to use null object to fill values in missing indices
        :return: None or self
        """
        from .matrixops.add import add
        add(self, lis, row, col, fillnull)
        if returnmat:
            return self

    # CORRECT
    def remove(self, row: int = None, col: int = None, returnmat: bool = False):
        """
        Deletes the given row and/or column
        :param row: int > 0
        :param col: int > 0
        :param returnmat: bool; whether or not to return self
        """
        from .matrixops.remove import remove
        remove(self, self.d0, self.d1, row, col)
        if returnmat:
            return self

    # CORRECT
    def concat(self, matrix: object, axis: [0, 1] = 1, returnmat: bool = False, fillnull=True):
        """
        Concatenate matrices row or column vice
        :param matrix: Matrix; matrix to concatenate to self
        :param axis: 0 or 1; 0 to add 'matrix' as rows, 1 to add 'matrix' as columns
        :param returnmat: bool; whether or not to return self
        :param fillnull: bool; whether or not to use null object to fill values in missing indices
        """
        from .matrixops.concat import concat
        concat(self, matrix, axis, fillnull, Matrix)
        if returnmat:
            return self

    # CORRECT
    def delDim(self, num: int):
        """
        Removes desired number of rows and columns from bottom right corner
        """
        from .matrixops.matdelDim import delDim
        delDim(self, num)

    # CORRECT
    def swap(self, swap: int, to: int, axis: [0, 1] = 1, returnmat: bool = False):
        """
        Swap two rows or columns
        :param swap: int > 0; row/column number
        :param to: int > 0; row/column number
        :param axis: 0 or 1; 0 for row swap, 1 for column swap
        :param returnmat: bool; whether or not to return self
        """
        def indexer(value: int, limit):
            assert ((value > 0) and (value <= limit)), f"0<index_value<={limit}"
            return value - 1
        swap = indexer(swap, self.d1)
        to = indexer(to, self.d1)
        # Row swap
        if axis == 0:
            self._matrix[swap], self._matrix[to] = self._matrix[to][:], self._matrix[swap][:]
        else:
            self[:, swap], self[:, to] = self[:, to], self[:, swap]
        if returnmat:
            return self

    # CORRECT
    def setdiag(self, val: Union[Tuple[Any], List[Any], object], returnmat: bool = False):
        """
        Set new diagonal elements
        :param val: Any; object to set as new diagonals
            ->If a Matrix is given, new diagonals are picked as given matrix's diagonals
            ->If a list/tuple is given, it should have the length of the smaller dimension of the matrix
            ->Any different value types are treated as single values, all diag values get replaced with given object
        :param returnmat: bool; whether or not to return self
        """
        expected_length = min(self.dim)
        if isinstance(val, Matrix):
            if min(val.dim) != expected_length:
                raise DimensionError(f"Expected {expected_length} diagonal elems, got {min(val.dim)} instead")
            for i in range(expected_length):
                self._matrix[i][i] = val._matrix[i][i]
        elif isinstance(val, (list, tuple)):
            if len(val) != expected_length:
                raise DimensionError(f"Expected {expected_length} elems, got {len(val)} instead")
            for i in range(expected_length):
                self._matrix[i][i] = val[i]
        else:
            for i in range(expected_length):
                self._matrix[i][i] = val
        if returnmat:
            return self

    # ==========================================================================
    """Basic properties"""
    # ==========================================================================
    @property
    def p(self):
        print(self)

    @property
    def grid(self):
        print(self.string)

    @property
    def copy(self):
        return Matrix(**self.kwargs)

    @property
    def string(self):
        return self._stringify()

    @property
    def dim(self):
        return self.__dim
    @dim.setter
    def dim(self, val: Union[int, List[int], Tuple[int]]):
        amount = self.__dim[0] * self.__dim[1]
        if isinstance(val, int):
            assert val > 0, "Dimensions can't be <= 0"
            val = [val, val]
        elif isinstance(val, list) or isinstance(val, tuple):
            assert len(val) == 2, f"Matrices accept 2 dimensions, {len(val)} length {type(val)} type can't be used"
        else:
            raise TypeError("dim setter only accepts int>0 or list/tuple with length = 2")
        assert val[0]*val[1] == amount, f"{amount} elements can't fill a matrix with {val} dimensions"
        m = self.matrix
        el = [m[i][j] for i in range(self.d0) for j in range(self.d1)]
        tmp = [[el[c+val[1]*r] for c in range(val[1])] for r in range(val[0])]
        self.__init__(dim=list(val), data=tmp, dtype=self.dtype)

    @property
    def d0(self):
        return self.__dim[0]

    @property
    def d1(self):
        return self.__dim[1]

    @property
    def fill(self):
        return self.__fill
    @fill.setter
    def fill(self, value: [object]):
        try:
            spec_names = ["method", "function", "builtin_function_or_method", int]
            assert (type(value).__name__ in spec_names) \
                or (type(value) in [int, float, list, range]) \
                or value is None
        except AssertionError:
            raise FillError(value)
        else:
            self.__fill = value
            self.setMatrix(self.__dim, self.__initRange, [], value, self._fMat)  # ????

    @property
    def initRange(self):
        return self.__initRange
    @initRange.setter
    def initRange(self, value: Union[List[Union[float, int]], Tuple[Union[float, int]]]):
        # Check given lists compatibility with matrix's 'fill' attr
        def list_checker(value, fill):
            if (fill in [uniform, gauss, gammavariate, betavariate, lognormvariate]) \
            or (isinstance(fill, (int, float))):
                if len(value) != 2:
                    raise IndexError("""initRange|ranged should be in the following format:
                                        fill is gauss|lognormvariate -> [mean, standard_deviation]
                                        fill is (gamma|beta)variate -> [alpha, beta]
                                        fill is uniform -> [minimum, maximum]""")
                if not isinstance(value[0], (float, int)) and not isinstance(value[1], (float, int)):
                    raise ValueError("list contains non int and non float numbers")
            elif fill in [triangular]:
                if len(value) != 3:
                    raise IndexError("initRange|ranged should be in the form of [minimum, maximum, mode]")
                if not (isinstance(value[0], (int, float))) and not (isinstance(value[1], (int, float))) \
                and not (isinstance(value[2], (int, float))):
                    raise ValueError("list contains non int and non float numbers")
            elif fill in [expovariate]:
                if len(value) != 1:
                    raise IndexError("initRange|ranged should be in the form of [lambda]")
            else:
                pass

        if isinstance(value, (list, tuple)):
            list_checker(value, self.fill)
            self.__initRange = list(value)
        else:
            raise TypeError(f"Can't use type '{type(value)}' as a list or a tuple")
        self.setMatrix(self.__dim, self.__initRange, [], self.__fill, self._fMat)

    @property
    def rank(self):
        return self._Rank()

    @property
    def trace(self):
        if not self.isSquare:
            return None
        return sum(self.diags)

    @property
    def matrix(self):
        return self._matrix

    @property
    def data(self):
        return self._matrix

    @property
    def det(self):
        if not self.isSquare:
            return None
        return self._determinantByLUForm()

    @property
    def diags(self):
        m = self._matrix
        return [m[i][i] for i in range(min(self.dim))]

    @property
    def eigenvalues(self):
        return self._eigenvals()

    @property
    def eigenvectors(self):
        res = self._eigenvecs(self.EIGENVEC_ITERS, 1+1e-4)
        return [vec[2] for vec in res[0]] if not res in [None, []] else []

    @property
    def eigenvecmat(self):
        res = self._eigenvecs(self.EIGENVEC_ITERS, 1+1e-4)
        return res[1] if not res in [None, []] else []

    @property
    def diagmat(self):
        res = self._eigenvecs(self.EIGENVEC_ITERS, 1+1e-4)
        return res[2] if not res in [None, []] else None

    @property
    def seed(self):
        return self.__seed
    @seed.setter
    def seed(self, value: int):
        if not isinstance(value, int):
            raise TypeError("Seed must be an int")
        self.__seed = value
        self.setMatrix(self.dim, self.initRange, [], self.fill, self._fMat)

    @property
    def dtype(self):
        return self.__dtype
    @dtype.setter
    def dtype(self, val: Union[int, float]):
        if not val in [int, float]:
            return DtypeError(val.__name__)
        else:
            self.__dtype = val
            self.__init__(dim=self.dim,
                          data=self._matrix,
                          ranged=self.initRange,
                          fill=self.fill,
                          seed=self.seed,
                          dtype=self.dtype)

    @property
    def obj(self):
        f = self.fill
        if type(self.fill).__name__ == "method":
            f = self.fill.__name__
        dm, m, r = self.dim, self.matrix, self.initRange
        s, dt = self.seed, self.dtype.__name__

    # ==========================================================================
    """Check special cases"""
    # ==========================================================================
    @property
    def isSquare(self):
        return self.d0 == self.d1

    @property
    def isIdentity(self):
        """

        """
        if not self.isSquare:
            return False
        return round(self, self.PRECISION).matrix == Identity(self.d0).matrix

    @property
    def isSymmetric(self):
        """
        A[i, j] == A[j, i]
        """
        if not self.isSquare:
            return False
        dig = self.PRECISION
        return self.t.roundForm(dig).matrix == self.roundForm(dig).matrix

    @property
    def isAntiSymmetric(self):
        """
        A[i, j] == -A[j, i]
        """
        if not self.isSquare:
            return False
        dig = self.PRECISION
        return (self.t * -1).roundForm(dig).matrix == self.roundForm(dig).matrix

    @property
    def isPerSymmetric(self):
        """
        A[i, j] == A[n+1 - j, n+1 - i]
        """
        if not self.isSquare:
            return False
        m = self.matrix
        dig = self.PRECISION
        d = self.d0
        for i in range(d):
            for j in range(d):
                if roundto(m[i][j], dig) != roundto(m[d-1-j][d-1-i], dig):
                    return False
        return True

    @property
    def isHermitian(self):
        return (self.ht).matrix == self.matrix

    @property
    def isTriangular(self):
        if not(self.isSquare and (self.isUpperTri or self.isLowerTri)):
            return False
        return True

    @property
    def isUpperTri(self):
        """
        A[i, j] == 0 where i > j
        """
        if not self.isSquare:
            return False
        m = self.matrix
        dig = self.PRECISION
        for i in range(1, self.d0):
            for j in range(i):
                if roundto(m[i][j], dig) != 0:
                    return False
        return True

    @property
    def isLowerTri(self):
        """
        A[i, j] == 0 where i < j
        """
        if not self.isSquare:
            return False
        m = self.matrix
        dig = self.PRECISION
        for i in range(1, self.d0):
            for j in range(i):
                if roundto(m[j][i], dig) != 0:
                    return False
        return True

    @property
    def isDiagonal(self):
        """
        A[i, j] == 0 where i != j
        """
        from functools import reduce
        if not self.isSquare:
            return False
        return self.isUpperTri and self.isLowerTri and (roundto(self.det, self.PRECISION) == reduce((lambda a, b: a*b), self.diags))

    @property
    def isBidiagonal(self):
        """
        A[i, j] == 0 where (i != j or i != j+1) XOR (i != j or i != j-1)
        """
        return self.isUpperBidiagonal or self.isLowerBidiagonal

    @property
    def isUpperBidiagonal(self):
        """
        A[i, j] == 0 where i != j or i != j+1
        """
        if not self.isSquare
            return False
        dig = self.PRECISION
        m = self.matrix
        if 0 in [roundto(m[i][i], dig) for i in range(self.d0)] + [roundto(m[i][i+1], dig) for i in range(self.d0-1)]:
            return False
        for i in range(self.d0-2):
            if [0] * (self.d0 - 2 - i) != roundto(m[i][i+2:], dig):
                return False
        return True

    @property
    def isLowerBidiagonal(self):
        """
        A[i, j] == 0 where i != j or i != j-1
        """
        if not self.isSquare:
            return False
        m = self.matrix
        dig = self.PRECISION
        if 0 in [roundto(m[i][i], dig) for i in range(self.d0)] + [roundto(m[i+1][i], dig) for i in range(self.d0-1)]:
            return False
        for i in range(self.d0-2):
            if [0] * (self.d0 - 2 - i) != roundto(m[i][i+2:], dig):
                return False
        return True

    @property
    def isUpperHessenberg(self):
        """
        A[i, j] == 0 where i < j-1
        """
        if not self.isSquare:
            return False
        m = self.matrix
        dig = self.PRECISION
        for i in range(2, self.d0):
            for j in range(i):
                # Check elems below subdiag to be 0
                if roundto(m[i][j], dig) != 0:
                    return False
        return True

    @property
    def isLowerHessenberg(self):
        """
        A[i, j] == 0 where i > j+1
        """
        if not self.isSquare:
            return False
        m = self.matrix
        dig = self.PRECISION
        for i in range(2, self.d0):
            for j in range(i):
                # Check elems above superdiag to be 0
                if roundto(m[j][i], dig) != 0:
                    return False
        return True

    @property
    def isHessenberg(self):
        """
        A[i, j] == where i > j+1 XOR i < j-1
        """
        return self.isUpperHessenberg or self.isLowerHessenberg

    @property
    def isTridiagonal(self):
        """
        A[i, j] == 0 where abs(i-j) > 1 AND A[i, j] != 0 where 0 <= abs(i-j) <= 1
        """
        if not self.isSquare or self.d0 <= 2:
            return False
        m = self.matrix
        dig = self.PRECISION
        # Check diag and first subdiag and first superdiag
        if 0 in [roundto(m[i][i], dig) for i in range(self.d0)] \
            + [roundto(m[i][i+1], dig) for i in range(self.d0-1)] \
            + [roundto(m[i+1][i], dig) for i in range(self.d0-1)]:
            return False
        # Assure rest of elems are 0
        for i in range(self.d0-2):
            # Non-0 check above first superdiag
            if [0] * (self.d0-2-i) != roundto(m[i][i+2:], dig):
                return False
            # Non-0 check below first subdiag
            if [0] * (self.d0-2-i) != roundto(m[self.d0-i-1][:self.d0-i-2], dig):
                return False
        return True

    @property
    def isToeplitz(self):
        """
        A[i, j] == A[i+1, j+1] for every i, j
        """
        m = self.matrix
        dig = self.PRECISION
        for i in range(self.d0-1):
            for j in range(self.d1-1):
                if roundto(m[i][j], dig) != roundto(m[i+1][j+1], dig):
                    return False
        return True

    @property
    def isIdempotent(self):
        if not self.isSquare:
            return False
        dig = self.PRECISION
        return self.roundForm(dig).matrix == (self@self).roundForm(dig).matrix

    @property
    def isOrthogonal(self):
        if not self.isSquare or self.isSingular:
            return False
        dig = self.PRECISION
        return self.inv.roundForm(dig).matrix == self.t.roundForm(dig).matrix

    @property
    def isUnitary(self):
        if not self.isSquare or self.isSingular:
            return False
        dig = self.PRECISION
        return self.ht.roundForm(dig).matrix == self.inv.roundForm(dig).matrix

    @property
    def isNormal(self):
        if not self.isSquare:
            return False
        dig = self.PRECISION
        return (self@self.ht).roundForm(dig).matrix == (self.ht@self).roundForm(dig).matrix

    @property
    def isCircular(self):
        if not self.isSquare or self.isSingular:
            return False
        dig = self.PRECISION
        return self.inv.roundForm(dig).matrix == self.roundForm(dig).matrix

    @property
    def isPositive(self):
        return bool(self>0)

    @property
    def isNonNegative(self):
        return bool(self>=0)

    @property
    def isProjection(self):
        if not self.isSquare:
            return False
        return self.isHermitian and self.isIdempotent

    @property
    def isInvolutory(self):
        if not self.isSquare:
            return False
        return (self@self).roundForm(4).matrix == Identity(self.d0).matrix

    @property
    def isIncidence(self):
        for i in range(self.d0):
            for j in range(self.d1):
                if not self._matrix[i][j] in [0, 1]:
                    return False
        return True

    @property
    def isZero(self):
        m = self.matrix
        for i in range(self.d0):
            for j in range(self.d1):
                if m[i][j] != 0:
                    return False
        return True

    @property
    def isDefective(self):
        eigs = self.eigenvalues
        return len(set(roundto(eigs, 3))) != len(eigs) if not eigs in [None, []] else False

    # ==========================================================================
    """Get special formats"""
    # ==========================================================================
    @property
    def signs(self):
        mm = self._matrix
        sign_l = [[1 if mm[i][j] >= 0 else -1 for j in range(self.d1)] for i in range(self.d0)]
        return Matrix(self.dim, sign_l, dtype=int)

    @property
    def echelon(self):
        return self._rrechelon(rr=False)[0]

    @property
    def rrechelon(self):
        return self._rrechelon(rr=True)[0]

    @property
    def conj(self):
        tmp = self.copy
        mm = tmp.matrix
        tmp._matrix = [[mm[i][j] for j in range(self.d1)] for i in ragne(self.d0)]
        return tmp

    @property
    def t(self):
        return self._transpose()

    @property
    def ht(self):
        return self._transpose(hermitian=True)

    @property
    def adj(self):
        return self._adjoint()

    @property
    def inv(self):
        return self._inverse()

    @property
    def pseudoinv(self):
        if self.isSquare:
            return self.inv
        if self.d0 > self.d1:
            return ((self.t@self).inv)@(self.t)
        return None

    @property
    def EIGENDEC(self):
        return self._EIGENVEC()

    @property
    def SVD(self):
        return self._SDV()

    @property
    def LU(self):
        lu = self._LU()
        return (lu[2], lu[0])

    @property
    def U(self):
        return self._LU()[0]

    @property
    def L(self):
        return self._LU()[2]

    @property
    def symdec(self):
        ant_sym = self._symDec()
        return (ant_sym[0], ant_sym[1])

    @property
    def sym(self):
        if self.isSquare:
            return self._symDecomp()[0]
        return []

    @property
    def anti(self):
        if self.isSquare:
            return self._symDecomp()[1]
        return []

    @property
    def QR(self):
        qr = self._QR()
        return (qr[0], qr[1])

    @property
    def Q(self):
        return self._QR()[0]

    @property
    def R(self):
        return self._QR()[1]

    @property
    def floorForm(self):
        return self.__floor__()

    @property
    def ceilForm(self):
        return self.__ceil__()

    @property
    def intForm(self):
        return self.__floor__()

    @property
    def floatForm(self):
        mm = self.Matrix
        t = [[float(mm[a][b]) for b in range(self.d1)] for a in range(self.d0)]
        return Matrix(self.dim, t, dtype=float, seed=self.seed)

    @property
    def roundForm(self, decimal: int = 3, printing_decimal: [int, None] = None):
        return self.__round__(decimal, printing_decimal)

    # ==========================================================================
    """Methods for special matrices and properties"""
    # ==========================================================================
    def _determinantByLUForm(self):
        return self._LU()[1]

    def _transpose(self, hermitian: bool = False):
        from linalg.transpose import transpose
        return transpose(self, hermitian, obj=Matrix)

    def minor(self, row: int, col: int, returndet: bool = True):
        from linalg.minor import minor
        return minor(self, row, col, returndet)

    def _inverse(self):
        from linalg.inverse import inverse
        return inverse(self, Identity(self.d0))

    def _adjoint(self):
        from linalg.adjoint import adjoint
        return Matrix(self.dim, adjoint(self), dtype=dt)

    def _Rank(self):
        return self.__rrechelon()[1]

    # ==========================================================================
    """Decomposition methods"""
    # ==========================================================================
    def _rrechelon(self, rr: bool = True):
        from linalg.rrechelon import rrechelon
        return rrechelon(self, [a[:] for a in self._matrix], Matrix)

    def _symDecomp(self):
        from linalg.symmetry import symDecomp
        return symDecomp(self, Matrix(self.dim, fill=0))

    def _LU(self):
        from linalg.LU

    # ==========================================================================
    """Logical-bitwise magic methods"""
    # ==========================================================================
    def __bool__(self):
        if not self.BOOL_MAT:
            return False
        m = self.matrix
        true = 1
        for i in range(self.d0):
            row = m[i]
            for j in range(self.d1):
                if row[j] != true:
                    return False
        return True

    def __invert__(self):
        from .matrixops.bitwise import _invert
        return _invert(self, self.intForm)

    def __and__(self, other: Union[object, list, int, float]):
        """
        Can only be used with '&' operator nat with 'and'
        """
        from .matrixops.bitwise import _and
        return _and(self, other, Matrix, self.matrix)

    def __or__(self, other: Union[object, list, int, float]):
        """
        Can only be used with '|' operator not with 'or'
        """
        from .matrixops.bitwise import _or
        return _or(self, other, Matrix, self.matrix)

    def __xor__(self, other: Union[object, list, int, float]):
        """
        Can only be used with '^' operator
        """
        from .matrixops.bitwise import _xor
        return _xor(self, other, Matrix, self.matrix)