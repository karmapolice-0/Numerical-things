from errors import *
from custom.funcs import overwrite_attributes

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
            from setup.listify import _listify
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
                 dtype: Union[int, float] = float,
                 **kwargs):
        # Constants
        if data is None:
            data = []
        self.ROW_LIMIT = 30  # Upper limit for amount of rows to be printed
        self.DEFAULT_NULL = 0  # Num to use as invalid value indicator
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
        self._fMat = False  # If it's float matrix

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
        self.setInstance(self.__dtype)  # Store what type of values matrix can hold
        # Setup the matrix, types e t.c.
        self.setup(True, self.__implicit)

    # ...
    # ==========================================================================
    """Attribute formatting and settings methods"""

    # ==========================================================================
    def _setDim(self, d: Union[int, list, tuple]):
        """
        Set the dimension to be a list if it's an integer
        """
        valid = False
        if isinstance(d, int):
            if d >= 0:
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
        """
        from setup.matfill import _setMatrix
        _setMatrix(self, dim, ranged, lis, fill, fmat, uniform=uniform, seed=seed, null=self.DEFAULT_NULL)

    def setup(self, first: bool, implicit: bool = False):
        """
        Process and validate
        :param first: bool; whether or not it's the first time running the setup
        :param implicit:
        """
        from setup.argpro import _setup
        _setup(self, first, implicit)

    def setInstance(self, dt: Union[int, float]):
        """
        Set the type
        """
        self._fMat = True if dt is float else False

    # ==========================================================================
    """Attribute recalculation methods"""

    # ==========================================================================
    def _declareDim(self):
        """
        Set new dimension
        """
        from setup.declare import declareDim
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
    def _listify(self, stringold: str):
        from setup.listify import _listify
        return _listify(self, stringold, False)

    # def _stringfy(self, ):

# ==========================================================================
    """Row/Column methods"""
# ==========================================================================
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

    def row(self, row: Union[List[int], int] = None, as_matrix: bool = True):
        """
        Get a specific row of the matrix
        :param row: 0 < int <= row_amount; list of ints; row number(s)
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

    def col(self, column: Union[List[Union[int, str]], int, str],
            as_matrix: bool = True):
        """
        Get a specific column of the matrix
        :param column: 0 < int <= column_amount/ indices(starting from 0) in tuples;
        :param as_matrix: False to get the column as a list, True to get a column matrix (default)
        :return: matrix or list
        """
        column = column - 1 if isinstance(column, int) else column
        return self[:, column] if as_matrix else [row[0] for row in self[:column].matrix]

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
        from matrixops.add import add
        add(self, lis, row, col, dtype, fillnull)
        if returnmat:
            return self

    def remove(self, row: int = None, col: int = None, returnmat: bool = False):
        """
        Deletes the given row and/or column
        :param row: int > 0
        :param col: int > 0
        :param returnmat: bool; whether or not to return self
        """
        from matrixops.remove import remove
        remove(self, self.d0, self.d1, row, col)
        if returnmat:
            return self

    def concat(self, matrix: object, axis: [0, 1] = 1, returnmat: bool = False, fillnull=True):
        """
        Concatenate matrices row or column vice
        :param matrix: Matrix; matrix to concatenate to self
        :param axis: 0/1; 0 to add 'matrix' as rows, 1 to add 'matrix' as columns
        :param returnmat: bool; whether or not to return self
        :param fillnull: bool; whether or not to use null object to fill values in missing indices
        """
        from matrixops.concat import concat
        concat(self, matrix, axis, fillnull, Matrix)
        if returnmat:
            return self

    def delDim(self, num: int):
        """
        Removes desired number of rows and columns from bottom right corner
        """
        from matrixops.matdelDim import delDim
        delDim(self, num)

    def swap(self, swap: int, to: int, axis: [0, 1] = 1, returnmat: bool = False):
        """
        Swap two rows or columns
        :param swap: int > 0; row/column number
        :param to: int > 0; row/column number
        :param axis: 0/1; 0 for row swap, 1 for column swap
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

    def setdiag(self, val: Union[Tuple[Any], List[Any], object], returnmat: bool = False):
        """
        Set new diagonal elements
        :param val: Any; object to set as new diagonals
            ->If a Matrix is given, new diagonals are picked as given matrix's diagonals
            ->If a list/tuple is given, it should have the  of the smaller dimension of the matrix
            ->Any different value types are treated as single values, all diag values get replaced with given object
        :param returnmat: bool; whether or not to return self
        """
        expected_length = min(self.dim)
        if isinstance(val, Matrix):
            if min(val.dim) != expected_length:
                raise DimensionError(f"Expected {expected_length} diagonal elems, got {min(val.dim)} instead")
            for i in range(expected_length):
                self._matrix[i][i] = val._matrix[i][i]  # ???
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



"""
class Matrix:
    EPS = 10e-6
    ROUND_DIG = 13

    def __init__(self,
                 data: Union[List[List[Any]], List[Any], Any] = []):
        self._matrix = data
        self.rows = len(self.data)
        self.columns = len(self.data[0])
        self.p = []
        self.init_p()

    def __deepcopy__(self, memodict={}):
        if memodict is None:
            memodict = {}
        new = Matrix(copy.deepcopy(self.data))
        return new

    def __len__(self):
        return self.rows

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __str__(self):
        tmp = copy.deepcopy(self)
        for i in range(len(tmp)):
            for j in range(len(tmp[i])):
                if abs(tmp[i][j]) <= self.EPS:
                    tmp[i][j] = 0
        s = [[str(e) for e in row] for row in tmp]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}.{}}}'.format(x + 2, x if x > self.ROUND_DIG else self.ROUND_DIG) for x in lens)
        table = [fmt.format(*row) for row in s]
        return '\n'.join(table)

    def __round__(self, n=14):
        for i in range(self.rows):
            for j in range(self.columns):
                if abs(self.data[i][j]) < self.EPS:
                    self.data[i][j] = 0
                else:
                    self.data[i][j] = round(self.data[i][j], n)

    # ==
    def __eq__(self, o: object) -> bool:
        if isinstance(o, Matrix):
            if self.rows != o.rows or self.columns != o.columns:
                return False
            else:
                return self.data == o.data
        return False

    # Addition
    def __add__(self, o: object):
        if not isinstance(o, (Matrix, list)):
            raise AttributeError("Unsupported operand type for +: '{}' and '{}'".format(self.__class__, type(o)))
        else:
            if isinstance(o, Matrix):
                if self.rows != o.rows or self.columns != o.columns:
                    raise AttributeError("Unsupported dimensions for +: '{}'x'{}' and '{}'x'{}'".format(self.rows,
                                                                                                        self.columns,
                                                                                                        o.rows,
                                                                                                        o.columns))
            else:
                if self.rows != len(o) or self.columns != len(o[0]):
                    raise AttributeError("Unsupported dimensions for +: '{}'x'{}' and '{}'x'{}'".format(self.rows,
                                                                                                        self.columns,
                                                                                                        len(o),
                                                                                                        len(o[0])))
            new_data = []
            for i in range(self.rows):
                tmp = []
                for j in range(self.columns):
                    tmp.append(self.data[i][j] + o.data[i][j])
                new_data.append(tmp)
            new_matrix = Matrix(data=new_data)
            return new_matrix

    # Subtraction
    def __sub__(self, o: object):
        if not isinstance(o, (Matrix, list)):
            raise AttributeError("Unsupported operand type for +: '{}' and '{}'".format(self.__class__, type(o)))
        else:
            if isinstance(o, Matrix):
                if self.rows != o.rows or self.columns != o.columns:
                    raise AttributeError("Unsupported dimensions for +: '{}'x'{}' and '{}'x'{}'".format(self.rows,
                                                                                                        self.columns,
                                                                                                        o.rows,
                                                                                                        o.columns))
            else:
                if self.rows != len(o) or self.columns != len(o[0]):
                    raise AttributeError("Unsupported dimensions for +: '{}'x'{}' and '{}'x'{}'".format(self.rows,
                                                                                                        self.columns,
                                                                                                        len(o),
                                                                                                        len(o[0])))
            new_data = []
            for i in range(self.rows):
                tmp = []
                for j in range(self.columns):
                    tmp.append(self.data[i][j] - o.data[i][j])
                new_data.append(tmp)
            new_matrix = Matrix(data=new_data)
            return new_matrix

    # 1. Matrix multiplication block

    # Return the top_left, top_right, bottom_left and bottom_right quadrants
    def split_matrix(self):
        if len(self) % 2 != 0 or len(self[0]) % 2 != 0:
            raise Exception('Odd matrices are not supported!')
        matrix_length = len(self)
        mid = matrix_length // 2
        top_left = [[self[i][j] for j in range(mid)] for i in range(mid)]
        bot_left = [[self[i][j] for j in range(mid)] for i in range(mid, matrix_length)]
        top_right = [[self[i][j] for j in range(mid, matrix_length)] for i in range(mid)]
        bot_right = [[self[i][j] for j in range(mid, matrix_length)] for i in range(mid, matrix_length)]
        return top_left, top_right, bot_left, bot_right

    def naive(self, o: object):
        if not isinstance(o, (Matrix, list)):
            raise AttributeError("Unsupported operand type for *: '{}' and '{}'".format(self.__class__, type(o)))
        elif isinstance(o, Matrix):
            A = self.__deepcopy__()
            B = o.__deepcopy__()
            n = self.rows
            m = self.columns
            p = o.columns
            C = Matrix([[0 for _ in range(p)] for _ in range(n)])
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        C[i][j] += A[i][k] * B[k][j]
            return C
        else:
            A = self.__deepcopy__()
            B = o
            n = self.rows
            m = self.columns
            p = len(o[0])
            C = Matrix([[0 for _ in range(p)] for _ in range(n)])
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        C[i][j] += A[i][k] * B[k][j]
            return C

    def strassenR(self, o: object):
        if not isinstance(o, Matrix):
            raise AttributeError("Unsupported operand type for *: '{}' and '{}'".format(self.__class__, type(o)))
        elif isinstance(o, list):
            return self.naive(o)
        A = self.__deepcopy__()
        B = o.__deepcopy__()
        n = len(A)
        mid = n // 2
        # Allocation
        a11 = Matrix([[0 for _ in range(mid)] for _ in range(mid)])
        a12 = a11.__deepcopy__()
        a21 = a11.__deepcopy__()
        a22 = a11.__deepcopy__()

        b11 = a11.__deepcopy__()
        b12 = a11.__deepcopy__()
        b21 = a11.__deepcopy__()
        b22 = a11.__deepcopy__()

        aResult = a11.__deepcopy__()
        bResult = a11.__deepcopy__()

        # Initializing
        for i in range(mid):
            for j in range(mid):
                a11[i][j] = A[i][j]  # top left
                a12[i][j] = A[i][j + mid]  # top right
                a21[i][j] = A[i + mid][j]  # bottom left
                a22[i][j] = A[i + mid][j + mid]  # bottom right

                b11[i][j] = B[i][j]  # top left
                b12[i][j] = B[i][j + mid]  # top right
                b21[i][j] = B[i + mid][j]  # bottom left
                b22[i][j] = B[i + mid][j + mid]  # bottom right
        # Calculating p1 to p7
        aResult = a11 + a22
        bResult = b11 + b22
        p1 = aResult.strassenR(bResult)

        aResult = a21 + a22
        p2 = aResult.strassenR(b11)

        bResult = b12 - b22
        p3 = a11.strassenR(bResult)

        bResult = b21 - b11
        p4 = a22.strassenR(bResult)

        aResult = a11 + a12
        p5 = aResult.strassenR(b22)

        aResult = a21 - a11
        bResult = b11 + b12
        p6 = aResult.strassenR(bResult)

        aResult = a12 - a22
        bResult = b21 + b22
        p7 = aResult.strassenR(bResult)

        c12 = p3 + p5
        c21 = p2 + p4

        aResult = p1 + p4
        bResult = aResult + p7
        c11 = bResult - p5

        aResult = p1 + p3
        bResult = aResult + p6
        c22 = bResult - p2

        C = Matrix([[0 for _ in range(n)] for _ in range(n)])
        for i in range(mid):
            for j in range(mid):
                C[i][j] = c11[i][j]
                C[i][j + mid] = c12[i][j]
                C[i + mid][j] = c21[i][j]
                C[i + mid][j + mid] = c22[i][j]
        return C

    def strassen(self, o: object):
        if not isinstance(o, Matrix):
            raise AttributeError("Unsupported operand type for *: '{}' and '{}'".format(self.__class__, type(o)))
        assert len(self.data) == len(self.data[0]) == len(o.data) == len(o.data[0])

        # Make matrices bigger so you can apply the strassen without
        # having to deal with odd matrix sizes
        nextPowerOfTwo = lambda k: 2 ** int(ceil(log(k, 2)))
        A = self.__deepcopy__()
        B = o.__deepcopy__()
        n = len(A)
        m = nextPowerOfTwo(n)
        APrep = Matrix([[0 for _ in range(m)] for _ in range(m)])
        BPrep = APrep.__deepcopy__()
        for i in range(n):
            for j in range(n):
                APrep[i][j] = A[i][j]
                BPrep[i][j] = B[i][j]
        CPrep = APrep.strassenR(BPrep)
        C = Matrix([[0 for _ in range(n)] for _ in range(n)])
        for i in range(n):
            for j in range(n):
                C[i][j] = CPrep[i][j]
        return C

    # Add Strassen algo for multiplication later
    def __mul__(self, o: object):
        if not isinstance(o, Matrix) and not isinstance(o, Number) and not isinstance(o, list):
            raise AttributeError("Unsupported operand type for +: '{}' and'{}'".format(self.__class__, type(o)))
        new_matrix = []

        if isinstance(o, Number):
            for row in self.data:
                tmp = []
                for column in row:
                    tmp.append(column * o)
                new_matrix.append(tmp)
            return Matrix(data=new_matrix)
        elif isinstance(o, Matrix):
            if self.columns != o.rows:
                raise AttributeError("Unsupported matrix dimensions for *: '{}'x'{}' and '{}'x'{}'".format(self.rows,
                                                                                                           self.columns,
                                                                                                           o.rows,
                                                                                                           o.columns))
            else:
                if self.rows == self.columns == o.columns:
                    return self.strassen(o)
                else:
                    return self.naive(o)
        else:
            if self.columns != len(o):
                raise AttributeError("Unsupported matrix dimensions for *: '{}'x'{}' and '{}'x'{}'".format(self.rows,
                                                                                                           self.columns,
                                                                                                           len(o),
                                                                                                           len(o[0])))
            else:
                if self.rows == self.columns == len(o[0]) and self.rows > 4:
                    return self.strassen(o)
                else:
                    return self.naive(o)

    def __rmul__(self, o: object):
        if not isinstance(o, Matrix) and not isinstance(o, Number) and not isinstance(o, list):
            raise AttributeError("Unsupported operand type for +: '{}' and'{}'".format(self.__class__, type(o)))
        new_matrix = []

        if isinstance(o, Number):
            for row in self.data:
                tmp = []
                for column in row:
                    tmp.append(column * o)
                new_matrix.append(tmp)
            return Matrix(data=new_matrix)
        elif isinstance(o, Matrix):
            if o.columns != self.rows:
                raise AttributeError("Unsupported matrix dimensions for *: '{}'x'{}' and '{}'x'{}'".format(self.rows,
                                                                                                           self.columns,
                                                                                                           o.rows,
                                                                                                           o.columns))
            else:
                if o.rows == o.columns == self.columns:
                    return o.strassen(self)
                else:
                    return o.naive(self)
        else:
            raise AttributeError("Sorry, __rmul__ is realised only for Matrices :(")
            '''
            if len(o[0]) != self.rows:
                raise AttributeError("Unsupported matrix dimensions for *: '{}'x'{}' and '{}'x'{}'".format(self.rows,
                                                                                                           self.columns,
                                                                                                           len(o),
                                                                                                           len(o[0])))
            else:
                if len(o) == len(o[0]) == self.columns and len(o) > 4:
                    return o.strassen(self)
                else:
                    return o.naive(self)
            '''
    # End of multiplication block

    def __invert__(self):
        # LUP decomposition
        self.lup()
        inversed_data = []
        for i_tmp in range(self.rows):
            inversed_data.append([])

        for i in range(self.rows):
            b = []
            for j in range(self.rows):
                if i == j:
                    b.append([1])
                else:
                    b.append([0])
            b_mat = Matrix(b)
            self.forward_substitution(b_mat)
            self.back_substitution(b_mat)

            for k in range(b_mat.rows):
                for v in range(b_mat.columns):
                    inversed_data[k].append(b_mat[k][v])
        inversed_mat = Matrix(data=inversed_data)
        return inversed_mat

    def init_p(self):
        # Init p
        for i in range(self.rows):
            tmp = []
            for j in range(self.columns):
                if i == j:
                    tmp.append(1)
                else:
                    tmp.append(0)
            if i < len(self.p):
                self.p[i] = tmp
            else:
                self.p.append(tmp)

    def transpose(self):
        if self.rows >= self.columns:

            for i in range(self.rows):
                for j in range(self.columns):
                    if i == j:
                        break
                    tmp = self.data[i][j]
                    if i >= self.columns:
                        self.data[j].append(tmp)
                    else:
                        self.data[i][j] = self.data[j][i]
                        self.data[j][i] = tmp
            for i in range(self.columns, self.rows):
                self.data.remove(self.data[i])
        else:

            for i in range(self.rows, self.columns):
                self.data.append([])
            for i in range(self.rows):
                new_row = []
                for j in range(i + 1, self.columns):
                    if j < self.rows:
                        tmp = self.data[i][j]
                        self.data[i][j] = self.data[j][i]
                        self.data[j][i] = tmp
                    else:
                        self.data[j].append(self.data[i][j])
                        self.data[i].remove(self.data[i][j])
        tmp = self.rows
        self.rows = self.columns
        self.columns = tmp

    '''
    def transposeOpt(self, tileSize=16):
        if self.rows > 2 * self.columns & self.columns <= 64:
            return self.transpose()
        elif self.columns > 2 * self.rows & self.rows <= 64:
            return self.transposeDst()
        elif self.rows % tileSize == 0:
            return self.transposeTilingSO()
        else:
            return self.transposeTiling()
    '''

    def getelem(self, x, y):
        try:
            return self.data[x][y]
        except IndexError:
            return None

    def switch_row(self, first, second):
        if first >= self.rows or second >= self.rows:
            return None
        try:
            tmp = copy.deepcopy(self.data[first])
            self.data[first] = self.data[second]
            self.data[second] = tmp
        except IndexError:
            return None
        try:
            tmp = copy.deepcopy(self.p[first])
            self.p[first] = self.p[second]
            self.p[second] = tmp
        except IndexError:
            pass

    def switch_column(self, first, second):
        if first >= self.columns or second >= self.columns:
            return None
        for i in range(0, self.rows):
            try:
                tmp = self.data[i][first]
                self.data[i][first] = self.data[i][second]
                self.data[i][second] = tmp
            except IndexError:
                return None

    # Find rank of a matrix
    def rank(self):
        res = self.columns
        for row in range(0, res, 1):
            ''' Before visiting curr row 'row', make sure A[row][0],...A[row][row-1] are 0.
            Diag elem != 0, else:
            1) If there's a row below it with non-0 entry, swap and process
            2) If all elems in curr col below A[r][row] are 0, then
            remove this col (swap with last col and self.cols--)
            '''
            if self[row][row] != 0:
                for col in range(0, self.rows, 1):
                    if col != row:
                        # Makes all entries of curr col as 0 except entry A[row][row]
                        k = (self[col][row] / self[row][row])
                        for i in range(res):
                            self[col][i] -= (k * self[row][i])
            else:
                reduce = True
                # Find non-0 elem in curr col
                for i in range(row + 1, self.rows, 1):
                    # Swap the row w\t non-0 elem and this row
                    if self[i][row] != 0:
                        self.switch_row(row, i)
                        reduce = False
                        break
                # If there aren't any rows w\t non-0 elems in curr col, then entire col is 0
                if reduce:
                    res -= 1
                    for i in range(0, self.rows, 1):
                        self[i][row] = self[i][res]
                # Process this row again
                row -= 1
        return res

    def lu(self):
        res = self.__deepcopy__()
        if res.rows != res.columns:
            raise AttributeError("Only square matrices are supported for LU decomposition")
        for i in range(res.rows - 1):
            if abs(res.data[i][i]) <= res.EPS:
                raise ArithmeticError("Pivot element equals zero.")
            for j in range(i + 1, res.rows):
                res.data[j][i] /= res.data[i][i]
                for k in range(i + 1, res.rows):
                    res.data[j][k] -= res.data[j][i] * res.data[i][k]
        return res

    def lup(self):
        res = self.__deepcopy__()
        if res.rows != res.columns:
            raise AttributeError("Only square matrices are supported for LUP decomposition.")
        res.init_p()

        for i in range(res.rows - 1):
            max_idx = i
            for k in range(i + 1, res.columns):
                # find max pivot in column
                if abs(res.data[k][i]) > abs(res.data[max_idx][i]):
                    max_idx = k
            if abs(res.data[max_idx][i]) <= res.EPS:
                raise ArithmeticError("Pivot element equals zero.")

            if max_idx != i:
                res.switch_row(i, max_idx)

            for j in range(i + 1, res.rows):
                res.data[j][i] /= res.data[i][i]
                for k in range(i + 1, res.rows):
                    res.data[j][k] -= res.data[j][i] * res.data[i][k]
        return res

    def get_u(self):
        u_mat = copy.deepcopy(self)
        for i in range(self.rows):
            for j in range(self.columns):
                if j < i:
                    u_mat[i][j] = 0
        return u_mat

    def get_l(self):
        l_mat = copy.deepcopy(self)
        for i in range(self.rows):
            for j in range(self.columns):
                if j > i:
                    l_mat[i][j] = 0
                elif j == i:
                    l_mat[i][j] = 1
        return l_mat

    def forward_substitution(self, b):
        if not isinstance(b, Matrix):
            raise AttributeError("Unsupported parameter type '{}' for parameter b, b has to be Matrix".format(type(b)))
        if self.rows != self.columns:
            raise AttributeError("Only square matrices are supported for LU decomposition")
        b.data = (self.p * b).data
        for i in range(self.rows - 1):  # maybe columns???
            for j in range(i + 1, self.rows):
                b.data[j][0] -= self.data[j][i] * b[i][0]

    def back_substitution(self, b):
        if not isinstance(b, Matrix):
            raise AttributeError("Unsupported parameter type '{}' for parameter b, b has to be Matrix".format(type(b)))
        if self.rows != self.columns:
            raise AttributeError("Only square matrices are supported for LU decomosition")
        for i in range(self.rows - 1, -1, -1):
            if abs(self.data[i][i]) < self.EPS:
                raise AttributeError("Matrix is singular.")
            b.data[i][0] /= self.data[i][i]
            for j in range(i):
                b.data[j][0] -= self.data[j][i] * b.data[i][0]

    # det(A) = sgn(P)*det(L)*det(U)
    def det(self):
        lup = self.lup()
        l = self.get_l()
        u = self.get_u()
        d = 1
        for i in range(self.rows):
            d = d * l[i][i] * u[i][i]
        return d

    def outputfile(self, filename):
        with open(filename, "w") as f:
            print(str(self), file=f)

    @classmethod
    def fromtextfile(cls, file):
        data = []
        with open(file) as f:
            i = 0
            for line in f:
                tmp = []
                j = 0
                line_split = line.split()
                for line_char in line_split:
                    try:
                        value_tmp = float(line_char)
                        tmp.append(value_tmp)
                    except ValueError:
                        tmp.append(0)
                    j += 1
                data.append(tmp)
                i += 1
        matrix = Matrix(data=data)
        return matrix

    @classmethod
    def get_identity_matrix(cls, n):
        data = []
        for i in range(n):
            tmp = []
            for j in range(n):
                if j == i:
                    tmp.append(1)
                else:
                    tmp.append(0)
            data.append(tmp)
        matrix = Matrix(data=data)
        return matrix


def add(A, B):
    n = len(A)
    C = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j] + B[i][j]
    return C


def subtract(A, B):
    n = len(A)
    C = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j] - B[i][j]
    return C


def naive(A, B):
    n = len(A)
    C = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C
"""
