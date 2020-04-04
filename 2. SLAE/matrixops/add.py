# CORRECT
def add(mat, lis, row, col, fill):
    r, c = 0, 0
    d0, d1 = mat.dim
    assert isinstance(lis, (list, tuple)), "'lis' parameter only accepts tuples or lists"
    length = len(lis)

    if row is None ^ col is None:
        # Insert a row
        if col is None:
            # Empty matrix
            if [d0, d1] == [0, 0]:
                mat.__init__([1, length], lis, dtype=mat.dtype)
                return mat
            r += 1
            if fill:
                # Given list is shorter
                for rest in range(0, d1 - length):
                    lis.append(0)
                length = len(lis)
                # Given list is longer
                for rest in range(0, length - d1):
                    mat.add([0 for _ in range(d0)], col=d1+rest+1)
                    d1 += 1
            if length != d1:
                raise ValueError(f"Given list's length doesn't match the dimensions; expected {d1} elems, got {length} instead")
            if row > 0 and isinstance(row, int):
                mat._matrix.insert(row - 1, list(lis))
            else:
                raise ValueError(f"'row' should be an int higher than 0")
        # Insert a column
        elif row is None:
            # Empty matrix
            if [d0, d1] == [0, 0]:
                mat.__init__([length, 1], lis, dtype=mat.dtype)
                return mat
            c += 1
            if fill:
                # Given list is shorter
                for rest in range(0, d0 - length):
                    lis.append(0)
                length = len(lis)
                # Given list is longer
                for rest in range(0, length - d0):
                    mat.add([0 for _ in range(d1)], row=d0+rest+1)
                    d0 += 1
            if length != d0:
                raise ValueError(f"Given list's length doesn't match the dimensions; expected {d0} elems, got {length} instead")
            if col > 0 and isinstance(col, int):
                col -= 1
                for i in range(mat.d0):
                    mat._matrix[i].insert(col, lis[i])
            else:
                raise ValueError(f"'col' should be an int higher than 0")

        else:
            raise TypeError("Either one of 'row' and 'col' param should have a value passed")
        mat._Matrix__dim = [d0+r, d1+c]
