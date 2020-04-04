# CORRECT
def concat(mat, matrix, axis, fill, obj):
    d0, d1 = mat.dim
    # Type check
    if not isinstance(matrix, obj):
        raise TypeError(f"Can't concatenate type '{type(matrix).__name__}' to self")
    # Empty matrix
    if [d0, d1] == [0, 0]:
        return mat

    md0, md1 = matrix.dim
    newmat = matrix.copy

    # Fill null if needed
    if fill:
        if axis == 0:
            # Given matrix is smaller
            for i in range(0, d1 - md1):
                newmat.add([0 for _ in range(md0)], col=d1+i+1)
                md1 += 1
            # Given matrix is bigger
            for i in range(0, md1 - d1):
                mat.add([0 for _ in range(d0)], col=md1+i+1)
                d1 += 1
        elif axis == 1:
            # Given matrix is smaller
            for i in range(0, d0 - md0):
                newmat.add([0 for _ in range(md1)], row=d0+i+1)
                md0 += 1
            # Given matrix is bigger
            for i in range(0, md0 - d0):
                mat.add([0 for _ in range(d1)], row=md0+i+1)
    if axis == 0:
        assert d1 == md1, "Dimensions don't match for concat"
    elif axis == 1:
        assert d0 == md0, "Dimensions don't match for concat"
    # Concat
    if axis == 0:
        new = newmat.matrix
        for rows in range(md0):
            mat._matrix.append(new[rows])
    elif axis == 1:
        new = newmat.matrix
        for rows in range(md0):
            mat._matrix[rows] += new[rows]
    else:
        return None
    # Update attributes
    mat._Matrix__dim = mat._declareDim()
