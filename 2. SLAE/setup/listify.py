# CORRECT
def _listify(mat, stringold, isvec):
    """
    Find all the numbers in the given string
    """
    import re
    string = stringold[:]
    d1 = mat.d1
    # Get all int and float values
    pattern = r"-?\d+\.?\d*"
    found = re.findall(pattern, string)
    found = [i for i in found if len(i) != 0]
    # String to number
    try:
        if mat._fMat:
            found = [float(a) for a in found if len(a) != 0]
        else:
            found = [int(a) for a in found if len(a) != 0]
    except ValueError as v:
        raise ValueError(v)
    # Check vector
    if isvec:
        return [[val] for val in found]
    # Fix dimensions
    if mat.dim == [0, 0]:
        mat._Matrix__dim = [1, len(found)]
    # Create matrix
    tmp = []
    e = 0
    for rows in range(mat.dim[0]):
        tmp.append([])
        for cols in range(mat.dim[1]):
            tmp[rows].append(found[cols + e])
        e += d1
    return tmp
