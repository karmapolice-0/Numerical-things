# CORRECT
def declareDim(mat):
    m = mat._matrix
    if m in [None, [], {}, [[]]]:
        mat._matrix = []
        return [0, 0]
    try:
        rows = len(m)
        col_len = max(len(row) for row in m)
    except:
        raise IndexError("'data' parameter should get list of lists")
    else:
        for i in range(rows):
            row_len = len(m[i])
            if col_len != row_len:
                m[i] += [0 for _ in range(col_len - row_len)]
        return [rows, col_len]
