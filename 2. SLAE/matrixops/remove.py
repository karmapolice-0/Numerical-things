# CORRECT
def remove(mat, m, n, r=None, c=None):
    if r is not None:
        del mat._matrix[r - 1]
        m -= 1
    if c is not None:
        for rows in range(m):
            del mat._matrix[rows][c - 1]
        n -= 1
    mat._Matrix__dim = [m, n]
