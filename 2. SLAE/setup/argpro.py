# _setup(self, first, implicit, Label, Matrix, validList)
def _setup(mat, first):
    # Rand numbers???
    randomly_filled = True if mat._matrix in [None, [], {}, [[]]] else False
    # Matrix fix
    if first:
        # setMatrix
        mat.setMatrix(mat.dim, mat.initRange, mat._matrix, mat.fill)
