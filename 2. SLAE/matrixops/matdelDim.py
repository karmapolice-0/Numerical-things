# CORRECT
def delDim(mat, num):
    try:
        if not mat.matrix:
            return "Empty matrix"
        assert isinstance(num, int) and num > 0 and mat.dim[0] - num >= 0 and mat.dim[1] - num >= 0
        goal1 = mat.dim[0] - num
        goal2 = mat.dim[1] - num
        if goal1 == 0 and goal2 == 0:
            print("All rows have been deleted")
        mat._Matrix__dim = [goal1, goal2]
        mm = mat._matrix
        mat._matrix = [mm[i][:goal2] for i in range(goal1)]
    except AssertionError:
        print("Enter a valid input")
    except Exception as err:
        print(err)
    else:
        return mat
