from TensorCode import C0, TensorCode
import galois


def count_vars(formula):
    vars = set()
    for f_clause in formula:
        for var in f_clause:
            vars.add(abs(var))
    return len(vars)

field_size = 17

F = galois.GF(2**field_size)
baseSubset = F((0,1))

basecode = C0

tensor_code = TensorCode(C0)

# number of rows to check in phase 3
NUM_ROWS_TO_CHECK = 3

phi = [(1,2,3), (-2, 4,3), (-1, 2, -3)] # w.l.o.g the variables are numberd from 1 to n
z = [1, 0, 1, 1] # the assignment is a list (or a tuple) of length n
# for all i \in [1, n-1] z[i] is the assignment of i. z[0] is the assignment of n

#phi = [(1,2,3),(3,4,5),(5,6,7),(7,8,-1),(-1,-2,3),(-3,-4,5),(-2,-4,-3),(-2,-7,8), (-2, -3, 9),(-10, 11, 12),(-13, -14, 15),(-16, -15, 13)]
#z=[0,1,0,1,1,1,0,1,1, 0, 1, 0, 1, 0, 0, 1]

n = count_vars(phi)