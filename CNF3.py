from parameters import n

def convert(args):
    return int("0b"+''.join(str(i) for i in args), base=0)

def convert_args(args, to_bits=False, to_int=True):
    if len(args)==1:
        args = args[0]
    b1,b2,b3 = tuple((int(a) if to_int else a) for a in args[-3:])
    i_s = args[:-3]
    size = len(i_s)//3
    i1, i2, i3 = i_s[:size], i_s[size:2*size], i_s[2*size:]
    if not to_bits:
        i1, i2, i3 = convert(i1), convert(i2), convert(i3)
    return i1,i2,i3,b1,b2,b3

def bits_to_int_clause(func):
    def inner(*args):
        phi = args[0]
        return func(phi, *convert_args(args[1:]))
    return inner

def bits_to_int_witness(func):
    def inner(*args):
        x = convert(args)
        return func(x)
    return inner

def eval(phi, z):
    '''
    phi is a 3-CNF formula represented as a list of tuples.
        each tuple represent a clause, and each element in the tuple can be a <number> or -<number> (negative number, representing "not") 
    '''
    return any(all(z[replace_n(x)] if x>=0 else not z[replace_n(-x)] for x in clause) for clause in phi)
    for clause in phi:
        if not any(z[replace_n(x)] if x>=0 else not z[replace_n(-x)] for x in clause):
            return False
    return True

def replace_zero(i):
    return i if i != 0 else n
def replace_n(i):
    return i if i != n else 0

@bits_to_int_clause
def clause(phi, i1, i2, i3, b1, b2, b3):
    '''
        tells whether or not exists in phi a clause that satisfies the assingment i1:b1, i2:b2, i3:b3
    '''
    i1, i2, i3 = replace_zero(i1), replace_zero(i2), replace_zero(i3)
    phi = [tuple(sorted(x)) for x in phi]
    tmp = tuple(sorted((-i1*pow(-1, b1), -i2*pow(-1, b2), -i3*pow(-1, b3))))
    return  int(tmp in phi)

def witness(w):
    return bits_to_int_witness(lambda x: w[replace_n(x)])


if __name__ == "__main__":
    phi = [(1,2,3), (-2, 4, 3), (-1, 2, -3)]
    z = [1, 0, 0, 1]
    print(f'{eval(phi, z)=}')
    print(f'{clause(phi,0,1,1,0,1,1,0,1,0)=}')
    print(f'{witness(z)(1,1)=}')
    print(f'{witness(z)(1,0)=}')
    print(f'{witness(z)(0,1)=}')
    print(f'{witness(z)(0,0)=}')