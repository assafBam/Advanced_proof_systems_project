import numpy as np
import functools
from MVL_LDE import lde, I0
from CNF3 import clause
from parameters import baseSubset
import concurrent.futures
import itertools
from tqdm import tqdm
import os
import json
from polynomial import jsonDump, jsonLoad

def calculateQ(args):
    Q, rs, x, h = args
    tmp = Q(*rs, x, *h)
    return tmp

def sumGF2(Q, rs, x, m):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        args = ((Q, rs, x, h) for h in itertools.product(baseSubset, repeat=m))
        results = tuple(tqdm(executor.map(calculateQ, args), total=2**m))
        return functools.reduce(lambda a, b: a + b, results)    

def get_phi_lde(formula, n):
    @functools.cache
    def inner(formula:tuple, n):
        def phi(*args):
            return clause(formula, *args)
        phi_hat = lde(f=phi, H=baseSubset, m=int(3 * np.ceil(np.log2(n)) + 3))
        return phi_hat
    
    ldes = None
    if f'phi_lde{n}.json' in os.listdir():
        fd = open(f'phi_lde{n}.json', 'r')
        ldes = json.load(fd)
        fd.close()
        if str(formula) in ldes:
            lde_res = jsonLoad(ldes[str(formula)])
            return lde_res
    else:
        ldes = {}
    res = inner(tuple(formula),n)
    i=0
    while f'phi_lde{n}_{i}.json' in os.listdir(): i+=1
    jsonDump(res, f'phi_lde{n}_{i}.json')
    ldes[str(formula)] = f'phi_lde{n}_{i}.json'
    fd = open(f'phi_lde{n}.json', 'w')
    json.dump(ldes, fd)
    return res
    

def fancy_zip(l1, l2):
    i = 0
    for e1, e2 in zip(l1, l2):
        yield e1, e2
        i += 1
    if i < len(l2):
        for k in range(i, len(l2)):
            yield np.array([]), l2[k]

def prod(seq):
    return functools.reduce(lambda a,b: a*b, seq)

def innerProduct_sums(m1, m2):
    for v in reversed(m2):
        m1 = v @ m1
    return m1

def theoretic_range(n):
    return range(1, int(n+1))

def GF_get_bits(z):
    bytes_z = reversed(z.tobytes())
    bits = []
    for byte in bytes_z:
        bits.extend(list(format(byte, 'b').zfill(8)))
    return tuple(int(bit) for bit in reversed(bits))

def int_get_bits(i):
    return tuple(int(c) for c in format(i, 'b')[::-1])

def get_at(seq, i):
    if i<len(seq): return seq[i]
    return 0


def get_delimiters(z, n=2):
    
    def inner(z: tuple, n):
        m = len(z)
        M = 2**m
        dels = []
        for j in range(n):
            y = []
            for k in range(int(np.round(np.power(M, (1. / n))))):
                bits_k = int_get_bits(k)
                Is = tuple(I0(get_at(z, i), get_at(bits_k, i - j * (m // n))) for i in range(j * (m // n), (j + 1) * (m // n)))
                non_empty_arrays = [arr for arr in Is if np.any(arr)]  # Filter out empty arrays
                if non_empty_arrays:
                    y.append(prod(non_empty_arrays))
            dels.append(y)
        return tuple(reversed(dels))
    
    return inner(tuple(z), n)

