# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:56:16 2023

@author: Assaf
"""

from parameters import F
from polynomial import MVLinear as MVL
from itertools import product
import functools
from tqdm import tqdm

def I0(xi, yi):
    if isinstance(xi, F) or isinstance(yi, F):
        return F(1) - F(xi) - F(yi) + 2 * F(xi) * F(yi)
    return 1 - xi - yi + 2 * xi * yi


def I(h,n):
    return functools.reduce(lambda a,b: a*b, (MVL(n, {1<<i:-F(1), 0:F(1)-F(h[i])}) for i in range(n)))

def lde(f, H, m):
    return sum((I(h,m)*f(*h) for h in tqdm(product(H, repeat=m), total=len(H)**m)), start=MVL(m,{0:F(0)}))

if __name__ == '__main__':
    from parameters import phi
    from CNF3 import clause
 
    phi_hat = lde(lambda *args: clause(phi, *args), F([0,1]), 9)
    print(phi_hat)