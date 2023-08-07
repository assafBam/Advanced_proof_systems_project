import copy
from typing import Dict, List, Union, Callable
from IPython.display import display, Latex
from parameters import F
import numpy as np
from collections.abc import Iterable
import json

class MVLinear:
    """
    A Sparse Representation of a multi-linear polynomial.
    """

    def __init__(self, num_variables: int, term: Dict[int, int]):
        """
        :param num_variables: total number of variables
        :param term: the terms of the polynomial. Index is the binary representation of the vector, where
        the nth bit represents the nth variable. For example, 0b1011 represents term x0x1x3. The value represents the coefficient.
        """

        self.num_variables = num_variables
        self.terms: Dict[int, int] = dict()
        for k,v in term.items():
            if k >> self.num_variables > 0:
                raise ValueError("Term is out of range.")
            if v == 0:
                continue
            if k in self.terms:
                self.terms[k] = (self.terms[k] + term[k])
            else:
                self.terms[k] = term[k]

    def __repr__(self):
        limit = 8
        s = ""
        s += "MVLinear("
        for k in self.terms:
            s += " + "
            if limit == 0:
                s += "..."
                break
            s += str(self.terms[k])
            if k != 0:
                s += "*"

            i = 0
            while k != 0:
                if k & 1 == 1:
                    s += "x" + str(i)
                i += 1
                k >>= 1
            limit -= 1
        s += ")"
        return s

    def __copy__(self) -> 'MVLinear':
        t = self.terms.copy()
        return MVLinear(self.num_variables, t)

    __deepcopy__ = __copy__

    def _assert_same_type(self, other: 'MVLinear'):
        if not isinstance(other, MVLinear):
            print(type(other))
            raise TypeError("MVLinear can only be added with MVLinear")

    def __add__(self, other: Union['MVLinear', int]) -> 'MVLinear':
        if type(other) is int:
            other = MVLinear(self.num_variables, {0b0: other})
        self._assert_same_type(other)

        ans: MVLinear = copy.copy(self)
        ans.num_variables = max(self.num_variables, other.num_variables)

        for k in other.terms:
            if k in self.terms:
                ans.terms[k] = (ans.terms[k] + other.terms[k])
                if ans.terms[k] == 0:
                    ans.terms.pop(k)
            else:
                ans.terms[k] = other.terms[k]

        return ans

    __radd__ = __add__

    def __sub__(self, other: Union['MVLinear', int]) -> 'MVLinear':
        if type(other) is int:
            other = MVLinear(self.num_variables, {0b0: other})
        self._assert_same_type(other)

        ans: MVLinear = copy.copy(self)
        ans.num_variables = max(self.num_variables, other.num_variables)

        for k in other.terms:
            if k in self.terms:
                ans.terms[k] = (ans.terms[k] - other.terms[k])
                if ans.terms[k] == 0:
                    ans.terms.pop(k)
            else:
                ans.terms[k] = (- other.terms[k])

        return ans

    def __neg__(self):
        return 0 - self

    def __rsub__(self, other):
        if type(other) is int:
            other = MVLinear(self.num_variables, {0b0: other})
        return other - self

    def __mul__(self, other: Union['MVLinear', int]) -> 'MVLinear':
        if type(other) is int:
            other = MVLinear(self.num_variables, {0b0: F(other)})
        if isinstance(other, F):
            other = MVLinear(self.num_variables, {0b0: other})
        self._assert_same_type(other)

        terms: Dict[int, int] = dict()
        # naive n^2 poly multiplication where n is number of terms
        for sk in self.terms:  # the term of self
            for ok in other.terms:  # the term of others
                if sk & ok > 0:
                    raise ArithmeticError("The product is no longer multi-linear function.")
                nk = sk + ok  # the result term
                if nk in terms:
                    terms[nk] = (terms[nk] + self.terms[sk] * other.terms[ok])
                else:
                    terms[nk] = (self.terms[sk] * other.terms[ok])
                if terms[nk] == 0:
                    terms.pop(nk)

        ans = MVLinear(max(self.num_variables, other.num_variables), terms,)
        return ans

    __rmul__ = __mul__  # commutative

    def eval(self, at: List[int]) -> int:
        s = F(0)
        for term in self.terms:
            i = 0
            val = self.terms[term]
            while term != 0:
                if term & 1 == 1:
                    val = (val * at[i])
                if val == 0:
                    break
                term >>= 1
                i += 1
            s = (s + val)

        return s

    def eval_bin(self, at: int) -> int:
        """
        Evaluate the polynomial where the arguments are in {0,1}. The ith argument is the ith bit of the polynomial.
        :param at: polynomial argument in binary form
        :return: polynomial evaluation
        """
        if at > 2 ** self.num_variables:
            raise ArithmeticError("Number of variables is larger than expected")
        args = [0 for _ in range(self.num_variables)]
        for i in range(self.num_variables):
            args[i] = at >> i & 1
        return self.eval(args)

    def __call__(self, *args, **kwargs) -> int:
        if len(args) == 0:
            return self.eval([])
        if isinstance(args[0], list):
            return self.eval(args[0])
        if isinstance(args[0], np.ndarray):
            return self.eval(args[0])
        if isinstance(args[0], set):
            if len(args[0]) != 1:
                raise TypeError("Binary representation should have only one element. ")
            return self.eval_bin(next(iter(args[0])))
        if isinstance(args[0], Iterable):
            return self.eval(args[0])
        return self.eval(list(args))

    def latex(self):
        s = ""
        for k in self.terms:
            s += " + "
            if self.terms[k] != 1:
                s += str(self.terms[k])

            i = 0
            while k != 0:
                if k & 1 == 1:
                    s += "x_{" + str(i) + "}"
                i += 1
                k >>= 1

        s = s[3:]
        display(Latex('$' + s + '$'))

    def __eq__(self, other: 'MVLinear') -> bool:
        diff = self - other
        return len(diff.terms) == 0  # zero polynomial

    def __getitem__(self, item):
        if item in self.terms.keys():
            return self.terms[item]
        else:
            return 0

    def eval_part(self, args: List[int]) -> 'MVLinear':
        """
        Evaluate part of the arguments of the multilinear polynomial.
        :param args: the arguments at beginning
        :return:
        """
        s = len(args)
        if s > self.num_variables:
            raise ValueError("len(args) > self.num_variables")
        new_terms: Dict[int, int] = dict()
        for t, v in self.terms.items():
            for k in range(s):
                if t & (1 << k) > 0:
                    v = v * (args[k])
                    t = t & ~(1 << k)
            t_shifted = t >> s
            if t_shifted not in new_terms:
                new_terms[t_shifted] = 0
            new_terms[t_shifted] = (new_terms[t_shifted] + v)
        return MVLinear(self.num_variables - len(args), new_terms)

    def collapse_left(self, n: int) -> 'MVLinear':
        """
        Remove redundant unused variable from left.
        :param n: number of variables to collpse
        :return:
        """
        new_terms: Dict[int, int] = dict()
        mask = (1 << n) - 1
        for t, v in self.terms.items():
            if t & mask > 0:
                raise ArithmeticError("Cannot collapse: Variable exist. ")
            new_terms[t >> n] = v
        return MVLinear(self.num_variables - n, new_terms)

    def collapse_right(self, n: int) -> 'MVLinear':
        """
        Remove redundant unused variable from right.
        :param n: number of variables to collpse
        :return:
        """
        new_terms: Dict[int, int] = dict()
        mask = ((1 << n) - 1) << (self.num_variables - n)
        anti_mask = (1 << (self.num_variables - n)) - 1
        for t, v in self.terms.items():
            if t & mask > 0:
                raise ArithmeticError("Cannot collapse: Variable exist. ")
            new_terms[t & anti_mask] = v
        return MVLinear(self.num_variables - n, new_terms)
    

def makeMVLinearConstructor(num_variables: int) -> Callable[[Dict[int, int]], MVLinear]:
    """
    Return a function that outputs MVLinear
    :param num_variables: total number of variables
    :return: Callable[[Dict[int, int]], MVLinear]
    """

    def f(term: Dict[int, int]) -> MVLinear:
        return MVLinear(num_variables, term)

    return f

def jsonDump(mvl: MVLinear, file: str):
    terms = {k:int(v) for k,v in mvl.terms.items()}
    d={"n": mvl.num_variables, "terms":terms}
    with open(file,'w') as fd:
        json.dump(d, fd, indent=4)

def jsonLoad(file:str)->MVLinear:
    with open(file, 'r') as fd:
        d=json.load(fd)
        n = int(d['n'])
        terms = {int(k):F(int(v)) for k,v in d['terms'].items()}
        return MVLinear(n, terms)

