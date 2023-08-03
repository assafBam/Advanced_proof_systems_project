import numpy as np
from collections import namedtuple

LinCode = namedtuple('LinCode', ['shape', 'codeword'])

class C0:
    @staticmethod
    def __new__(_, w):
        shape = w.shape if isinstance(w, np.ndarray) else len(w)
        cw = np.concatenate((w, [sum(w)%2]))
        return LinCode(shape, cw)
        
    
    def __eq__(self, other):
        if isinstance(other, C0):
            return np.array_equal(self.cw, other.cw)
        raise NotImplementedError
    
    @staticmethod
    def test(codeword):
        w=codeword.codeword[:codeword.shape]
        return C0(w) == codeword
    
    
class TensorCode:
    NUM_TO_CHECK = 3
    def __init__(self, baseCode=C0):
        self.code=baseCode
    
    def __call__(self, w):
        if isinstance(w, np.ndarray):
            shape = w.shape
            rows = np.apply_along_axis(lambda x: self.code(x).codeword, 1, w)
            cols = np.apply_along_axis(lambda x: self.code(x).codeword, 0, rows)
            return LinCode(shape, cols)
        w = np.array(w)
        dim = int(np.sqrt(w.shape[0]))
        w = w.reshape((dim, dim))
        return self(w)
        
    def get_word(self, codeword: LinCode):
        shape = codeword.shape
        return codeword.codeword[:shape[0],  : shape[1]]
    
    def test(self, codeword: LinCode):
        cols, rows = codeword.shape
        w=codeword.codeword[:cols, :rows]
        for i in range(self.NUM_TO_CHECK):
            # choose random row
            row_idx = np.random.randint(rows)
            r_w = w[row_idx,:]
            tmp_cw = self.code(r_w)
            if not np.array_equal(tmp_cw.codeword, codeword.codeword[row_idx, :]):
                return False
            # choose random column
            col_idx = np.random.randint(cols)
            c_w = w[:, col_idx]
            tmp_cw = self.code(c_w)
            if not np.array_equal(tmp_cw.codeword, codeword.codeword[:, col_idx]):
                return False
        return True