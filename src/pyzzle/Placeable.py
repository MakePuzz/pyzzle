import sys

import numpy as np
from collections import Counter


class Placeable:
    """
    Placeable class stores the positions where words can be placed.
    """
    def __init__(self, width, height, words=None, mask=None):
        self.width = width
        self.height = height
        self.ori = []
        self.i = []
        self.j = []
        self.k = []
        self.word = []
        self.mask = mask
        if isinstance(mask, list):
            self.mask = np.array(mask)
            if mask.shape != (self.height, self.width):
                raise ValueError("The shape of the mask must be the same as (height, width)")

        if words is not None:
            self.add(words, mask=mask)

    def __sizeof__(self):
        size = sys.getsizeof(self.ori) + sys.getsizeof(self.i) + sys.getsizeof(self.j) + sys.getsizeof(self.k) + sys.getsizeof(self.word)
        size += sys.getsizeof(self.width) + sys.getsizeof(self.height) + sys.getsizeof(self.mask)
        return size
    
    def add(self, word, mask=None, base_k=0):
        if isinstance(word, str):
            word = [word]
        if mask is None:
            mask = self.mask
        if isinstance(mask, list):
            mask = np.array(mask)
            if mask.shape != (self.height, self.width):
                raise ValueError("The shape of the mask must be the same as (height, width)")
        len_arr = np.vectorize(len)(word)
        len_count = Counter(len_arr)
        for ori in (0, 1):
            for l, c in len_count.items():
                if ori == 0:
                    i_max = self.height - l + 1
                    j_max = self.width
                if ori == 1:
                    i_max = self.height
                    j_max = self.width - l + 1
                for i in range(i_max):
                    for j in range(j_max):
                        if mask is not None:
                            if ori == 0 and np.any(mask[i:i+l, j] == True):
                                continue
                            if ori == 1 and np.any(mask[i, j:j+l] == True):
                                continue
                        self.ori += [ori]*c
                        self.i += [i]*c
                        self.j += [j]*c
                        self.k += (np.where(len_arr == l)[0] + base_k).tolist()
                        self.word += np.array(word, dtype=object)[np.where(len_arr == l)[0]].tolist()
    
    @property
    def size(self):
        return len(self.word)

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        if type(key) in (int, np.int):
            return {"ori": self.ori[key], "i": self.i[key], "j": self.j[key], "word": self.word[key]}
        if type(key) is str:
            return eval(f"self.{key}")

    def __str__(self):
        return f"ori:{self.ori}, i:{self.i}, j:{self.j}, word:{self.word}"

    def __repr__(self):
        return f"ori:{self.ori}, i:{self.i}, j:{self.j}, word:{self.word}"
