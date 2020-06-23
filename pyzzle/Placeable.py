import numpy as np
from collections import Counter


class Placeable:
    def __init__(self, width, height, words=None, mask=None):
        self.size = 0
        self.width = width
        self.height = height
        self.ori, self.i, self.j, self.k, self.word = [], [], [], [], []

        if words is not None:
            self._compute(words, mask=mask)

    def _compute(self, word, mask=None, base_k=0):
        if isinstance(word, str):
            word = [word]
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
                        self.size += c

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        if type(key) in (int, np.int):
            return {"ori": self.ori[key], "i": self.i[key], "j": self.j[key], "word": self.word[key]}
        if type(key) is str:
            return eval(f"self.{key}")

    def __str__(self):
        return f"ori:{self.ori}, i:{self.i}, j:{self.j}, word:{self.word}"
