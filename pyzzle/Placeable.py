import numpy as np


class Placeable:
    def __init__(self, width, height, dic, mask=None):
        self.size = 0
        self.width = width
        self.height = height
        self.ori, self.i, self.j, self.k = [], [], [], []
        self.inv_p = np.full((2, self.height, self.width, 0), np.nan, dtype="int")

        self._compute(dic.word, mask)

    def _compute(self, word, mask=None, base_k=0):
        if mask is not None:
            mask = np.array(mask)
        if type(word) is str:
            word = [word]
        if self.size == 0 or base_k != 0:
            ap = np.full((2, self.height, self.width, len(word)), np.nan, dtype="int")
            self.inv_p = np.append(self.inv_p, ap, axis=3)
        for ori in (0,1):
            for k,w in enumerate(word):
                w_len = len(w)
                if ori == 0:
                    i_max = self.height - w_len + 1
                    j_max = self.width
                elif ori == 1:
                    i_max = self.height
                    j_max = self.width - w_len + 1
                for i in range(i_max):
                    for j in range(j_max):
                        if mask is not None:
                            if ori == 0 and np.any(mask[i:i+w_len, j] == True):
                                continue
                            if ori == 1 and np.any(mask[i, j:j+w_len] == True):
                                continue
                        self.inv_p[ori,i,j,base_k+k] = len(self.ori)
                        self.ori.append(ori)
                        self.i.append(i)
                        self.j.append(j)
                        self.k.append(base_k+k)
        self.size = len(self.k)

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        if type(key) in (int, np.int):
            return {"ori": self.ori[key], "i": self.i[key], "j": self.j[key], "k": self.k[key]}
        if type(key) is str:
            return eval(f"self.{key}")

    def __str__(self):
        return f"ori:{self.ori}, i:{self.i}, j:{self.j}, k:{self.k}"
