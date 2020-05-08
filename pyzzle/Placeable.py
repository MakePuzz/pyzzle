import numpy as np


class Placeable:
    def __init__(self, width, height, dic, msg=True):
        self.size = 0
        self.width = width
        self.height = height
        self.ori, self.i, self.j, self.k = [], [], [], []
        self.inv_p = np.full((2, self.height, self.width, 0), np.nan, dtype="int")

        self._compute(dic.word)

        if msg is True:
            print(f"Imported Dictionary name: `{dic.name}`, size: {dic.size}")
            print(f"Placeable size : {self.size}")

    def _compute(self, word, base_k=0):
        if type(word) is str:
            word = [word]
        if self.size is 0 or base_k is not 0:
            ap = np.full((2, self.height, self.width, len(word)), np.nan, dtype="int")
            self.inv_p = np.append(self.inv_p, ap, axis=3)
        for ori in (0,1):
            for k,w in enumerate(word):
                if ori == 0:
                    i_max = self.height - len(w) + 1
                    j_max = self.width
                elif ori == 1:
                    i_max = self.height
                    j_max = self.width - len(w) + 1
                for i in range(i_max):
                    for j in range(j_max):
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
