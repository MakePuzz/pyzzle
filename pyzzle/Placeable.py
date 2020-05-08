import numpy as np


class Placeable:
    def __init__(self, width, height, dic, msg=True):
        self.size = 0
        self.width = width
        self.height = height
        self.div, self.i, self.j, self.k = [], [], [], []
        self.invP = np.full((2, self.height, self.width, 0), np.nan, dtype="int")

        self._compute(dic.word)

        if msg is True:
            print(f"Imported Dictionary name: `{dic.name}`, size: {dic.size}")
            print(f"Placeable size : {self.size}")

    def _compute(self, word, baseK=0):
        if type(word) is str:
            word = [word]
        if self.size is 0 or baseK is not 0:
            ap = np.full((2, self.height, self.width, len(word)), np.nan, dtype="int")
            self.invP = np.append(self.invP, ap, axis=3)
        for div in (0,1):
            for k,w in enumerate(word):
                if div == 0:
                    iMax = self.height - len(w) + 1
                    jMax = self.width
                elif div == 1:
                    iMax = self.height
                    jMax = self.width - len(w) + 1
                for i in range(iMax):
                    for j in range(jMax):
                        self.invP[div,i,j,baseK+k] = len(self.div)
                        self.div.append(div)
                        self.i.append(i)
                        self.j.append(j)
                        self.k.append(baseK+k)
        self.size = len(self.k)

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        if type(key) in (int, np.int):
            return {"div": self.div[key], "i": self.i[key], "j": self.j[key], "k": self.k[key]}
        if type(key) is str:
            return eval(f"self.{key}")
