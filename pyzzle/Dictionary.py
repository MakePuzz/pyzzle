import os
import collections
import shutil

import numpy as np

class Dictionary:
    def __init__(self, fpath=None, msg=True):
        self.fpath = fpath
        self.size = 0
        self.name = ''
        self.word = []
        self.weight = []
        self.w_len = []
        self.removed_words = []
        if fpath is not None:
            self.name = os.path.basename(fpath)[:-4]
            self.read(fpath)

        # Message
        if msg is True:
            print("Dictionary object has made.")
            print(f" - file path         : {self.fpath}")
            print(f" - dictionary size   : {self.size}")
            if self.size > 0:
                print(f" - top of dictionary : {self[0]}")

    def __getitem__(self, key):
        return {'word': self.word[key], 'weight': self.weight[key], 'len': self.w_len[key]}

    def __str__(self):
        return self.name

    def __len__(self):
        return self.size

    def getK(self, word):

        return np.where(self.word == word)[0][0]

    def include(self, word):
        return word in self.word

    def add(self, word=None, weight=None, fpath=None, msg=True):
        if (word,fpath) == (None,None):
            raise ValueError("'word' or 'fpath' must be specified")
        if word is not None and fpath is not None:
            raise ValueError("'word' or 'fpath' must be specified")
        if fpath is not None:
            self.read(fpath)
        if word is not None:
            if type(word) is str:
                    word = [word]
            if weight is None:
                weight = [0]*len(word)
            else:
                if type(weight) is int:
                    weight = [weight]
                if len(word) != len(weight):
                    raise ValueError(f"'word' and 'weight' must be same size")

            for wo, we in zip(word, weight):
                if self.include(wo) and msg is True:
                    print(f"The word '{wo}' already exists")
                self.word.append(wo)
                self.weight.append(we)
                self.w_len.append(len(wo))
                self.size += 1

    def read(self, fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            data = f.readlines()

        # Remove "\n"
        def removed_new_line_code(word):
            line = word.rstrip("\n").split(" ")
            if len(line) == 1:
                line.append(0)
            line[1] = int(line[1])
            return line

        dic_list = list(map(removed_new_line_code, data))
        word = [d[0] for d in dic_list]
        weight = [d[1] for d in dic_list]
        self.add(word, weight)

    def delete_unusable_words(self, msg=True):
        """
        This method checks words in the dictionary and erases words that can not cross any other words.
        """
        merged_words = "".join(self.word)
        counts = collections.Counter(merged_words)
        for i, w in enumerate(self.word[:]):
            char_value = 0
            for char in set(w):
                char_value += counts[char]
            if char_value == len(w):
                self.removed_words.append(w)
                del self.word[i]
                del self.weight[i]
                del self.w_len[i]
                self.size -= 1
                if msg is True:
                    print(f"'{w}' can not cross with any other words")

    def calc_weight(self, msg=True):
        """
        Calculate word weights in the dictionary.
        """
        merged_words = "".join(self.word)
        counts = collections.Counter(merged_words)

        for i, w in enumerate(self.word):
            for char in w:
                self.weight[i] += counts[char]

        if msg is True:
            print("All weights are calculated.")
            print("TOP 5 characters:")
            print(counts.most_common()[:5])
            idx = sorted(range(self.size), key=lambda k: self.weight[k], reverse=True)[:5]
            print("TOP 5 words:")
            print(np.array(self.word)[idx])
