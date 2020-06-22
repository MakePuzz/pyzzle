import os, copy
from glob import glob
from pathlib import PurePath
import collections

import numpy as np

from pyzzle.Word import Word

class Dictionary:
    class Dataset:
        dict_dir = str(PurePath(__file__).parent/PurePath("dict"))
        dict_list = list(map(lambda x: PurePath(x).stem, glob(f"{dict_dir}/*.txt")))           

        def __getattr__(self, key):
            if key not in (self.dict_list):
                raise AttributeError(f"{key} must be an element of the 'dict_list'")
            return Dictionary(f"{self.dict_dir}/{key}.txt")
        
        def __getitem__(self, key):
            return Dictionary(f"{self.dict_dir}/{key}.txt")

    dataset = Dataset()

    def __init__(self, dict_specifier=None, word=None, weight=None):
        self.dict_specifier = dict_specifier
        self.word = []
        self.removed_words = []
        self._i = 0
        if isinstance(dict_specifier, (list, np.ndarray)):
            self.add(dict_specifier)
        if isinstance(dict_specifier, str):
            self.read(dict_specifier)
        if word is not None:
            self.add(word, weight)

    @property
    def size(self):
        return len(self.word)
    
    @property
    def weight(self):
        return list(map(lambda x: x.weight, self.word))

    @property
    def w_len(self):
        return list(map(len, self.word))

    def __getitem__(self, key):
        return {'word': self.word[key], 'weight': self.word[key].weight, 'len': self.w_len[key]}

    def __str__(self):
        return str({"words": self.word, "weight": self.weight})

    def __len__(self):
        return self.size

    def __add__(self, other):
        new_dict = copy.deepcopy(self)
        if isinstance(other, Dictionary):
            for wo, we in other:
                new_dict.add(wo, we)
        if isinstance(other, str):
            new_dict.add(other, 0)
        if isinstance(other, (tuple, list)):
            new_dict.add(other[0], other[1])
        if isinstance(other, dict):
            new_dict.add(other["word"], other["weight"])
        return new_dict

    def __iter__(self):
        return self

    def __next__(self):
        if self._i == self.size:
            self._i = 0
            raise StopIteration()
        word = self.word[self._i]
        self._i += 1
        return word, word.weight
    
    def get_k(self, word):
        return np.where(self.word == word)[0][0]

    def include(self, word):
        return word in self.word

    def add(self, word=None, weight=None, dict_specifier=None):
        if (word, dict_specifier) == (None, None):
            raise ValueError("'word' or 'dict_specifier' must be specified")
        if word is dict_specifier is not None:
            raise ValueError("'word' or 'dict_specifier' must be specified")
        if dict_specifier is not None:
            self.read(dict_specifier)
        if word is not None:
            if isinstance(word, str):
                word = [word]
            if weight is None:
                weight = [0]*len(word)
            if isinstance(weight, (int, float)):
                weight = [weight]
            if len(word) != len(weight):
                raise ValueError(f"'word' and 'weight' must be same size")
            for wo, we in zip(word, weight):
                if self.include(wo) is True: # replace the weight
                    self.word[self.word.index(wo)].weight = we
                else:
                    self.word.append(Word(wo, we))

    def read(self, dict_specifier):
        with open(dict_specifier, 'r', encoding='utf-8') as f:
            data = f.readlines()
        data = [l for l in data if l != os.linesep]

        # Remove new_line_code
        def removed_new_line_code(word):
            line = word.rstrip(os.linesep).split(" ")
            if len(line) == 1:
                line.append(0) # weight = 0
            line[1] = int(line[1])
            return line

        dic_list = list(map(removed_new_line_code, data))
        word = [d[0] for d in dic_list]
        weight = [d[1] for d in dic_list]
        self.add(word, weight)

    def delete_unusable_words(self):
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

    def calc_weight(self):
        """
        Calculate word weights in the dictionary.
        """
        merged_words = "".join(self.word)
        counts = collections.Counter(merged_words)

        for i, w in enumerate(self.word):
            for char in w:
                self.word[i].weight += counts[char]
