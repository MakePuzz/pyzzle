import unittest

import numpy as np

from pyzzle import Dictionary

class TestDictionary(unittest.TestCase):
    """Test the Dictionary class."""

    def test_add(self):
        d = Dictionary()
        d += "word1"
        d += ["word2", 1]
        self.assertTrue(["word1", "word2"], d.word)
        self.assertTrue([0, 1], d.weight)

    def test_add_with_dictionary(self):
        d = Dictionary(word="word1")
        d += Dictionary(word="word2", weight=1)
        self.assertTrue(["word1", "word2"], d.word)
        self.assertTrue([0, 1], d.weight)

    def test_add_word(self):
        d = Dictionary()
        d.add("word1")
        d.add("word2", 1)
        self.assertTrue(["word1", "word2"], d.word)
        self.assertTrue([0, 1], d.weight)

    def test_add_multiple_words(self):
        d = Dictionary()
        d.add(["word1", "word2", "word3"], [0, 1, 2])
        self.assertTrue(["word1", "word2", "word3"], d.word)
        self.assertTrue([0, 1, 2], d.weight)

    def test_add_duplicated_word(self):
        d = Dictionary()
        d.add("word1", 0)
        d.add("word1", 1)
        self.assertTrue(["word1"], d.word)
        self.assertTrue([1], d.weight)

    def test_iter(self):
        d = Dictionary()
        d.add(["word1", "word2", "word3"], [0, 1, 2])
        words = []
        weights = []
        for wo, we in d:
            words.append(wo)
            weights.append(we)
        self.assertTrue(["word1", "word2", "word3"], words)
        self.assertTrue([0, 1, 2], weights)

    def test_size_property(self):
        d = Dictionary(word="word1")
        self.assertTrue(1, d.size)

    def test_w_len_property(self):
        d = Dictionary(word="word1")
        self.assertTrue([5], d.w_len)