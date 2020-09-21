import unittest

import numpy as np

from pyzzle import Placeable

class TestPlaceable(unittest.TestCase):
    """Test the Placeable class."""

    def test_add(self):
        plc = Placeable(width=5, height=5)
        plc.add("HOGE")
        ori = sorted([0]*10 + [1]*10)
        i = sorted([0,1,2,3,4]*2 + [0]*5 + [1]*5)
        j = sorted([0,1,2,3,4]*2 + [0]*5 + [1]*5)
        k = [0]*20
        word = ["HOGE"]*20
        self.assertTrue(plc.size == 20)
        self.assertTrue(sorted(plc.ori) == ori)
        self.assertTrue(sorted(plc.i) == i)
        self.assertTrue(sorted(plc.j) == j)
        self.assertTrue(plc.k == k)
        self.assertTrue(plc.word == word)

    def test_add_with_mask(self):
        mask = np.array([
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, True]
        ])
        plc = Placeable(width=5, height=5, mask=mask)
        plc.add("HOGE")
        ori = sorted([0]*9 + [1]*9)
        i = sorted([0,1,2,3,4,0,1,2,3] + [0]*5 + [1]*4)
        j = sorted([0,1,2,3,4,0,1,2,3] + [0]*5 + [1]*4)
        k = [0]*18
        word = ["HOGE"]*18
        self.assertTrue(plc.size == 18)
        self.assertTrue(sorted(plc.ori) == ori)
        self.assertTrue(sorted(plc.i) == i)
        self.assertTrue(sorted(plc.j) == j)
        self.assertTrue(plc.k == k)
        self.assertTrue(plc.word == word)