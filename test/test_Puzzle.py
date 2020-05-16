import sys
import unittest

import numpy as np

sys.path.append("../")
from pyzzle import Puzzle

class TestPuzzleMethods(unittest.TestCase):

    def test_add(self):
        puzzle = Puzzle(5,5)
        puzzle.add(0, 0, 0, "TEST")
        puzzle.add(1, 1, 0, "ESTA")
        puzzle.add(0, 0, 2, "STEM")
        puzzle.add(1, 3, 2, "ME")
        puzzle.add(0, 3, 3, "ET")
        cell_answer = np.array([
            ['T', '', 'S', '', ''],
            ['E', 'S', 'T', 'A', ''],
            ['S', '', 'E', '', ''],
            ['T', '', 'M', 'E', ''],
            ['', '', '', 'T', '']    
        ], dtype="<U1")
        cover_answer = [
            [1, 0, 1, 0, 0],
            [2, 1, 2, 1, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 2, 2, 0],
            [0, 0, 0, 1, 0]
        ]
        enable_answer = [
            [ True,  True,  True,  True,  True],
            [ True,  True,  True,  True, False],
            [ True,  True,  True, False,  True],
            [ True, False,  True,  True, False],
            [False,  True, False,  True,  True]
        ]
        self.assertTrue(np.all(puzzle.cell==cell_answer))
        self.assertTrue(np.all(puzzle.cover==cover_answer))
        self.assertTrue(np.all(puzzle.enable==enable_answer))

if __name__ == '__main__':
    unittest.main()