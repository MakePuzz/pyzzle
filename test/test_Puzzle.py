import sys
import unittest

import numpy as np

sys.path.append("../")
from pyzzle import Puzzle, Dictionary

class TestPuzzleMethods(unittest.TestCase):

    def test_add(self):
        puzzle = Puzzle(5,5)
        dic = Dictionary(["TEST", "ESTA", "STEM", "ME", "ET"])
        puzzle.import_dict(dic)
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
        ])
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
        used_words_answer = ["TEST", "ESTA", "STEM", "ME", "ET",'','','','','','','','','','','','','','','','','','','','']
        self.assertTrue(np.all(puzzle.cell==cell_answer))
        self.assertTrue(np.all(puzzle.cover==cover_answer))
        self.assertTrue(np.all(puzzle.enable==enable_answer))
        self.assertTrue(np.all(puzzle.used_words==used_words_answer))

    def test_drop(self):
        puzzle = Puzzle(5,5)
        dic = Dictionary(["TEST", "ESTA", "STEM", "ME", "ET"])
        puzzle.import_dict(dic)

        puzzle.add(0, 0, 0, "TEST")
        puzzle.add(1, 1, 0, "ESTA")
        puzzle.add(0, 0, 2, "STEM")
        puzzle.add(1, 3, 2, "ME")
        puzzle.add(0, 3, 3, "ET")
        puzzle.drop("TEST")
        puzzle.drop("ME")
        cell_answer = np.array([
            ['', '', 'S', '', ''],
            ['E', 'S', 'T', 'A', ''],
            ['', '', 'E', '', ''],
            ['', '', 'M', 'E', ''],
            ['', '', '', 'T', '']    
        ])
        cover_answer = [
            [0, 0, 1, 0, 0],
            [1, 1, 2, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0]
        ]
        enable_answer = [
            [ True,  True,  True,  True,  True],
            [ True,  True,  True,  True, False],
            [ True,  True,  True, False,  True],
            [ True,  True,  True,  True,  True],
            [ True,  True, False,  True,  True]
        ]
        used_words_answer = ['']*25
        used_words_answer[0] = "ESTA"
        used_words_answer[1] = "STEM"
        used_words_answer[2] = "ET"
        self.assertTrue(np.all(puzzle.cell==cell_answer))
        self.assertTrue(np.all(puzzle.cover==cover_answer))
        self.assertTrue(np.all(puzzle.enable==enable_answer))
        self.assertTrue(np.all(puzzle.used_words==used_words_answer))


if __name__ == '__main__':
    unittest.main()