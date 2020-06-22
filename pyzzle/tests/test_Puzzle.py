import unittest
from unittest import mock

import numpy as np


class TestPuzzle(unittest.TestCase):
    from pyzzle import Word
    """Test the Puzzle class."""
    cell = np.array([
        ['T', '', 'S', '', ''],
        ['E', 'S', 'T', 'A', ''],
        ['S', '', 'E', '', ''],
        ['T', '', 'M', 'E', ''],
        ['', '', '', 'T', '']
    ])
    cover = np.array([
        [1, 0, 1, 0, 0],
        [2, 1, 2, 1, 0],
        [1, 0, 1, 0, 0],
        [1, 0, 2, 2, 0],
        [0, 0, 0, 1, 0]
    ])
    enable = np.array([
        [True,  True,  True,  True,  True],
        [True,  True,  True,  True, False],
        [True,  True,  True, False,  True],
        [True, False,  True,  True, False],
        [False,  True, False,  True,  True]
    ])
    used_words = np.array([Word("TEST"), Word("ESTA"), Word("STEM"), Word("ME"), Word("ET")], dtype=object)
    used_ori = np.array([0, 1, 0, 1, 0])
    used_i = np.array([0, 1, 0, 3, 3])
    used_j = np.array([0, 0, 2, 2, 3])

    def test_add(self, *mocks):
        from pyzzle import Puzzle
        puzzle = Puzzle(5, 5)
        puzzle.add(0, 0, 0, "TEST")
        puzzle.add(1, 1, 0, "ESTA")
        puzzle.add(0, 0, 2, "STEM")
        puzzle.add(1, 3, 2, "ME")
        puzzle.add(0, 3, 3, "ET")
        self.assertTrue(np.all(puzzle.cell == self.cell))
        self.assertTrue(np.all(puzzle.cover == self.cover))
        self.assertTrue(np.all(puzzle.enable == self.enable))
        self.assertTrue(np.all(puzzle.used_ori[:puzzle.nwords] == self.used_ori))
        self.assertTrue(np.all(puzzle.used_i[:puzzle.nwords] == self.used_i))
        self.assertTrue(np.all(puzzle.used_j[:puzzle.nwords] == self.used_j))
        self.assertTrue(np.all(puzzle.used_words[:puzzle.nwords] == self.used_words))

    def test_drop(self, *mocks):
        from pyzzle import Puzzle
        puzzle = Puzzle(5, 5)
        with mock.patch.object(puzzle, "dic"):
            with mock.patch.object(puzzle, "plc"):
                puzzle.nwords = 5
                puzzle.cell = self.cell
                puzzle.cover = self.cover
                puzzle.enable = self.enable
                puzzle.used_ori = self.used_ori
                puzzle.used_i = self.used_i
                puzzle.used_j = self.used_j
                puzzle.used_words = self.used_words

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
                    [True,  True,  True,  True,  True],
                    [True,  True,  True,  True, False],
                    [True,  True,  True, False,  True],
                    [True,  True,  True,  True,  True],
                    [True,  True, False,  True,  True]
                ]
                self.assertTrue(np.all(puzzle.cell == cell_answer))
                self.assertTrue(np.all(puzzle.cover == cover_answer))
                self.assertTrue(np.all(puzzle.enable == enable_answer))
                self.assertTrue(np.all(puzzle.used_ori[:puzzle.nwords] == [1, 0, 0]))
                self.assertTrue(np.all(puzzle.used_i[:puzzle.nwords] == [1, 0, 3]))
                self.assertTrue(np.all(puzzle.used_j[:puzzle.nwords] == [0, 2, 3]))
                self.assertTrue(np.all(puzzle.used_words[:puzzle.nwords] == ["ESTA", "STEM", "ET"]))

if __name__ == '__main__':
    unittest.main()
