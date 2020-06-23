import unittest
from unittest import mock

import numpy as np


class TestPuzzle(unittest.TestCase):
    """Test the Puzzle class."""
    def test_add(self, *mocks):
        from pyzzle import Puzzle
        puzzle = Puzzle(5, 5)
        added_cell = np.array([
            ['T', '', 'S', '', ''],
            ['E', 'S', 'T', 'A', ''],
            ['S', '', 'E', '', ''],
            ['T', '', 'M', 'E', ''],
            ['', '', '', 'T', '']
        ])
        added_cover = np.array([
            [1, 0, 1, 0, 0],
            [2, 1, 2, 1, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 2, 2, 0],
            [0, 0, 0, 1, 0]
        ])
        added_enable = np.array([
            [True,  True,  True,  True,  True],
            [True,  True,  True,  True, False],
            [True,  True,  True, False,  True],
            [True, False,  True,  True, False],
            [False,  True, False,  True,  True]
        ])
        with mock.patch.object(puzzle, "dic"):
            with mock.patch.object(puzzle, "plc"):
                puzzle.add(0, 0, 0, "TEST")
                puzzle.add(1, 1, 0, "ESTA")
                puzzle.add(0, 0, 2, "STEM")
                puzzle.add(1, 3, 2, "ME")
                puzzle.add(0, 3, 3, "ET")
                self.assertTrue(np.all(puzzle.cell == added_cell))
                self.assertTrue(np.all(puzzle.cover == added_cover))
                self.assertTrue(np.all(puzzle.enable == added_enable))
                self.assertTrue(np.all(puzzle.used_ori[:puzzle.nwords] == [0, 1, 0, 1, 0]))
                self.assertTrue(np.all(puzzle.used_i[:puzzle.nwords] == [0, 1, 0, 3, 3]))
                self.assertTrue(np.all(puzzle.used_j[:puzzle.nwords] == [0, 0, 2, 2, 3]))
                self.assertTrue(np.all(puzzle.used_words[:puzzle.nwords] == ["TEST", "ESTA", "STEM", "ME", "ET"]))

    def test_drop(self, *mocks):
        from pyzzle import Puzzle, Word
        puzzle = Puzzle(5, 5)
        puzzle.nwords = 5
        puzzle.cell = np.array([
            ['T', '', 'S', '', ''],
            ['E', 'S', 'T', 'A', ''],
            ['S', '', 'E', '', ''],
            ['T', '', 'M', 'E', ''],
            ['', '', '', 'T', '']
        ])
        puzzle.cover = np.array([
            [1, 0, 1, 0, 0],
            [2, 1, 2, 1, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 2, 2, 0],
            [0, 0, 0, 1, 0]
        ])
        puzzle.enable = np.array([
            [True,  True,  True,  True,  True],
            [True,  True,  True,  True, False],
            [True,  True,  True, False,  True],
            [True, False,  True,  True, False],
            [False,  True, False,  True,  True]
        ])
        puzzle.used_ori = np.array([0, 1, 0, 1, 0])
        puzzle.used_i = np.array([0, 1, 0, 3, 3])
        puzzle.used_j = np.array([0, 0, 2, 2, 3])
        puzzle.used_words = np.array([Word("TEST"), Word("ESTA"), Word("STEM"), Word("ME"), Word("ET")], dtype=object)

        puzzle.drop("TEST")
        puzzle.drop("ME")
        dropped_cell = np.array([
            ['', '', 'S', '', ''],
            ['E', 'S', 'T', 'A', ''],
            ['', '', 'E', '', ''],
            ['', '', 'M', 'E', ''],
            ['', '', '', 'T', '']
        ])
        dropped_cover = [
            [0, 0, 1, 0, 0],
            [1, 1, 2, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0]
        ]
        dropped_enable = [
            [True,  True,  True,  True,  True],
            [True,  True,  True,  True, False],
            [True,  True,  True, False,  True],
            [True,  True,  True,  True,  True],
            [True,  True, False,  True,  True]
        ]
        self.assertTrue(np.all(puzzle.cell == dropped_cell))
        self.assertTrue(np.all(puzzle.cover == dropped_cover))
        self.assertTrue(np.all(puzzle.enable == dropped_enable))
        self.assertTrue(np.all(puzzle.used_ori[:puzzle.nwords] == [1, 0, 0]))
        self.assertTrue(np.all(puzzle.used_i[:puzzle.nwords] == [1, 0, 3]))
        self.assertTrue(np.all(puzzle.used_j[:puzzle.nwords] == [0, 2, 3]))
        self.assertTrue(np.all(puzzle.used_words[:puzzle.nwords] == ["ESTA", "STEM", "ET"]))

    def test_move(self, *mocks):
        import copy
        from pyzzle import Puzzle
        puzzle = Puzzle(5, 5)
        puzzle.nwords = 5
        puzzle.cell = np.array([
            ['T', '', 'S', '', ''],
            ['E', 'S', 'T', 'A', ''],
            ['S', '', 'E', '', ''],
            ['T', '', 'M', 'E', ''],
            ['', '', '', 'T', '']
        ])
        puzzle.cover = np.array([
            [1, 0, 1, 0, 0],
            [2, 1, 2, 1, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 2, 2, 0],
            [0, 0, 0, 1, 0]
        ])
        puzzle.enable = np.array([
            [True,  True,  True,  True,  True],
            [True,  True,  True,  True, False],
            [True,  True,  True, False,  True],
            [True, False,  True,  True, False],
            [False,  True, False,  True,  True]
        ])
        puzzle.used_i = np.array([0, 1, 0, 3, 3])
        puzzle.used_j = np.array([0, 0, 2, 2, 3])

        moved_cell = np.array([
            ['', 'T', '', 'S', ''],
            ['', 'E', 'S', 'T', 'A'],
            ['', 'S', '', 'E', ''],
            ['', 'T', '', 'M', 'E'],
            ['', '', '', '', 'T']
        ])
        moved_cover = np.array([
            [0, 1, 0, 1, 0],
            [0, 2, 1, 2, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 2, 2],
            [0, 0, 0, 0, 1]
        ])
        moved_enable = np.array([
            [True,  True,  True,  True,  True],
            [False,  True,  True,  True, True],
            [True,  True,  True, True,  False],
            [True, True,  False,  True, True],
            [True,  False, True,  False,  True]
        ])

        init_cell = copy.deepcopy(puzzle.cell)
        init_cover = copy.deepcopy(puzzle.cover)
        init_enable = copy.deepcopy(puzzle.enable)
        puzzle.move("L", 1)
        self.assertTrue(np.all(puzzle.cell == init_cell))
        self.assertTrue(np.all(puzzle.cover == init_cover))
        self.assertTrue(np.all(puzzle.enable == init_enable))
        self.assertTrue(np.all(puzzle.used_i[:puzzle.nwords] == [0, 1, 0, 3, 3]))
        self.assertTrue(np.all(puzzle.used_j[:puzzle.nwords] == [0, 0, 2, 2, 3]))
        puzzle.move("R", 1)
        self.assertTrue(np.all(puzzle.cell == moved_cell))
        self.assertTrue(np.all(puzzle.cover == moved_cover))
        self.assertTrue(np.all(puzzle.enable == moved_enable))
        self.assertTrue(np.all(puzzle.used_i[:puzzle.nwords] == [0, 1, 0, 3, 3]))
        self.assertTrue(np.all(puzzle.used_j[:puzzle.nwords] == [1, 1, 3, 3, 4]))
        puzzle.move("U", 1)
        self.assertTrue(np.all(puzzle.cell == moved_cell))
        self.assertTrue(np.all(puzzle.cover == moved_cover))
        self.assertTrue(np.all(puzzle.enable == moved_enable))
        self.assertTrue(np.all(puzzle.used_i[:puzzle.nwords] == [0, 1, 0, 3, 3]))
        self.assertTrue(np.all(puzzle.used_j[:puzzle.nwords] == [1, 1, 3, 3, 4]))
        puzzle.move("D", 1)
        self.assertTrue(np.all(puzzle.cell == moved_cell))
        self.assertTrue(np.all(puzzle.cover == moved_cover))
        self.assertTrue(np.all(puzzle.enable == moved_enable))
        self.assertTrue(np.all(puzzle.used_i[:puzzle.nwords] == [0, 1, 0, 3, 3]))
        self.assertTrue(np.all(puzzle.used_j[:puzzle.nwords] == [1, 1, 3, 3, 4]))

if __name__ == '__main__':
    unittest.main()
