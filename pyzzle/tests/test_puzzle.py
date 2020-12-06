import unittest
from unittest import mock

import numpy as np

from pyzzle import Puzzle


class TestPuzzle(unittest.TestCase):
    """Test the Puzzle class."""
    def test_add(self, *mocks):
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
        with mock.patch.object(puzzle, "_dic"):
            with mock.patch.object(puzzle, "_plc"):
                puzzle.add(0, 0, 0, "TEST")
                puzzle.add(1, 1, 0, "ESTA")
                puzzle.add(0, 0, 2, "STEM")
                puzzle.add(1, 3, 2, "ME")
                puzzle.add(0, 3, 3, "ET")
                self.assertTrue(np.all(puzzle.cell == added_cell))
                self.assertTrue(np.all(puzzle.cover == added_cover))
                self.assertTrue(np.all(puzzle.enable == added_enable))
                self.assertTrue(np.all(puzzle.uori[:puzzle.nwords] == [0, 1, 0, 1, 0]))
                self.assertTrue(np.all(puzzle.ui[:puzzle.nwords] == [0, 1, 0, 3, 3]))
                self.assertTrue(np.all(puzzle.uj[:puzzle.nwords] == [0, 0, 2, 2, 3]))
                self.assertTrue(np.all(puzzle.uwords[:puzzle.nwords] == ["TEST", "ESTA", "STEM", "ME", "ET"]))

    def test_drop(self, *mocks):
        from pyzzle import Word
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
        puzzle.uori = np.array([0, 1, 0, 1, 0])
        puzzle.ui = np.array([0, 1, 0, 3, 3])
        puzzle.uj = np.array([0, 0, 2, 2, 3])
        puzzle.uwords = np.array([Word("TEST"), Word("ESTA"), Word("STEM"), Word("ME"), Word("ET")], dtype=object)

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
        self.assertTrue(np.all(puzzle.uori[:puzzle.nwords] == [1, 0, 0]))
        self.assertTrue(np.all(puzzle.ui[:puzzle.nwords] == [1, 0, 3]))
        self.assertTrue(np.all(puzzle.uj[:puzzle.nwords] == [0, 2, 3]))
        self.assertTrue(np.all(puzzle.uwords[:puzzle.nwords] == ["ESTA", "STEM", "ET"]))

    def test_move(self, *mocks):
        import copy
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
        puzzle.ui = np.array([0, 1, 0, 3, 3])
        puzzle.uj = np.array([0, 0, 2, 2, 3])

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
        self.assertTrue(np.all(puzzle.ui[:puzzle.nwords] == [0, 1, 0, 3, 3]))
        self.assertTrue(np.all(puzzle.uj[:puzzle.nwords] == [0, 0, 2, 2, 3]))
        puzzle.move("R", 1)
        self.assertTrue(np.all(puzzle.cell == moved_cell))
        self.assertTrue(np.all(puzzle.cover == moved_cover))
        self.assertTrue(np.all(puzzle.enable == moved_enable))
        self.assertTrue(np.all(puzzle.ui[:puzzle.nwords] == [0, 1, 0, 3, 3]))
        self.assertTrue(np.all(puzzle.uj[:puzzle.nwords] == [1, 1, 3, 3, 4]))
        puzzle.move("U", 1)
        self.assertTrue(np.all(puzzle.cell == moved_cell))
        self.assertTrue(np.all(puzzle.cover == moved_cover))
        self.assertTrue(np.all(puzzle.enable == moved_enable))
        self.assertTrue(np.all(puzzle.ui[:puzzle.nwords] == [0, 1, 0, 3, 3]))
        self.assertTrue(np.all(puzzle.uj[:puzzle.nwords] == [1, 1, 3, 3, 4]))
        puzzle.move("D", 1)
        self.assertTrue(np.all(puzzle.cell == moved_cell))
        self.assertTrue(np.all(puzzle.cover == moved_cover))
        self.assertTrue(np.all(puzzle.enable == moved_enable))
        self.assertTrue(np.all(puzzle.ui[:puzzle.nwords] == [0, 1, 0, 3, 3]))
        self.assertTrue(np.all(puzzle.uj[:puzzle.nwords] == [1, 1, 3, 3, 4]))
    
    def test_get_cover(self, *mocks):
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
        self.assertTrue(np.all(Puzzle.get_cover(cell) == cover))
        
    def test_get_enable(self, *mocks):
        cell = np.array([
            ['T', '', 'S', '', ''],
            ['E', 'S', 'T', 'A', ''],
            ['S', '', 'E', '', ''],
            ['T', '', 'M', 'E', ''],
            ['', '', '', 'T', '']
        ])
        enable = np.array([
            [True,  True,  True,  True,  True],
            [True,  True,  True,  True, False],
            [True,  True,  True, False,  True],
            [True, False,  True,  True, False],
            [False,  True, False,  True,  True]
        ])
        self.assertTrue(np.all(Puzzle.get_enable(cell) == enable))

    def test_get_uwords(self, *mocks):
        from pyzzle import Word
        cell = np.array([
            ['T', '', 'S', '', ''],
            ['E', 'S', 'T', 'A', ''],
            ['S', '', 'E', '', ''],
            ['T', '', 'M', 'E', ''],
            ['', '', '', 'T', '']
        ])
        uwords = np.array([Word("TEST"), Word("ESTA"), Word("STEM"), Word("ME"), Word("ET")], dtype=object)
        self.assertTrue(np.all(sorted(Puzzle.get_uwords(cell)) == sorted(uwords)))

    def test_is_unique(self, *mocks):
        cell = np.array([
            ['T', '', 'S', '', ''],
            ['E', 'S', 'T', 'A', ''],
            ['S', '', 'E', '', ''],
            ['T', '', 'M', 'E', ''],
            ['', '', '', 'T', '']
        ])
        puzzle = Puzzle.from_cell(cell)
        self.assertTrue(puzzle.is_unique)
        cell = np.array([
            ['T', '', 'S', '', ''],
            ['E', 'S', 'T', 'A', ''],
            ['S', '', 'E', '', ''],
            ['T', '', 'M', 'E', ''],
            ['E', 'A', '', 'T', '']
        ])
        puzzle = Puzzle.from_cell(cell)
        self.assertFalse(puzzle.is_unique)
    
    def test_rect(self, *mocks):
        cell = np.array([
            ['', '', 'S', '', ''],
            ['', 'S', 'T', 'A', ''],
            ['', '', 'E', '', ''],
            ['', '', 'M', '', ''],
            ['', '', '', '', '']
        ])
        puzzle = Puzzle.from_cell(cell)
        self.assertEqual(puzzle.rect.tolist(), cell[:4, 1:4].tolist())            

if __name__ == '__main__':
    unittest.main()
