import unittest
from pathlib import PurePath

import numpy as np

from pyzzle import Puzzle, ObjectiveFunction


class TestObjectiveFunction(unittest.TestCase):
    """Test the ObjectiveFunction class."""
    cell = np.array([
        ['T', '', 'S', '', ''],
        ['E', 'S', 'T', 'A', ''],
        ['S', '', 'E', '', ''],
        ['T', '', 'M', 'E', ''],
        ['', '', '', 'T', '']
    ])
    def test_obj_func_type(self):
        puzzle = Puzzle.from_cell(self.cell)
        puzzle.obj_func = ObjectiveFunction(ObjectiveFunction.flist)
        of_scores = puzzle.obj_func.get_score(puzzle, all=True)
        for key, value in of_scores.items():
            print(key, value, type(value))
            self.assertTrue(type(value) in (int, float))