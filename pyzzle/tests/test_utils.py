import unittest
from unittest import mock

import numpy as np

from pyzzle import utils


class TestPuzzle(unittest.TestCase):
    def test_ZeroSizePuzzleException(self, *mocks):
        from pyzzle.Exception import ZeroSizePuzzleException
        cover = np.full([5,5], 0)
        with self.assertRaises(ZeroSizePuzzleException):
            utils.get_rect(cover)