import unittest
from unittest import mock

import numpy as np


class TestFortranUtils(unittest.TestCase):
    """Test the Puzzle class."""
    def test_import_add_to_limit(self):
        from pyzzle.futils import add_to_limit
        assert 'add_to_limit' in dir()

if __name__ == '__main__':
    unittest.main()
