import numpy as np
from scipy import ndimage


class ObjectiveFunction:
    def __init__(self):
        self.flist = [
            "total_weight",
            "sol_size",
            "cross_count",
            "fill_count",
            "max_connected_empties"
        ]
        self.registered_funcs = []

    def __len__(self):
        return len(self.registered_funcs)

    def get_funcs(self):
        return self.registered_funcs

    @staticmethod
    def sol_size(puzzle):
        """
        This method returns the number of words used in the solution
        """
        return puzzle.sol_size

    @staticmethod
    def cross_count(puzzle):
        """
        This method returns the number of crosses of a word
        """
        return np.sum(puzzle.cover == 2)

    @staticmethod
    def fill_count(puzzle):
        """
        This method returns the number of character cells in the puzzle
        """
        return np.sum(puzzle.cover >= 1)

    @staticmethod
    def total_weight(puzzle):
        """
        This method returns the sum of the word weights used for the solution
        """
        return puzzle.total_weight

    @staticmethod
    def max_connected_empties(puzzle):
        """
        This method returns the maximum number of concatenations for unfilled squares
        """
        reverse_cover = puzzle.cover < 1
        zero_label, n_label = ndimage.label(reverse_cover)
        mask = zero_label > 0
        sizes = ndimage.sum(mask, zero_label, range(n_label+1))
        score = puzzle.width*puzzle.height - sizes.max()
        return score

    def register(self, func_names, msg=True):
        """
        This method registers an objective function in an instance
        """
        for func_name in func_names:
            if func_name not in self.flist:
                raise RuntimeError(f"ObjectiveFunction class does not have '{func_name}' function")
            if msg is True:
                print(f" - '{func_name}' function has registered.")
        self.registered_funcs = func_names
        return

    def get_score(self, puzzle, i=0, func=None, all=False):
        """
        This method returns any objective function value
        """
        if all is True:
            scores = np.zeros(len(self.registered_funcs), dtype="int")
            for n in range(scores.size):
                scores[n] = eval(f"self.{self.registered_funcs[n]}(puzzle)")
            return scores
        if func is None:
            func = self.registered_funcs[i]
        return eval(f"self.{func}(puzzle)")
