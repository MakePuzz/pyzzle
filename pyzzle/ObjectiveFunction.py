import numpy as np
from scipy import ndimage


class ObjectiveFunction:
    def __init__(self, objective_function=["nwords"]):
        self.flist = [
            "weight",
            "nwords",
            "cross_count",
            "fill_count",
            "max_connected_empties",
            "difficulty",
            "ease",
            "circulation",
            "gravity",
        ]
        if not isinstance(objective_function, (list, tuple, set)):
            raise TypeError("'nwords' must be list or tuple or set")
        self.register(objective_function)

    def __len__(self):
        return len(self.registered_funcs)

    def get_funcs(self):
        return self.registered_funcs

    @staticmethod
    def nwords(puzzle):
        """
        This method returns the number of words used in the solution.
        """
        return puzzle.nwords

    @staticmethod
    def cross_count(puzzle):
        """
        This method returns the number of crosses of a word.
        """
        return np.sum(puzzle.cover == 2)

    @staticmethod
    def fill_count(puzzle):
        """
        This method returns the number of character cells in the puzzle.
        """
        return np.sum(puzzle.cover >= 1)

    @staticmethod
    def weight(puzzle):
        """
        This method returns the sum of the word weights used for the solution.
        """
        return puzzle.weight

    @staticmethod
    def max_connected_empties(puzzle):
        """
        This method returns the maximum number of concatenations for unfilled squares.
        """
        reverse_cover = puzzle.cover < 1
        zero_label, n_label = ndimage.label(reverse_cover)
        mask = zero_label > 0
        sizes = ndimage.sum(mask, zero_label, range(n_label+1))
        score = puzzle.width*puzzle.height - sizes.max()
        return score

    @staticmethod
    def difficulty(puzzle):
        return puzzle.difficulty

    @staticmethod
    def ease(puzzle):
        return 1 - puzzle.difficulty

    @staticmethod
    def circulation(puzzle):
        return puzzle.circulation

    @staticmethod
    def gravity(puzzle):
        return puzzle.gravity[puzzle.cover != 0].sum()

    def register(self, func_names):
        """
        This method registers an objective function in an instance
        """
        for func_name in func_names:
            if func_name not in self.flist:
                raise RuntimeError(f"ObjectiveFunction class does not have '{func_name}' function")
        self.registered_funcs = func_names

    def get_score(self, puzzle, i=0, func=None, all=False):
        """
        This method returns any objective function value
        """
        if all is True:
            scores = np.zeros(len(self.registered_funcs), dtype="float")
            for n in range(scores.size):
                scores[n] = eval(f"self.{self.registered_funcs[n]}(puzzle)")
            return scores
        if func is None:
            func = self.registered_funcs[i]
        return eval(f"self.{func}(puzzle)")
