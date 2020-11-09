import numpy as np
from scipy import ndimage


class ObjectiveFunction:
    flist = [
        "weight",
        "nwords",
        "cross_count",
        "cross_rate",
        "fill_count",
        "max_connected_empties",
        "difficulty",
        "ease",
        "circulation",
        "gravity",
        "uniqueness",
        "weight_r",
        "nwords_r",
        "cross_count_r",
        "cross_rate_r",
        "fill_count_r",
        "max_connected_empties_r",
        "difficulty_r",
        "ease_r",
        "circulation_r",
        "gravity_r",
        "uniqueness_r",
    ]

    def __init__(self, objective_function=["nwords"]):
        if not isinstance(objective_function, (list, tuple, set)):
            raise TypeError("'objective_function' must be list or tuple or set")
        self.register(objective_function)

    def __len__(self):
        return len(self.registered_funcs)

    def get_funcs(self):
        return self.registered_funcs

    @classmethod
    def nwords(self, puzzle):
        """This method returns the number of words used in the solution."""
        return puzzle.nwords

    @classmethod
    def cross_count(self, puzzle):
        """This method returns the number of crosses of a word."""
        return np.sum(puzzle.cover == 2)

    @classmethod
    def cross_rate(self, puzzle):
        """This method returns the rate of crosses of a word."""
        return ObjectiveFunction.cross_count(puzzle)/ObjectiveFunction.nwords(puzzle)

    @classmethod
    def fill_count(self, puzzle):
        """This method returns the number of character cells in the puzzle."""
        return np.sum(puzzle.cover >= 1)

    @classmethod
    def weight(self, puzzle):
        """This method returns the sum of the word weights used for the solution."""
        return puzzle.weight

    @classmethod
    def max_connected_empties(self, puzzle):
        """This method returns the maximum number of concatenations for unfilled squares."""
        reverse_cover = puzzle.cover < 1
        zero_label, n_label = ndimage.label(reverse_cover)
        mask = zero_label > 0
        sizes = ndimage.sum(mask, zero_label, range(n_label+1))
        score = puzzle.width*puzzle.height - sizes.max()
        return score

    @classmethod
    def difficulty(self, puzzle):
        return puzzle.difficulty

    @classmethod
    def ease(self, puzzle):
        return 1 - puzzle.difficulty

    @classmethod
    def circulation(self, puzzle):
        return puzzle.circulation

    @classmethod
    def gravity(self, puzzle):
        return puzzle.gravity[puzzle.cover != 0].sum()

    @classmethod
    def uniqueness(self, puzzle):
        return int(puzzle.is_unique)

    @classmethod
    def nwords_r(self, puzzle):
        """This method returns the number of words used in the solution."""
        return -ObjectiveFunction.nwords(puzzle)

    @classmethod
    def cross_count_r(self, puzzle):
        """This method returns the number of crosses of a word."""
        return -ObjectiveFunction.cross_count(puzzle)

    @classmethod
    def cross_rate_r(self, puzzle):
        """This method returns the rate of crosses of a word."""
        return -ObjectiveFunction.cross_rate(puzzle)

    @classmethod
    def fill_count_r(self, puzzle):
        """This method returns the number of character cells in the puzzle."""
        return -ObjectiveFunction.fill_count(puzzle)

    @classmethod
    def weight_r(self, puzzle):
        """This method returns the sum of the word weights used for the solution."""
        return -ObjectiveFunction.weight(puzzle)

    @classmethod
    def max_connected_empties_r(self, puzzle):
        """This method returns the maximum number of concatenations for unfilled squares."""
        return -ObjectiveFunction.max_connected_empties(puzzle)

    @classmethod
    def difficulty_r(self, puzzle):
        return -ObjectiveFunction.difficulty(puzzle)

    @classmethod
    def ease_r(self, puzzle):
        return -ObjectiveFunction.ease(puzzle)

    @classmethod
    def circulation_r(self, puzzle):
        return -ObjectiveFunction.circulation(puzzle)

    @classmethod
    def gravity_r(self, puzzle):
        return -ObjectiveFunction.gravity(puzzle)
    
    @classmethod
    def uniqueness_r(self, puzzle):
        return -ObjectiveFunction.uniqueness(puzzle)

    def register(self, func_names):
        """
        This method registers an objective function in an instance
        """
        for func_name in func_names:
            if func_name not in ObjectiveFunction.flist:
                raise RuntimeError(f"ObjectiveFunction class does not have '{func_name}' function")
        self.registered_funcs = func_names

    def get_score(self, puzzle, i=0, func=None, all=False):
        """
        This method returns any objective function value
        """
        if all:
            scores = {}
            # scores = np.zeros(len(self.registered_funcs), dtype="float")
            for func_name in self.registered_funcs:
                scores[func_name] = eval(f"self.{func_name}(puzzle)")
            return scores
        if func is None:
            func = self.registered_funcs[i]
        return eval(f"self.{func}(puzzle)")
