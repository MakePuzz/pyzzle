import itertools
import numpy as np
from scipy import ndimage


class ObjectiveFunction:
    def __init__(self, msg=True):
        self.flist = [
            "totalWeight",
            "solSize",
            "crossCount",
            "fillCount",
            "maxConnectedEmpties"
        ]
        self.registeredFuncs = []
        if msg is True:
            print("ObjectiveFunction object has made.")

    def __len__(self):
        return len(self.registeredFuncs)

    def getFuncs(self):
        return self.registeredFuncs

    def solSize(self, puzzle):
        """
        This method returns the number of words used in the solution
        """
        return puzzle.solSize

    def crossCount(self, puzzle):
        """
        This method returns the number of crosses of a word
        """
        return np.sum(puzzle.cover == 2)

    def fillCount(self, puzzle):
        """
        This method returns the number of character cells in the puzzle
        """
        return np.sum(puzzle.cover >= 1)

    def totalWeight(self, puzzle):
        """
        This method returns the sum of the word weights used for the solution
        """
        return puzzle.totalWeight

    def maxConnectedEmpties(self, puzzle):
        """
        This method returns the maximum number of concatenations for unfilled squares
        """
        reverse_cover = puzzle.cover < 1
        zero_label, nlbl = ndimage.label(reverse_cover)
        mask = zero_label > 0
        sizes = ndimage.sum(mask, zero_label, range(nlbl+1))
        score = puzzle.width*puzzle.height - sizes.max()
        return score

    def register(self, funcNames, msg=True):
        """
        This method registers an objective function in an instance
        """
        for funcName in funcNames:
            if funcName not in self.flist:
                raise RuntimeError(f"ObjectiveFunction class does not have '{funcName}' function")
            if msg is True:
                print(f" - '{funcName}' function has registered.")
        self.registeredFuncs = funcNames
        return

    def getScore(self, puzzle, i=0, func=None, all=False):
        """
        This method returns any objective function value
        """
        if all is True:
            scores=np.zeros(len(self.registeredFuncs), dtype="int")
            for n in range(scores.size):
                scores[n] = eval(f"self.{self.registeredFuncs[n]}(puzzle)")
            return scores
        if func is None:
            func = self.registeredFuncs[i]
        return eval(f"self.{func}(puzzle)")
