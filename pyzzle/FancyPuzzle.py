import copy
import sys

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from pyzzle.Puzzle import Puzzle
from pyzzle.Placeable import Placeable
from pyzzle.Judgement import Judgement
from pyzzle import utils

class FancyPuzzle(Puzzle):
    def __init__(self, mask, name="Criss Cross"):
        self.mask = mask
        height = mask.shape[0]
        width = mask.shape[1]
        super().__init__(width, height, name)

    @property
    def circulation(self):
        """
        Circulation means that when there is a hole in the puzzle, 
        the words on the board surround it and are connected unbroken.
        This method returns the number of the circulation.
        So, if the puzzle has more than one hole, 
        the circulation will match the number of holes at most.

        See Also
        --------
        is_perfect_circulation
        """
        empties = np.zeros([self.width+2, self.height+2], dtype="int")
        empties[1:-1, 1:-1] = self.cover
        label, nlabel = ndimage.label(empties == False, structure=ndimage.generate_binary_structure(2,2))
        if nlabel <= 2:
            return 0
        circulation = 0
        for ilabel in range(2, nlabel+1):
            if np.any(self.mask[label[1:-1,1:-1]==ilabel] == False):
                # If an island with cover==0 is on the mask==False, then it represents a circulation.
                circulation += 1
            return circulation
    
    @property
    def is_perfect_circulation(self):
        """
        If the number of holes in the puzzle is the same as the circulation, it returns True.

        See Also
        --------
        circulation
        """
        mask = np.zeros([self.width+2, self.height+2], dtype=bool)
        mask[1:-1, 1:-1] = self.mask
        _, nlabel = ndimage.label(mask == False)
        return nlabel-1 == self.circulation

    def is_placeable(self, ori, i, j, word, w_len):
        """
        Returns the word placeability.

        Parameters
        ----------
        ori : int
            Direction of the word (0:Vertical, 1:Horizontal)
        i : int
            Row number of the word
        j : int
            Column number of the word
        word : str
            The word to be checked whether it can be added
        w_len : int
            length of the word

        Returns
        -------
        result : int
            Number of the judgment result

        Notes
        -----
        The result number corresponds to the judgment result
        0. The word can be placed (only succeeded)
        1. The preceding and succeeding cells are already filled
        2. At least one place must cross other words
        3. Not a correct intersection
        4. The same word is in use
        5. The Neighbor cells are filled except at the intersection
        6. US/USA, DOMINICA/DOMINICAN problem
        7. The word overlap with the mask
        """
        if ori == 0:
            if np.any(self.mask[i:i+w_len, j] == False):
                return Judgement.THE_WORD_OVERLAP_WITH_THE_MASK
        if ori == 1:
            if np.any(self.mask[i, j:j+w_len] == False):
                return Judgement.THE_WORD_OVERLAP_WITH_THE_MASK
    
        return super().is_placeable(ori, i, j, word, w_len)

    def show(self):
        utils.show_2Darray(self.cell, self.mask)
    
    def jump(self, idx):
        """
        Jump to the specified log state.

        Parameters
        ----------
        idx : int
            Index of log

        Returns
        -------
        jumped_puzzle : Puzzle
            Jumped Puzzle
        """
        tmp_puzzle = self.__class__(self.mask, self.name)
        tmp_puzzle.dic = copy.deepcopy(self.dic)
        tmp_puzzle.plc = Placeable(self.width, self.height, tmp_puzzle.dic)
        tmp_puzzle.optimizer = copy.deepcopy(self.optimizer)
        tmp_puzzle.obj_func = copy.deepcopy(self.obj_func)
        tmp_puzzle.base_history = copy.deepcopy(self.base_history)
        
        if set(self.history).issubset(self.base_history) is False:
            if idx <= len(self.history):
                tmp_puzzle.base_history = copy.deepcopy(self.history)
            else:
                raise RuntimeError('This puzzle is up to date')

        for code, k, ori, i, j in tmp_puzzle.base_history[:idx]:
            if code == 1:
                tmp_puzzle._add(ori, i, j, k)
            elif code in (2,3):
                tmp_puzzle._drop(ori, i, j, k)
        tmp_puzzle.init_sol = True
        return tmp_puzzle

    def save_problem_image(self, fname, list_label="word list", dpi=300):
        """
        Generate a puzzle problem image with word lists.
        
        Parameters
        ----------
        fname : str, default "problem.png"
            File name for output
        list_label : str, default "[Word List]" 
            Title label for word lists
        dpi : int, default 300
            Dot-per-inch
        """
        empty_cell = np.full(self.cell.shape, "", dtype="unicode")
        word_list = self.used_words[self.used_words != ""]
        utils.save_image(fname, empty_cell, word_list, mask=self.mask, title=self.name, label=list_label, dpi=dpi)

    def save_answer_image(self, fname, list_label="word list", dpi=300):
        """
        Generate a puzzle answer image with word lists.
        
        Parameters
        ----------
        fname : str, default "problem.png"
            File name for output
        list_label : str, default "[Word List]" 
            Title label for word lists
        dpi : int, default 300
            Dot-per-inch
        """
        word_list = self.used_words[self.used_words != ""]
        utils.save_image(fname, self.cell, word_list, mask=self.mask, title=self.name, label=list_label, dpi=dpi)

    def move(self, direction, n=0, limit=False):
        """
        Move the word-enclosing-region in the specified direction for the specified number of times.

        Parameters
        ----------
        direction : int or str
            The direction in which to move the word group.
            The correspondence between each int or str and the direction is as follows:
                1 or "U" : upward
                2 or "D" : downward
                3 or "R" : right
                4 or "L" : left
        n : int
            Number of times to move
        limit : bool, default False
            If True, move as much as possible in the specified direction.
        """
        super().move(direction, n, limit)