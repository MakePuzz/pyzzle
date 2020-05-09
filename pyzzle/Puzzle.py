import copy
import math
import pickle
import datetime
from enum import Enum

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pyzzle.Placeable import Placeable
from pyzzle.Dictionary import Dictionary
from pyzzle import utils

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


class Judgement(Enum):
    """
    Enumeration of the possible placement of the word to be placed on the board.
    """
    THE_WORD_CAN_BE_PLACED = 0
    THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED = 1
    AT_LEAST_ONE_PLACE_MUST_CROSS_OTHER_WORDS = 2
    NOT_A_CORRECT_INTERSECTION = 3
    THE_SAME_WORD_IS_IN_USE = 4
    THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION = 5
    US_USA_DOMINICA_DOMINICAN_PROBLEM = 6


class Puzzle:
    """
    The Almighty Puzzle Class.
    Example usage (not using dictionary)::
        from pyzzle import Puzzle
        puzzle = Puzzle(8,8)
        puzzle.add(0,0,0,'hoge')
        puzzle.add(1,0,0,'hotel')
        puzzle.add(1,0,2,'gateball')
        puzzle.add(0,3,0,'elevator')
        puzzle.show()
        puzzle.save_problem_image("problem.png")
        puzzle.save_answer_image("answer.png")
        puzzle.export_json("out.json")

    Example usage (using dictionary)::
        from pyzzle import Puzzle
        puzzle = Puzzle(8,8)
        dic = Dictionary("path_to_dict")
        puzzle.import_dict(dic)

        obj_func = ObjectiveFunction()
        obj_func.register(["total_weight", "sol_size", "cross_count", "fill_count", "max_connected_empties"])

        optimizer = Optimizer()
        optimizer.set_method("local_search")

        puzzle.compile(obj_func=obj_func, optimizer=optimizer)

        puzzle.first_solve()
        puzzle.solve(epoch=2)

        puzzle.save_problem_image("problem.png")
        puzzle.save_answer_image("answer.png")
        puzzle.export_json("out.json")
    """

    def __init__(self, width, height, title="Criss Cross"):
        """
        Initialize the puzzle object.
        
        Parameters
        ----------
        width : int
            Width of the puzzle.
        height : int
            Height of the puzzle.
        title : str, default "Criss Cross"
            Title of the puzzle.
        """
        self.width = width
        self.height = height
        self.total_weight = 0
        self.title = title
        self.cell = np.full(width * height, "", dtype="unicode").reshape(height, width)
        self.cover = np.zeros(width * height, dtype="int").reshape(height, width)
        self.label = np.zeros(width * height, dtype="int").reshape(height, width)
        self.enable = np.ones(width * height, dtype="bool").reshape(height, width)
        self.used_words = np.full(width * height, "", dtype=f"U{max(width, height)}")
        self.used_plc_idx = np.full(width * height, -1, dtype="int")
        self.sol_size = 0
        self.history = []
        self.base_history = []
        self.log = None
        self.epoch = 0
        self.nlabel = None
        self.first_solved = False
        self.init_seed = None
        self.dic = Dictionary()
        self.plc = Placeable(self.width, self.height, self.dic)
        self.obj_func = None
        self.optimizer = None

    def __str__(self):
        """
        Retrun the puzzle's title.
        """
        return self.title

    @property
    def is_unique_sol(self):
        """
        This method deter_mines whether it is the unique solution
        """
        rtn_bool = True
        # Get word1
        for s, p in enumerate(self.used_plc_idx[:self.sol_size]):
            i = self.plc.i[p]
            j = self.plc.j[p]
            word1 = self.used_words[s]
            if self.plc.ori[p] == 0:
                cross_idx1 = np.where(self.cover[i:i + len(word1), j] == 2)[0]
            elif self.plc.ori[p] == 1:
                cross_idx1 = np.where(self.cover[i, j:j + len(word1)] == 2)[0]
            # Get word2
            for t, q in enumerate(self.used_plc_idx[s + 1:self.sol_size]):
                i = self.plc.i[q]
                j = self.plc.j[q]
                word2 = self.used_words[s + t + 1]
                if len(word1) != len(word2):  # If word1 and word2 have different lengths, they can not be replaced
                    continue
                if self.plc.ori[q] == 0:
                    cross_idx2 = np.where(self.cover[i:i + len(word2), j] == 2)[0]
                if self.plc.ori[q] == 1:
                    cross_idx2 = np.where(self.cover[i, j:j + len(word2)] == 2)[0]
                replaceable = True
                # Check cross part from word1
                for w1idx in cross_idx1:
                    if word1[w1idx] != word2[w1idx]:
                        replaceable = False
                        break
                # Check cross part from word2
                if replaceable is True:
                    for w2idx in cross_idx2:
                        if word2[w2idx] != word1[w2idx]:
                            replaceable = False
                            break
                # If word1 and word2 are replaceable, this puzzle doesn't have a simple solution -> return False
                if replaceable is True:
                    print(f" - words '{word1}' and '{word2}' are replaceable")
                    rtn_bool = False
        return rtn_bool

    def reinit(self, all=False):
        """
        Reinitilize Puzzle information.

        Parameters
        ----------
        all : bool default False
            If True, Reinitialize the Dictionary, ObjectiveFunction, and Optimizer as well.
        """
        if all is True:
            self.dic = None
            self.plc = None
            self.obj_func = None
            self.optimizer = None
        self.total_weight = 0
        self.enable = np.ones(self.width * self.height, dtype="bool").reshape(self.height, self.width)
        self.cell = np.full(self.width * self.height, "", dtype="unicode").reshape(self.height, self.width)
        self.cover = np.zeros(self.width * self.height, dtype="int").reshape(self.height, self.width)
        self.label = np.zeros(self.width * self.height, dtype="int").reshape(self.height, self.width)
        self.enable = np.ones(self.width * self.height, dtype="bool").reshape(self.height, self.width)
        self.used_words = np.full(self.width * self.height, "", dtype=f"U{max(self.width, self.height)}")
        self.used_plc_idx = np.full(self.width * self.height, -1, dtype="int")
        self.sol_size = 0
        self.base_history = []
        self.history = []
        self.log = None
        self.epoch = 0
        self.first_solved = False
        self.init_seed = None

    def import_dict(self, dic):
        """
        Import the Dictionary, and generate the Placeable internally.

        Parameters
        ----------
        dic : Dictionary
            Dictionary object imported by Puzzle
        """
        self.dic = dic
        self.plc = Placeable(self.width, self.height, self.dic)

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
        """
        if ori == 0:
            empties = self.cell[i:i + w_len, j] == ""
        if ori == 1:
            empties = self.cell[i, j:j + w_len] == ""

        # If 0 words used, return True
        if self.sol_size is 0:
            return Judgement.THE_WORD_CAN_BE_PLACED

        # If the preceding and succeeding cells are already filled
        if ori == 0:
            if i > 0 and self.cell[i - 1, j] != "":
                return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED
            if i + w_len < self.height and self.cell[i + w_len, j] != "":
                return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED
        if ori == 1:
            if j > 0 and self.cell[i, j - 1] != "":
                return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED
            if j + w_len < self.width and self.cell[i, j + w_len] != "":
                return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED

        # At least one place must cross other words
        if np.all(empties == True):
            return Judgement.AT_LEAST_ONE_PLACE_MUST_CROSS_OTHER_WORDS

        # Judge whether correct intersection
        where = np.where(empties == False)[0]
        if ori == 0:
            j_all = np.full(where.size, j, dtype="int")
            if np.any(self.cell[where + i, j_all] != np.array(list(word))[where]):
                return Judgement.NOT_A_CORRECT_INTERSECTION
        if ori == 1:
            i_all = np.full(where.size, i, dtype="int")
            if np.any(self.cell[i_all, where + j] != np.array(list(word))[where]):
                return Judgement.NOT_A_CORRECT_INTERSECTION

        # If the same word is in use, return False
        if word in self.used_words:
            return Judgement.THE_SAME_WORD_IS_IN_USE

        # If neighbor cells are filled except at the intersection, return False
        where = np.where(empties == True)[0]
        if ori == 0:
            j_all = np.full(where.size, j, dtype="int")
            # Left side
            if j > 0 and np.any(self.cell[where + i, j_all - 1] != ""):
                return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION
            # Right side
            if j < self.width - 1 and np.any(self.cell[where + i, j_all + 1] != ""):
                return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION
        if ori == 1:
            i_all = np.full(where.size, i, dtype="int")
            # Upper
            if i > 0 and np.any(self.cell[i_all - 1, where + j] != ""):
                return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION
            # Lower
            if i < self.height - 1 and np.any(self.cell[i_all + 1, where + j] != ""):
                return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION

        # US/USA, DOMINICA/DOMINICAN problem
        if ori == 0:
            if np.any(self.enable[i:i + w_len, j] == False) or np.all(empties == False):
                return Judgement.US_USA_DOMINICA_DOMINICAN_PROBLEM
        if ori == 1:
            if np.any(self.enable[i, j:j + w_len] == False) or np.all(empties == False):
                return Judgement.US_USA_DOMINICA_DOMINICAN_PROBLEM

        # Returns True if no conditions are encountered.
        return Judgement.THE_WORD_CAN_BE_PLACED

    def _add(self, ori, i, j, k):
        """
        This internal method places a word at arbitrary positions.
        If it is impossible to place, do nothing.

        Parameters
        ----------
        ori : int
            Direction of the word (0:Vertical, 1:Horizontal)
        i : int
            Row number of the word
        j : int
            Column number of the word
        k : int
            The number of the word registered in Placeable
        """
        word = self.dic.word[k]
        weight = self.dic.weight[k]
        w_len = self.dic.w_len[k]

        # Judge whether adding is enabled
        code = self.is_placeable(ori, i, j, word, w_len)
        if code is not Judgement.THE_WORD_CAN_BE_PLACED:
            return code

        # Put the word to puzzle
        if ori == 0:
            self.cell[i:i + w_len, j] = list(word)[0:w_len]
        if ori == 1:
            self.cell[i, j:j + w_len] = list(word)[0:w_len]

        # Set the prohibited cell before and after placed word
        if ori == 0:
            if i > 0:
                self.enable[i - 1, j] = False
            if i + w_len < self.height:
                self.enable[i + w_len, j] = False
        if ori == 1:
            if j > 0:
                self.enable[i, j - 1] = False
            if j + w_len < self.width:
                self.enable[i, j + w_len] = False

        # Update cover array
        if ori == 0:
            self.cover[i:i + w_len, j] += 1
        if ori == 1:
            self.cover[i, j:j + w_len] += 1

        # Update properties
        word_idx = self.dic.word.index(word)
        self.used_plc_idx[self.sol_size] = self.plc.inv_p[ori, i, j, word_idx]
        self.used_words[self.sol_size] = self.dic.word[k]
        self.sol_size += 1
        self.total_weight += weight
        self.history.append((1, word_idx, ori, i, j))
        return code

    def add(self, ori, i, j, word, weight=0):
        """
        Places a word at arbitrary positions.
        If it is impossible to place, do nothing.

        Parameters
        ----------
        ori : int
            Direction of the word (0:Vertical, 1:Horizontal)
        i : int
            Row number of the word
        j : int
            Column number of the word
        word : str or int
            Word to be placed. Int refers to the word number in the Placeable.
        weight : int, default 0
            Word weight
        """
        if type(word) is int:
            k = word
        elif type(word) is str:
            self.dic.add(word, weight)
            self.plc._compute([word], self.dic.size - 1)
            k = self.dic.word.index(word)
        else:
            raise TypeError("word must be int or str.")
        self._add(ori, i, j, k)

    def add_to_limit(self):
        """
        Adds the words as much as possible.
        """
        # Make a random index of plc
        random_index = np.arange(self.plc.size)
        np.random.shuffle(random_index)

        # Add as much as possible
        sol_size_tmp = None
        while self.sol_size != sol_size_tmp:
            sol_size_tmp = self.sol_size
            drop_idx = []
            for i, r in enumerate(random_index):
                code = self._add(self.plc.ori[r], self.plc.i[r], self.plc.j[r], self.plc.k[r])
                if code is not Judgement.AT_LEAST_ONE_PLACE_MUST_CROSS_OTHER_WORDS:
                    drop_idx.append(i)
            random_index = np.delete(random_index, drop_idx)
        return

    def first_solve(self):
        """
        Create an initial solution.
        This method should always be called only once.
        """
        # Check the first_solved
        if self.first_solved:
            raise RuntimeError("'first_solve' method has already called")
        # Save initial seed number
        self.init_seed = np.random.get_state()[1][0]
        # Add as much as possible
        self.add_to_limit()
        self.first_solved = True

    def show(self):
        """
        Display the puzzle.
        """
        utils.show_2Darray(self.cell)

    def logging(self):
        """
        This method logs the current objective function values
        """
        if self.obj_func is None:
            raise RuntimeError("Logging method must be executed after compilation method")
        if self.log is None:
            self.log = pd.DataFrame(columns=self.obj_func.get_funcs())
            self.log.index.name = "epoch"
        tmp_series = pd.Series(self.obj_func.get_score(self, all=True), index=self.obj_func.get_funcs())
        self.log = self.log.append(tmp_series, ignore_index=True)

    def _drop(self, ori, i, j, k, is_kick=False):
        """
        This internal method removes the specified word from the puzzle.

        Parameters
        ----------
        ori : int
            Direction of the word (0:Vertical, 1:Horizontal)
        i : int
            Row number of the word
        j : int
            Column number of the word
        k : int
            The number of the word registered in Placeable
        is_kick : bool default False
            If this dropping is in the kick process, it should be True.
            This information is used in making ``history``.

        Notes
        -----
        This method pulls out the specified word without taking it
        into consideration, which may break the connectivity of the puzzle
        or cause LAOS / US / USA problems.
        """
        # Get p, pidx
        p = self.plc.inv_p[ori, i, j, k]
        pidx = np.where(self.used_plc_idx == p)[0][0]

        w_len = self.dic.w_len[k]
        weight = self.dic.weight[k]
        # Pull out a word
        if ori == 0:
            self.cover[i:i + w_len, j] -= 1
            where = np.where(self.cover[i:i + w_len, j] == 0)[0]
            j_all = np.full(where.size, j, dtype="int")
            self.cell[i + where, j_all] = ""
        if ori == 1:
            self.cover[i, j:j + w_len] -= 1
            where = np.where(self.cover[i, j:j + w_len] == 0)[0]
            i_all = np.full(where.size, i, dtype="int")
            self.cell[i_all, j + where] = ""
        # Update used_words, used_plc_idx, sol_size, total_weight
        self.used_words = np.delete(self.used_words, pidx)  # delete
        self.used_words = np.append(self.used_words, "")  # append
        self.used_plc_idx = np.delete(self.used_plc_idx, pidx)  # delete
        self.used_plc_idx = np.append(self.used_plc_idx, -1)  # append
        self.sol_size -= 1
        self.total_weight -= weight
        # Insert data to history
        code = 3 if is_kick else 2
        self.history.append((code, k, ori, i, j))
        # Release prohibited cells
        remove_flag = True
        if ori == 0:
            if i > 0:
                if i > 2 and np.all(self.cell[[i - 3, i - 2], [j, j]] != ""):
                    remove_flag = False
                if j > 2 and np.all(self.cell[[i - 1, i - 1], [j - 2, j - 1]] != ""):
                    remove_flag = False
                if j < self.width - 2 and np.all(self.cell[[i - 1, i - 1], [j + 1, j + 2]] != ""):
                    remove_flag = False
                if remove_flag == True:
                    self.enable[i - 1, j] = True
            if i + w_len < self.height:
                if i + w_len < self.height - 2 and np.all(self.cell[[i + w_len + 1, i + w_len + 2], [j, j]] != ""):
                    remove_flag = False
                if j > 2 and np.all(self.cell[[i + w_len, i + w_len], [j - 2, j - 1]] != ""):
                    remove_flag = False
                if j < self.width - 2 and np.all(self.cell[[i + w_len, i + w_len], [j + 1, j + 2]] != ""):
                    remove_flag = False
                if remove_flag == True:
                    self.enable[i + w_len, j] = True
        if ori == 1:
            if j > 0:
                if j > 2 and np.all(self.cell[[i, i], [j - 3, j - 2]] != ""):
                    remove_flag = False
                if i > 2 and np.all(self.cell[[i - 2, i - 1], [j - 1, j - 1]] != ""):
                    remove_flag = False
                if i < self.height - 2 and np.all(self.cell[[i + 1, i + 2], [j - 1, j - 1]] != ""):
                    remove_flag = False
                if remove_flag == True:
                    self.enable[i, j - 1] = True
            if j + w_len < self.width:
                if j + w_len < self.width - 2 and np.all(self.cell[[i, i], [j + w_len + 1, j + w_len + 2]] != ""):
                    remove_flag = False
                if i > 2 and np.all(self.cell[[i - 2, i - 1], [j + w_len, j + w_len]] != ""):
                    remove_flag = False
                if i < self.height - 2 and np.all(self.cell[[i + 1, i + 2], [j + w_len, j + w_len]] != ""):
                    remove_flag = False
                if remove_flag == True:
                    self.enable[i, j + w_len] = True

    def drop(self, word=None, ori_i_j=None):
        """
        Remove the specified word from the puzzle.

        Parameters
        ----------
        word : int or str
            The word number or word in the puzlle to drop.
        ori_i_j : tuple of int, optional
            Tuple indicating a specific word to drop.

        Notes
        -----
        This method pulls out the specified word without taking it
        into consideration, which may break the connectivity of the puzzle
        or cause LAOS / US / USA problems.
        """
        if word is None and ori_i_j is None:
            raise ValueError("'word' or 'ori_i_j' must be specified")
        if word is not None and ori_i_j is not None:
            raise ValueError("Both 'word' and 'ori_i_j' must not be specified at the same time.")
        if word is not None:
            if type(word) is int:
                k = word
            elif type(word) is str:
                k = self.dic.word.index(word)
            else:
                raise TypeError("'word' must be int or str")
            for p in self.used_plc_idx:
                if self.plc.k[p] == k:
                    ori, i, j = self.plc.ori[p], self.plc.i[p], self.plc.j[p]
                    break
        if ori_i_j is not None:
            if type(ori_i_j) not in (list, tuple):
                raise TypeError(f"ori_i_j must be list or tuple")
            if len(ori_i_j) is not 3:
                raise ValueError(f"Length of 'ori_i_j' must be 3, not {len(ori_i_j)}")
            for p in self.used_plc_idx:
                _ori, _i, _j = self.plc.ori[p], self.plc.i[p], self.plc.j[p]
                if _ori == ori_i_j[0] and _i == ori_i_j[1] and _j == ori_i_j[2]:
                    ori, i, j = _ori, _i, _j
                    k = self.plc.k[p]
                    break
        self._drop(ori, i, j, k)

    def collapse(self):
        """
        Pull out the puzzle words at random until the connectivity breaks down.
        """
        # If sol_size = 0, return
        if self.sol_size == 0:
            return

        # Make a random index of sol_size
        random_index = np.arange(self.sol_size)
        np.random.shuffle(random_index)

        # Drop words until connectivity collapses
        tmp_used_plc_idx = copy.deepcopy(self.used_plc_idx)
        for r, p in enumerate(tmp_used_plc_idx[random_index]):
            # Get ori, i, j, k, w_len
            ori = self.plc.ori[p]
            i = self.plc.i[p]
            j = self.plc.j[p]
            k = self.plc.k[p]
            w_len = self.dic.w_len[self.plc.k[p]]
            # If '2' is aligned in the cover array, the word can not be dropped
            if ori == 0:
                if not np.any(np.diff(np.where(self.cover[i:i + w_len, j] == 2)[0]) == 1):
                    self._drop(ori, i, j, k)
            if ori == 1:
                if not np.any(np.diff(np.where(self.cover[i, j:j + w_len] == 2)[0]) == 1):
                    self._drop(ori, i, j, k)

            self.label, self.nlabel = ndimage.label(self.cover)
            if self.nlabel >= 2:
                break
        return

    def export_json(self, name="out.json", indent=None):
        """
        Export puzzle answer as json.
        
        Parameters
        ----------
        name : str, default "out.json"
            Output file name
        indent : int, default None
            The indent in json output
        """
        import json
        word_list = []
        for p in self.used_plc_idx:
            word_list.append({"word":self.dic.word[self.plc.k[p]], "ori":self.plc.ori[p], "i":self.plc.i[p], "j":self.plc.j[p]})
            if p == -1:
                break
        try:
            mask = self.mask
        except:
            mask = np.full(self.cell.shape, True)
        with open(name, "w", encoding="utf-8") as f:
            json.dump({"list":word_list, "mask":mask.tolist()}, f, sort_keys=True, indent=indent, ensure_ascii=False)

    def kick(self):
        """
        Remove words other than the maximum CCL from the board
        """
        # If sol_size = 0, return
        if self.sol_size == 0:
            return
        mask = self.cover > 0
        self.label, self.nlabel = ndimage.label(mask)
        sizes = ndimage.sum(mask, self.label, range(self.nlabel + 1))
        largest_ccl = sizes.argmax()
        # Erase elements except CCL ('kick' in C-program)
        for idx, p in enumerate(self.used_plc_idx[:self.sol_size]):
            if p == -1:
                continue
            if self.label[self.plc.i[p], self.plc.j[p]] != largest_ccl:
                self._drop(self.plc.ori[p], self.plc.i[p], self.plc.j[p], self.plc.k[p], is_kick=True)

    def compile(self, obj_func, optimizer, debug=False):
        """
        Compile the objective function and optimization method into the Puzzle instance.

        Parameters
        ----------
        obj_func : ObjectiveFunction
            ObjectiveFunction object for compile to Puzzle
        optimizer : Optimizer
            Optimizer object for compile to Puzzle
        """
        self.obj_func = obj_func
        self.optimizer = optimizer

        if debug is True:
            print(">>> Compile succeeded.")
            print(" --- objective functions:")
            for func_num in range(len(obj_func)):
                print(f"  |-> {func_num} {obj_func.registered_funcs[func_num]}")
            print(f" --- optimizer: {optimizer.method}")

    def solve(self, epoch):
        """
        This method repeats the solution improvement by the specified number of epoch.

        Parameters
        ----------
        epoch : int
            The number of epoch
        """
        if self.first_solved is False:
            raise RuntimeError("'first_solve' method has not called")
        if epoch <= 0:
            raise ValueError("'epoch' must be lather than 0")
        exec(f"self.optimizer.{self.optimizer.method}(self, {epoch})")

    def show_log(self, title="Objective Function's epoch series", grid=True, figsize=None, **kwargs):
        """
        Show the epoch series for each objective function.

        Parameters
        ----------
        title : str default "Objective Function's epoch series"
            title of figure
        grid : bool default True
            grid on/off

        Returns
        -------
        ax : Axes
            Axes object plotted logs Pandas.DataFrame.plot

        See Also
        --------
        Pandas.DataFrame.plot
        """
        if self.log is None:
            raise RuntimeError("Puzzle has no log")
        return self.log.plot(subplots=True, title=title, grid=grid, figsize=figsize, **kwargs)

    def save_image(self, data, fpath, list_label="[Word List]", dpi=300):
        """
        Generate a puzzle image with word lists.
        
        Parameters
        ----------
        data : ndarray
            2D array for imaging
        fpath : str
            Output file path
        list_label : str, default "[Word List]" 
            Title label for word lists
        dpi : int, default 300
            Dot-per-inch
        """
        # Generate puzzle image
        colors = np.where(self.cover < 1, "#000000", "#FFFFFF")
        df = pd.DataFrame(data)
        fig = plt.figure(figsize=(16, 8), dpi=dpi)
        ax1 = fig.add_subplot(121)  # puzzle
        ax2 = fig.add_subplot(122)  # word list
        ax1.axis("off")
        ax2.axis("off")
        fig.set_facecolor('#EEEEEE')
        # Draw puzzle
        ax1_table = ax1.table(cellText=df.values, cellColumnours=colors, cellLoc="center", bbox=[0, 0, 1, 1])
        ax1_table.auto_set_font_size(False)
        ax1_table.set_fontsize(18)
        ax1.set_title(label="*** " + self.title + " ***", size=20)
        # Draw word list
        words = [word for word in self.used_words if word != ""]
        if words == []:
            words = [""]
        words.sort()
        words = sorted(words, key=len)
        rows = self.height
        cols = math.ceil(len(words) / rows)
        padnum = cols * rows - len(words)
        words += [''] * padnum
        words = np.array(words).reshape(cols, rows).T
        ax2_table = ax2.table(cellText=words, cellColumnours=None, cellLoc="left", edges="open", bbox=[0, 0, 1, 1])
        ax2.set_title(label=list_label, size=20)
        ax2_table.auto_set_font_size(False)
        ax2_table.set_fontsize(18)
        plt.tight_layout()
        plt.savefig(fpath, dpi=dpi)
        plt.close()

    def save_problem_image(self, fpath="problem.png", list_label="[Word List]", dpi=300):
        """
        Generate a puzzle problem image with word lists.
        
        Parameters
        ----------
        fpath : str, default "problem.png"
            Output file path
        list_label : str, default "[Word List]" 
            Title label for word lists
        dpi : int, default 300
            Dot-per-inch
        """
        data = np.full(self.width * self.height, "", dtype="unicode").reshape(self.height, self.width)
        self.save_image(data, fpath, list_label, dpi)

    def save_answer_image(self, fpath="answer.png", list_label="[Word List]", dpi=300):
        """
        Generate a puzzle answer image with word lists.
        
        Parameters
        ----------
        fpath : str, default "problem.png"
            Output file path
        list_label : str, default "[Word List]" 
            Title label for word lists
        dpi : int, default 300
            Dot-per-inch
        """
        data = self.cell
        self.save_image(data, fpath, list_label, dpi)

    def jump(self, idx):
        """
        
        """
        tmp_puzzle = self.__class__(self.width, self.height, self.title)
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
            elif code == 2:
                tmp_puzzle._drop(ori, i, j, k, is_kick=False)
            elif code == 3:
                tmp_puzzle._drop(ori, i, j, k, is_kick=True)
        tmp_puzzle.first_solved = True
        return tmp_puzzle

    def get_prev(self, n=1):
        if len(self.history) - n < 0:
            return self.jump(0)
        return self.jump(len(self.history) - n)

    def get_next(self, n=1):
        if len(self.history) + n > len(self.base_history):
            return self.get_latest()
        return self.jump(len(self.history) + n)

    def get_latest(self):
        return self.jump(len(self.base_history))

    def to_pickle(self, name=None):
        """
        This method saves Puzzle object as a binary file
        """
        now = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
        name = name or f"{now}_{self.dic.name}_{self.width}_{self.height}_{self.init_seed}_{self.epoch}.pickle"
        with open(name, mode="wb") as f:
            pickle.dump(self, f)

    def get_rect(self):
        rows = np.any(self.cover, axis=1)
        cols = np.any(self.cover, axis=0)
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]
        return r_min, r_max, c_min, c_max

    def move(self, direction, n=0, limit=False):
        r_min, r_max, c_min, c_max = self.get_rect()
        str2int = {'U': 1, 'D': 2, 'R': 3, 'L': 4}
        if direction in ('U', 'D', 'R', 'L', 'u', 'd', 'r', 'l'):
            direction = str2int[direction.upper()]
        if direction not in (1, 2, 3, 4):
            raise ValueError()
        if n < 0:
            reverse = {'1': 2, '2': 1, '3': 4, '4': 3}
            direction = reverse[str(direction)]
            n = -n
        if limit is True:
            n2limit = {1: r_min, 2: self.height - (r_max + 1), 3: c_min, 4: self.width - (c_max + 1)}
            n = n2limit[direction]

        if direction is 1:
            if r_min < n:
                raise RuntimeError()
            self.cell = np.roll(self.cell, -n, axis=0)
            self.cover = np.roll(self.cover, -n, axis=0)
            self.label = np.roll(self.label, -n, axis=0)
            self.enable = np.roll(self.enable, -n, axis=0)
            for i, p in enumerate(self.used_plc_idx[:self.sol_size]):
                self.used_plc_idx[i] = self.plc.inv_p[self.plc.ori[p], self.plc.i[p] - n, self.plc.j[p], self.plc.k[p]]
        if direction is 2:
            if self.height - (r_max + 1) < n:
                raise RuntimeError()
            self.cell = np.roll(self.cell, n, axis=0)
            self.cover = np.roll(self.cover, n, axis=0)
            self.label = np.roll(self.label, n, axis=0)
            self.enable = np.roll(self.enable, n, axis=0)
            for i, p in enumerate(self.used_plc_idx[:self.sol_size]):
                self.used_plc_idx[i] = self.plc.inv_p[self.plc.ori[p], self.plc.i[p] + n, self.plc.j[p], self.plc.k[p]]
        if direction is 3:
            if c_min < n:
                raise RuntimeError()
            self.cell = np.roll(self.cell, -n, axis=1)
            self.cover = np.roll(self.cover, -n, axis=1)
            self.label = np.roll(self.label, -n, axis=1)
            self.enable = np.roll(self.enable, -n, axis=1)
            for i, p in enumerate(self.used_plc_idx[:self.sol_size]):
                self.used_plc_idx[i] = self.plc.inv_p[self.plc.ori[p], self.plc.i[p], self.plc.j[p] - n, self.plc.k[p]]
        if direction is 4:
            if self.width - (c_max + 1) < n:
                raise RuntimeError()
            self.cell = np.roll(self.cell, n, axis=1)
            self.cover = np.roll(self.cover, n, axis=1)
            self.label = np.roll(self.label, n, axis=1)
            self.enable = np.roll(self.enable, n, axis=1)
            for i, p in enumerate(self.used_plc_idx[:self.sol_size]):
                self.used_plc_idx[i] = self.plc.inv_p[self.plc.ori[p], self.plc.i[p], self.plc.j[p] + n, self.plc.k[p]]

        self.history.append((4, direction, n))
