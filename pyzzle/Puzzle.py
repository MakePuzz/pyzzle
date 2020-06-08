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
from pyzzle.Optimizer import Optimizer
from pyzzle.ObjectiveFunction import ObjectiveFunction
from pyzzle.Judgement import Judgement
from pyzzle import utils

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meiryo',
                               'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


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

        puzzle.first_solve()

        obj_func = ["weight", "nwords", "cross_count", "fill_count", "max_connected_empties"]
        puzzle.solve(epoch=5, optimizer="local_search", objective_function=obj_func)

        puzzle.save_problem_image("problem.png")
        puzzle.save_answer_image("answer.png")
        puzzle.export_json("out.json")
    """

    def __init__(self, width=None, height=None, mask=None, name="Criss Cross"):
        """
        Initialize the puzzle object.

        Parameters
        ----------
        width : int
            Width of the puzzle.
        height : int
            Height of the puzzle.
        mask : array_like
            Mask of the puzzle.
        name : str, default "Criss Cross"
            Title of the puzzle.
        """
        self.width = width
        self.height = height
        self.mask = mask
        if self.mask is not None:
            self.mask = np.array(self.mask)
            self.width = self.mask.shape[1]
            self.height = self.mask.shape[0]
        self.weight = 0
        self.name = name
        self.cell = np.full(width * height, "",
                            dtype="unicode").reshape(height, width)
        self.cover = np.zeros(
            width * height, dtype="int").reshape(height, width)
        self.label = np.zeros(
            width * height, dtype="int").reshape(height, width)
        self.enable = np.ones(
            width * height, dtype="bool").reshape(height, width)
        self.used_words = np.full(
            width * height, "", dtype=f"U{max(width, height)}")
        self.used_plc_idx = np.full(width * height, -1, dtype="int")
        self.nwords = 0
        self.history = []
        self.base_history = []
        self.log = None
        self.epoch = 0
        self.nlabel = None
        self.first_solved = False
        self.seed = None
        self.dic = Dictionary()
        self.plc = Placeable(self.width, self.height, self.dic, self.mask)
        self.obj_func = None
        self.optimizer = None

    def __str__(self):
        """
        Retrun the puzzle's name.
        """
        return self.name

    @property
    def is_unique(self):
        """
        This method deter_mines whether it is the unique solution
        """
        rtn_bool = True
        # Get word1
        for s, p in enumerate(self.used_plc_idx[:self.nwords]):
            i = self.plc.i[p]
            j = self.plc.j[p]
            word1 = self.used_words[s]
            if self.plc.ori[p] == 0:
                cross_idx1 = np.where(self.cover[i:i + len(word1), j] == 2)[0]
            elif self.plc.ori[p] == 1:
                cross_idx1 = np.where(self.cover[i, j:j + len(word1)] == 2)[0]
            # Get word2
            for t, q in enumerate(self.used_plc_idx[s + 1:self.nwords]):
                i = self.plc.i[q]
                j = self.plc.j[q]
                word2 = self.used_words[s + t + 1]
                # If word1 and word2 have different lengths, they can not be replaced
                if len(word1) != len(word2):
                    continue
                if self.plc.ori[q] == 0:
                    cross_idx2 = np.where(
                        self.cover[i:i + len(word2), j] == 2)[0]
                if self.plc.ori[q] == 1:
                    cross_idx2 = np.where(
                        self.cover[i, j:j + len(word2)] == 2)[0]
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

    @property
    def difficulty(self):
        """
        Returns the difficulty of the puzzle.

        Returns
        -------
        difficulty : float
            Difficulty based on word lengths.
        """
        if self.nwords == 0:
            return 0
        # 単語の長さ別にリスト内の単語を分類
        # 各長さの単語数をカウント{長さ2:1個，長さ3:3個，長さ4:2個}
        w_len_count = {}
        for used_word in self.used_words:
            w_len = len(used_word)
            if w_len == 0:
                break
            if w_len not in w_len_count.keys():
                w_len_count[w_len] = 1
            else:
                w_len_count[w_len] += 1
        # カウント値の平均を全単語数で割った値は[1/単語数]に近いほど簡単で，[1]に近いほど難しい。
        w_len_count_mean = np.mean(list(w_len_count.values()))
        count_mean = w_len_count_mean/self.nwords
        # 0から1で難易度を表現するために正規化する
        difficulty = (count_mean - 1/self.nwords)/(1 - 1/self.nwords)
        return difficulty

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
        if self.mask is None:
            return 0
        empties = np.zeros([self.width+2, self.height+2], dtype="int")
        empties[1:-1, 1:-1] = self.cover
        label, nlabel = ndimage.label(
            empties == False, structure=ndimage.generate_binary_structure(2, 2))
        if nlabel <= 2:
            return 0
        circulation = 0
        for ilabel in range(2, nlabel+1):
            if np.any(self.mask[label[1:-1, 1:-1] == ilabel] == False):
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
        if self.mask is None:
            return False
        mask = np.zeros([self.width+2, self.height+2], dtype=bool)
        mask[1:-1, 1:-1] = self.mask
        _, nlabel = ndimage.label(mask == False)
        return nlabel-1 == self.circulation

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
        self.weight = 0
        self.enable = np.ones(self.width * self.height,
                              dtype="bool").reshape(self.height, self.width)
        self.cell = np.full(self.width * self.height, "",
                            dtype="unicode").reshape(self.height, self.width)
        self.cover = np.zeros(self.width * self.height,
                              dtype="int").reshape(self.height, self.width)
        self.label = np.zeros(self.width * self.height,
                              dtype="int").reshape(self.height, self.width)
        self.enable = np.ones(self.width * self.height,
                              dtype="bool").reshape(self.height, self.width)
        self.used_words = np.full(
            self.width * self.height, "", dtype=f"U{max(self.width, self.height)}")
        self.used_plc_idx = np.full(self.width * self.height, -1, dtype="int")
        self.nwords = 0
        self.base_history = []
        self.history = []
        self.log = None
        self.epoch = 0
        self.first_solved = False
        self.seed = None

    def import_dict(self, dic):
        """
        Import the Dictionary, and generate the Placeable internally.

        Parameters
        ----------
        dic : Dictionary
            Dictionary object imported by Puzzle
        """
        self.dic = dic
        self.plc = Placeable(self.width, self.height, self.dic, self.mask)

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

        # If 0 words used, return True
        if self.nwords == 0:
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
        if ori == 0:
            empties = self.cell[i:i + w_len, j] == ""
        if ori == 1:
            empties = self.cell[i, j:j + w_len] == ""
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
        self.used_plc_idx[self.nwords] = self.plc.inv_p[ori, i, j, word_idx]
        self.used_words[self.nwords] = self.dic.word[k]
        self.nwords += 1
        self.weight += weight
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
        return self._add(ori, i, j, k)

    def add_to_limit(self):
        """
        Adds the words as much as possible.
        """
        # Make a random index of plc
        random_index = np.arange(self.plc.size)
        np.random.shuffle(random_index)

        # Add as much as possible
        nwords_tmp = None
        while self.nwords != nwords_tmp:
            nwords_tmp = self.nwords
            drop_idx = []
            for i, r in enumerate(random_index):
                code = self._add(
                    self.plc.ori[r], self.plc.i[r], self.plc.j[r], self.plc.k[r])
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
        self.seed = np.random.get_state()[1][0]
        # Add as much as possible
        self.add_to_limit()
        self.first_solved = True

    def show(self):
        """
        Display the puzzle.
        """
        utils.show_2Darray(self.cell, self.mask)

    def logging(self):
        """
        This method logs the current objective function values
        """
        if self.obj_func is None:
            raise RuntimeError(
                "Logging method must be executed after compilation method")
        if self.log is None:
            self.log = pd.DataFrame(columns=self.obj_func.get_funcs())
            self.log.index.name = "epoch"
        tmp_series = pd.Series(self.obj_func.get_score(
            self, all=True), index=self.obj_func.get_funcs())
        self.log = self.log.append(tmp_series, ignore_index=True)

    def _drop(self, ori, i, j, k, is_kick=False):
        """
        Remove the specified word from the puzzle.

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
        # Get p, p_idx
        p = self.plc.inv_p[ori, i, j, k]
        p_idx = np.where(self.used_plc_idx == p)[0][0]

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
        # Update used_words, used_plc_idx, nwords, weight
        self.used_words = np.delete(self.used_words, p_idx)  # delete
        self.used_words = np.append(self.used_words, "")  # append
        self.used_plc_idx = np.delete(self.used_plc_idx, p_idx)  # delete
        self.used_plc_idx = np.append(self.used_plc_idx, -1)  # append
        self.nwords -= 1
        self.weight -= weight
        # Insert data to history
        code = 3 if is_kick else 2
        self.history.append((code, k, ori, i, j))
        # Update enable cells
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
        if word is ori_i_j is None:
            raise ValueError("'word' or 'ori_i_j' must be specified")
        if word is ori_i_j is not None:
            raise ValueError(
                "Both 'word' and 'ori_i_j' must not be specified at the same time.")
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
            if len(ori_i_j) != 3:
                raise ValueError(
                    f"Length of 'ori_i_j' must be 3, not {len(ori_i_j)}")
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
        # If nwords = 0, return
        if self.nwords == 0:
            return

        # Make a random index of nwords
        random_index = np.arange(self.nwords)
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
            if p == -1:
                break
            word_list.append(
                {"word": self.dic.word[self.plc.k[p]], "ori": self.plc.ori[p], "i": self.plc.i[p], "j": self.plc.j[p]})
        try:
            mask = self.mask
        except:
            mask = np.full(self.cell.shape, True)
        with open(name, "w", encoding="utf-8") as f:
            json.dump({"list": word_list, "mask": mask.tolist(), "name": self.name, "width": self.width, "height": self.height, "nwords": self.nwords,
                       "dict_name": self.dic.name, "seed": int(self.seed), "epoch": self.epoch}, f, sort_keys=True, indent=indent, ensure_ascii=False)

    def kick(self):
        """
        Remove words other than the maximum CCL from the board
        """
        # If nwords = 0, return
        if self.nwords == 0:
            return
        mask = self.cover > 0
        self.label, self.nlabel = ndimage.label(mask)
        sizes = ndimage.sum(mask, self.label, range(self.nlabel + 1))
        largest_ccl = sizes.argmax()
        # Erase elements except CCL ('kick' in C-program)
        for idx, p in enumerate(self.used_plc_idx[:self.nwords]):
            if p == -1:
                continue
            if self.label[self.plc.i[p], self.plc.j[p]] != largest_ccl:
                self._drop(self.plc.ori[p], self.plc.i[p],
                           self.plc.j[p], self.plc.k[p], is_kick=True)

    def solve(self, epoch, optimizer="local_search", objective_function=None, of=None):
        """
        This method repeats the solution improvement by the specified number of epoch.

        Parameters
        ----------
        epoch : int
            The number of epoch
        optimizer : str or Optimizer
            Optimizer
        objective_function or of : list or ObjectiveFunction
            ObjectiveFunction
        """
        if self.first_solved is False:
            raise RuntimeError("'first_solve' method shoukd be called earlier")
        if epoch <= 0:
            raise ValueError("'epoch' must be lather than 0")
        if isinstance(optimizer, str):
            self.optimizer = Optimizer(optimizer)
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        if objective_function is of is not None:
            raise ValueError(
                "'objective_function' and 'of' must not both be specified")
        if objective_function is None:
            objective_function = of
        if isinstance(objective_function, (list, tuple, set)):
            self.obj_func = ObjectiveFunction(objective_function)
        if isinstance(objective_function, ObjectiveFunction):
            self.obj_func = objective_function
        if self.optimizer.method == "local_search":
            exec(f"self.optimizer.{self.optimizer.method}(self, {epoch})")

    def show_log(self, name="Objective Function's epoch series", grid=True, figsize=None, **kwargs):
        """
        Show the epoch series for each objective function.

        Parameters
        ----------
        name : str default "Objective Function's epoch series"
            name of figure
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
        if figsize is None:
            figsize = [len(self.obj_func), len(self.obj_func)]
        return self.log.plot(subplots=True, title=name, grid=grid, figsize=figsize, **kwargs)

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
        utils.save_image(fname, empty_cell, word_list, mask=self.mask,
                         title=self.name, label=list_label, dpi=dpi)

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
        utils.save_image(fname, self.cell, word_list, mask=self.mask,
                         title=self.name, label=list_label, dpi=dpi)

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
        jumped_puzzle = self.__class__(
            self.width, self.height, slef.mask, self.name)
        jumped_puzzle.dic = copy.deepcopy(self.dic)
        jumped_puzzle.plc = Placeable(
            self.width, self.height, jumped_puzzle.dic, self.mask)
        jumped_puzzle.optimizer = copy.deepcopy(self.optimizer)
        jumped_puzzle.obj_func = copy.deepcopy(self.obj_func)
        jumped_puzzle.base_history = copy.deepcopy(self.base_history)

        if set(self.history).issubset(self.base_history) is False:
            if idx <= len(self.history):
                jumped_puzzle.base_history = copy.deepcopy(self.history)
            else:
                raise RuntimeError('This puzzle is up to date')

        for code, k, ori, i, j in jumped_puzzle.base_history[:idx]:
            if code == 1:
                jumped_puzzle._add(ori, i, j, k)
            elif code == 2:
                jumped_puzzle._drop(ori, i, j, k, is_kick=False)
            elif code == 3:
                jumped_puzzle._drop(ori, i, j, k, is_kick=True)
        jumped_puzzle.first_solved = True
        return jumped_puzzle

    def get_prev(self, n=1):
        """
        Returns to the previous log state for the specified number of times.

        Parameters
        ----------
        n : int, default 1
            The number of previous logs to go back to.

        Returns
        -------
        jumped_puzzle : Puzzle
            Previous Puzzle
        """
        if len(self.history) - n < 0:
            return self.jump(0)
        previous_puzzle = self.jump(len(self.history) - n)
        return previous_puzzle

    def get_next(self, n=1):
        """
        Returns to the next log state for the specified number of times.

        Parameters
        ----------
        n : int, default 1
            The number of logs to proceed after.

        Returns
        -------
        next_puzzle : Puzzle
            Next Puzzle
        """
        if len(self.history) + n > len(self.base_history):
            return self.get_latest()
        next_puzzle = self.jump(len(self.history) + n)
        return next_puzzle

    def get_latest(self):
        """
        Return a puzzle with the state of the latest log.

        Retruns
        -------
        latest_puzzle : Puzzle
            Latest puzzle
        """
        return self.jump(len(self.base_history))

    def to_pickle(self, name=None):
        """
        Save Puzzle object as Pickle
        """
        now = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
        name = name or f"{now}_{self.dic.name}_{self.width}_{self.height}_{self.seed}_{self.epoch}.pickle"
        with open(name, mode="wb") as f:
            pickle.dump(self, f)

    def get_rect(self):
        """
        Return a rectangular region that encloses a words

        Returns
        -------
        r_min : int
           Minimum number of rows
        r_max : int
           Maximum number of rows
        c_min : int
           Minimum number of cols
        c_min : int
           Maximum number of cols
        """
        rows = np.any(self.cover, axis=1)
        cols = np.any(self.cover, axis=0)
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]
        return r_min, r_max, c_min, c_max

    def move(self, direction, n=0, limit=False):
        """
        Move the word-enclosing-region in the specified direction for the specified number of times.

        Parameters
        ----------
        direction : int or str
            The direction in which to move the word group.
            The correspondence between each int or str and the direction == as follows:
                1 or "U" : upward
                2 or "D" : downward
                3 or "R" : right
                4 or "L" : left
        n : int
            Number of times to move
        limit : bool, default False
            If True, move as much as possible in the specified direction.
        """
        r_min, r_max, c_min, c_max = self.get_rect()
        str2int = {'U': 1, 'D': 2, 'R': 3, 'L': 4}
        if direction.upper() in ('U', 'D', 'R', 'L'):
            direction = str2int[direction.upper()]
        if direction not in (1, 2, 3, 4):
            raise ValueError()
        if n < 0:
            reverse = {'1': 2, '2': 1, '3': 4, '4': 3}
            direction = reverse[str(direction)]
            n = -n

        n2limit = {1: r_min, 2: self.height -
                   (r_max + 1), 3: c_min, 4: self.width - (c_max + 1)}

        if limit is True:
            n = n2limit[direction]

        if direction == 1:
            if r_min < n:
                n = n2limit[direction]
            num = -n
            axis = 0
            di = num
            dj = 0
        if direction == 2:
            if self.height - (r_max + 1) < n:
                n = n2limit[direction]
            num = n
            axis = 0
            di = num
            dj = 0
        if direction == 3:
            if c_min < n:
                n = n2limit[direction]
            num = -n
            axis = 1
            di = 0
            dj = num
        if direction == 4:
            if self.width - (c_max + 1) < n:
                n = n2limit[direction]
            num = n
            axis = 1
            di = 0
            dj = num
        self.cell = np.roll(self.cell, num, axis=axis)
        self.cover = np.roll(self.cover, num, axis=axis)
        self.label = np.roll(self.label, num, axis=axis)
        self.enable = np.roll(self.enable, num, axis=axis)
        for i, p in enumerate(self.used_plc_idx[:self.nwords]):
            self.used_plc_idx[i] = self.plc.inv_p[self.plc.ori[p],
                                                  self.plc.i[p] + di, self.plc.j[p] + dj, self.plc.k[p]]
        self.history.append((4, direction, n))

    def get_used_words_and_enable(self):
        """
        Returns
        -------
        used_words : list
        enable : np.ndarray
        """
        jj, ii = np.meshgrid(np.arange(self.width), np.arange(self.height))
        # 縦
        head0 = (self.cell[ii[0, :], jj[0, :]] != "") * \
            (self.cell[ii[0, :]+1, jj[0, :]] != "")
        body0 = (self.cell[ii[1:-1, :]-1, jj[1:-1, :]] == "") * (self.cell[ii[1:-1, :],
                                                                           jj[1:-1, :]] != "") * (self.cell[ii[1:-1, :]+1, jj[1:-1, :]] != "")
        start0 = np.vstack([head0, body0])

        # 横
        head1 = (self.cell[ii[:, 0], jj[:, 0]] != "") * \
            (self.cell[ii[:, 0], jj[:, 0]+1] != "")
        body1 = (self.cell[ii[:, 1:-1], jj[:, 1:-1]-1] == "") * (self.cell[ii[:, 1:-1],
                                                                           jj[:, 1:-1]] != "") * (self.cell[ii[:, 1:-1], jj[:, 1:-1]+1] != "")
        start1 = np.hstack([head1.reshape(self.height, 1), body1])

        indices = {"vertical": np.where(
            start0), "horizontal": np.where(start1)}

        used_words = []
        enable = np.ones(self.cell.shape).astype(bool)
        for i, j in zip(indices["vertical"][0], indices["vertical"][1]):
            # used_words
            try:
                imax = i + np.where(self.cell[i:, j] == '')[0][0]
            except:
                imax = self.height
            used_words.append(''.join(self.cell[i:imax, j]))
            # enable
            if i != 0:
                enable[i-1, j] = False
            if imax != self.height:
                enable[imax, j] = False

        for i, j in zip(indices["horizontal"][0], indices["horizontal"][1]):
            # used_words
            try:
                jmax = j + np.where(self.cell[i, j:] == '')[0][0]
            except:
                jmax = self.width
            used_words.append(''.join(self.cell[i, j:jmax]))
            # enable
            if j != 0:
                enable[i, j-1] = False
            if jmax != self.height:
                enable[i, jmax] = False
