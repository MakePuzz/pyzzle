import copy
import json
import pickle
import datetime
import logging

import numpy as np
import pandas as pd
from scipy import ndimage
from matplotlib import rcParams

from pyzzle.Word import Word
from pyzzle.Placeable import Placeable
from pyzzle.Dictionary import Dictionary
from pyzzle.Optimizer import Optimizer
from pyzzle.ObjectiveFunction import ObjectiveFunction
from pyzzle.Judgement import Judgement
from pyzzle.History import History
from pyzzle import utils

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meiryo',
                               'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

LOG = logging.getLogger(__name__)
BLANK = ""


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

        obj_func = ["weight", "nwords", "cross_count", "fill_count", "max_connected_empties"]
        puzzle = puzzle.solve(epoch=5, optimizer="local_search", objective_function=obj_func)

        puzzle.save_problem_image("problem.png")
        puzzle.save_answer_image("answer.png")
        puzzle.export_json("out.json")
    """

    def __init__(self, width=None, height=None, mask=None, gravity=None, name="Criss Cross"):
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
        if mask is not None:
            self.mask = np.array(self.mask)
            self.width = self.mask.shape[1]
            self.height = self.mask.shape[0]
        if gravity is None:
            gravity = np.zeros([self.height, self.width])
        self.gravity = np.array(gravity)
        self.weight = 0
        self.name = name
        self.cell = np.full([self.height, self.width], BLANK, dtype="unicode")
        self.cover = np.zeros(self.cell.shape, dtype=np.int32)
        self.enable = np.ones(self.cell.shape, dtype="bool")
        self.used_ori = np.full(self.width * self.height, -1, dtype=np.int32)
        self.used_i = np.full(self.used_ori.size, -1, dtype=np.int32)
        self.used_j = np.full(self.used_ori.size, -1, dtype=np.int32)
        self.used_words = np.full(self.used_ori.size, BLANK, dtype=object)
        self.nwords = 0
        self.history = []
        self.base_history = []
        self.log = None
        self.epoch = 0
        self.seed = None
        self._dic = Dictionary()
        self._plc = Placeable(width=self.width, height=self.height)

    def __str__(self):
        """
        Return the puzzle's name.
        """
        return self.name

    def __lt__(self, other):
        if not isinstance(other, Puzzle):
            raise TypeError(f"'<' not supported between instances of 'Puzzle' and '{type(other)}'")
        if self.obj_func.registered_funcs != other.obj_func.registered_funcs:
            raise ValueError("Puzzles with different registered objective functions cannot be compared with each other")
        for func_num in range(len(self.obj_func)):
            self_score = self.obj_func.get_score(self, func_num)
            other_score = other.obj_func.get_score(other, func_num)
            if self_score < other_score:
                return True
        return False
    
    def __eq__(self, other):
        if not isinstance(other, Puzzle):
            raise TypeError(f"'==' not supported between instances of 'Puzzle' and '{type(other)}'")
        if self.obj_func.registered_funcs != other.obj_func.registered_funcs:
            raise ValueError("Puzzles with different registered objective functions cannot be compared with each other")
        for func_num in range(len(self.obj_func)):
            self_score = self.obj_func.get_score(self, func_num)
            other_score = other.obj_func.get_score(other, func_num)
            if self_score != other_score:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)
    
    @property
    def dic(self):
        return self._dic

    @property
    def is_unique(self):
        """
        This method deter_mines whether it is the unique solution
        """
        rtn_bool = True
        # Get word1
        for s, (ori1, i1, j1, word1) in enumerate(zip(self.used_ori[:self.nwords], self.used_i[:self.nwords], self.used_j[:self.nwords], self.used_words[:self.nwords])):
            if ori1 == 0:
                cross_idx1 = np.where(self.cover[i1:i1 + len(word1), j1] == 2)[0]
            elif ori1 == 1:
                cross_idx1 = np.where(self.cover[i1, j1:j1 + len(word1)] == 2)[0]
            # Get word2
            for ori2, i2, j2, word2 in zip(self.used_ori[s+1:self.nwords], self.used_i[s+1:self.nwords], self.used_j[s+1:self.nwords], self.used_words[s+1:self.nwords]):
                # If word1 and word2 have different lengths, they can not be replaced
                if len(word1) != len(word2):
                    continue
                if ori2 == 0:
                    cross_idx2 = np.where(
                        self.cover[i2:i2 + len(word2), j2] == 2)[0]
                if ori2 == 1:
                    cross_idx2 = np.where(
                        self.cover[i2, j2:j2 + len(word2)] == 2)[0]
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
                    LOG.info(f" - Words '{word1}' and '{word2}' are replaceable")
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
        empties = np.zeros([self.height+2, self.width+2], dtype="int")
        empties[1:-1, 1:-1] = self.cover
        label, nlabel = ndimage.label(empties == False, structure=ndimage.generate_binary_structure(2, 2))
        if nlabel <= 2:
            return 0
        circulation = 0
        for ilabel in range(2, nlabel+1):
            if np.any(self.mask[label[1:-1, 1:-1] == ilabel] == True):
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
        mask = np.zeros([self.height+2, self.width+2], dtype=bool)
        mask[1:-1, 1:-1] = self.mask
        mask_component = ndimage.label(mask == True)[1]
        return mask_component - 1 == self.circulation
    
    @property
    def component(self):
        return ndimage.label(self.cover)[1]

    def import_dict(self, dic):
        """
        Import the Dictionary, and generate the Placeable internally.

        Parameters
        ----------
        dic : Dictionary
            Dictionary object imported to Puzzle
        """
        self._dic += dic
        self._plc = Placeable(self.width, self.height, self._dic.word, self.mask)
        LOG.info(f"Dictionary imported")

    def replace_dict(self, dic):
        """
        Replace the imported Dictionary, and generate the Placeable internally.

        Parameters
        ----------
        dic : Dictionary
            Dictionary object replaced in Puzzle
        """
        self._dic = dic
        self._plc = Placeable(self.width, self.height, self._dic.word, self.mask)
        LOG.info(f"Dictionary replaced")

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
            if i > 0 and self.cell[i - 1, j] != BLANK:
                return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED
            if i + w_len < self.height and self.cell[i + w_len, j] != BLANK:
                return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED
        if ori == 1:
            if j > 0 and self.cell[i, j - 1] != BLANK:
                return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED
            if j + w_len < self.width and self.cell[i, j + w_len] != BLANK:
                return Judgement.THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED

        # At least one place must cross other words
        if ori == 0:
            empties = self.cell[i:i + w_len, j] == BLANK
        if ori == 1:
            empties = self.cell[i, j:j + w_len] == BLANK
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
            if j > 0 and np.any(self.cell[where + i, j_all - 1] != BLANK):
                return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION
            # Right side
            if j < self.width - 1 and np.any(self.cell[where + i, j_all + 1] != BLANK):
                return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION
        if ori == 1:
            i_all = np.full(where.size, i, dtype="int")
            # Upper
            if i > 0 and np.any(self.cell[i_all - 1, where + j] != BLANK):
                return Judgement.THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION
            # Lower
            if i < self.height - 1 and np.any(self.cell[i_all + 1, where + j] != BLANK):
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

    def _add(self, ori, i, j, word):
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
        word : Word or str
            The word registered in Placeable
        """
        w_len = len(word)

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
        self.used_ori[self.nwords] = ori
        self.used_i[self.nwords] = i
        self.used_j[self.nwords] = j
        self.used_words[self.nwords] = word
        self.nwords += 1
        self.weight += word.weight
        self.history.append((History.ADD, ori, i, j, word))
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
        if not isinstance(word, str):
            raise TypeError("word must be Word or str")
        self._dic.add(word, weight)
        self._plc._compute([Word(word, weight)], mask=self.mask, base_k=self._dic.size - 1)
        return self._add(ori, i, j, Word(word, weight))

    def add_to_limit(self):
        """
        Adds the words as much as possible.
        """
        # Make a random index of plc
        random = np.arange(self._plc.size)
        np.random.shuffle(random)

        # Add as much as possible
        nwords_tmp = None
        while self.nwords != nwords_tmp:
            nwords_tmp = self.nwords
            drop_idx = []
            for i, r in enumerate(random):
                code = self._add(self._plc.ori[r], self._plc.i[r], self._plc.j[r], self._plc.word[r])
                if code is not Judgement.AT_LEAST_ONE_PLACE_MUST_CROSS_OTHER_WORDS:
                    drop_idx.append(i)
            random = np.delete(random, drop_idx)
        return

    def add_to_limit_f(self, blank="*"):
        """
        Adds the words as much as possible.
        """
        try:
            from pyzzle.Puzzle.add_to_limit import add_to_limit as _add_to_limit
        except(ImportError) as err:
            LOG.debug(str(err))
            raise ImportError("Puzzle.add_to_limit is not installed.\
                            After installing GCC and GFortran, you need to reinstall pyzzle.")
        # Make a random index of plc
        not_used_words_idx = np.ones(len(self._plc), dtype=bool)
        plc_words = np.array(self._plc.word, dtype=object)
        for used_word in self.used_words[:self.nwords]:
            not_used_words_idx[plc_words == used_word] = False
        random = np.arange(self._plc.size - np.count_nonzero(~not_used_words_idx))
        np.random.shuffle(random)

        # Add as much as possible
        n = random.size
        w_len_max = max(self._dic.w_len)

        cell = np.where(self.cell == BLANK, blank, self.cell)
        cell = np.array(list(map(lambda x: ord(x), cell.ravel()))).reshape(cell.shape)
        cell = np.asfortranarray(cell.astype(np.int32))

        ori_s = np.array(self._plc.ori)[not_used_words_idx][random]
        i_s = np.array(self._plc.i)[not_used_words_idx][random] + 1
        j_s = np.array(self._plc.j)[not_used_words_idx][random] + 1
        k_s = np.array(self._plc.k)[not_used_words_idx][random]
        plc_words = plc_words[not_used_words_idx][random]
        w_lens = np.array(self._dic.w_len)[k_s]
        # convert str to int
        str2int = lambda plc_word: list(map(ord, plc_word))
        words_int = list(map(str2int, plc_words))
        # 0 padding
        padding = lambda x: x + [0] * (w_len_max - len(x))
        words_int = np.asfortranarray(np.array(list(map(padding, words_int)), dtype=np.int32))
        enable = np.asfortranarray(self.enable.astype(np.int32))
        used_idx = _add_to_limit(self.height, self.width, n, w_len_max, ord(blank),
                                ori_s, i_s, j_s, k_s, words_int, w_lens, cell, enable)
        for p in used_idx[used_idx != -1]-1:
            self._add(ori_s[p], i_s[p] - 1, j_s[p] - 1, Word(plc_words[p], plc_words[p].weight))
        return

    def show(self):
        """
        Display the puzzle.
        """
        utils.show_2Darray(self.cell, self.mask, blank=BLANK)

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
        tmp_series = pd.Series(self.obj_func.get_score(self, all=True), index=self.obj_func.get_funcs())
        self.log = self.log.append(tmp_series, ignore_index=True)

    def _drop(self, ori, i, j, word, is_kick=False):
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
        word : Word or str
            The word to be drop
        is_kick : bool default False
            If this dropping is in the kick process, it should be True.
            This information is used in making ``history``.

        Notes
        -----
        This method pulls out the specified word without taking it
        into consideration, which may break the connectivity of the puzzle
        or cause LAOS / US / USA problems.
        """
        # Get p_idx
        drop_idx = np.where(self.used_words == word)[0][0]
        w_len = len(word)

        # Pull out a word
        if ori == 0:
            self.cover[i:i + w_len, j] -= 1
            where = np.where(self.cover[i:i + w_len, j] == 0)[0]
            j_all = np.full(where.size, j, dtype="int")
            self.cell[i + where, j_all] = BLANK
        if ori == 1:
            self.cover[i, j:j + w_len] -= 1
            where = np.where(self.cover[i, j:j + w_len] == 0)[0]
            i_all = np.full(where.size, i, dtype="int")
            self.cell[i_all, j + where] = BLANK
        # Update        
        self.used_ori[drop_idx:-1] = self.used_ori[drop_idx+1:]
        self.used_ori[-1] = -1
        self.used_i[drop_idx:-1] = self.used_i[drop_idx+1:]
        self.used_i[-1] = -1
        self.used_j[drop_idx:-1] = self.used_j[drop_idx+1:]
        self.used_j[-1] = -1
        self.used_words[drop_idx:-1] = self.used_words[drop_idx+1:]
        self.used_words[-1] = BLANK
        self.nwords -= 1
        self.weight -= word.weight
        # Insert data to history
        code = History.DROP_KICK if is_kick else History.DROP
        self.history.append((code, ori, i, j, word))
        # Update enable cells
        remove_flag = True
        if ori == 0:
            if i > 0:
                if i > 2 and np.all(self.cell[[i - 3, i - 2], [j, j]] != BLANK):
                    remove_flag = False
                if j > 2 and np.all(self.cell[[i - 1, i - 1], [j - 2, j - 1]] != BLANK):
                    remove_flag = False
                if j < self.width - 2 and np.all(self.cell[[i - 1, i - 1], [j + 1, j + 2]] != BLANK):
                    remove_flag = False
                if remove_flag == True:
                    self.enable[i - 1, j] = True
            if i + w_len < self.height:
                if i + w_len < self.height - 2 and np.all(self.cell[[i + w_len + 1, i + w_len + 2], [j, j]] != BLANK):
                    remove_flag = False
                if j > 2 and np.all(self.cell[[i + w_len, i + w_len], [j - 2, j - 1]] != BLANK):
                    remove_flag = False
                if j < self.width - 2 and np.all(self.cell[[i + w_len, i + w_len], [j + 1, j + 2]] != BLANK):
                    remove_flag = False
                if remove_flag == True:
                    self.enable[i + w_len, j] = True
        if ori == 1:
            if j > 0:
                if j > 2 and np.all(self.cell[[i, i], [j - 3, j - 2]] != BLANK):
                    remove_flag = False
                if i > 2 and np.all(self.cell[[i - 2, i - 1], [j - 1, j - 1]] != BLANK):
                    remove_flag = False
                if i < self.height - 2 and np.all(self.cell[[i + 1, i + 2], [j - 1, j - 1]] != BLANK):
                    remove_flag = False
                if remove_flag == True:
                    self.enable[i, j - 1] = True
            if j + w_len < self.width:
                if j + w_len < self.width - 2 and np.all(self.cell[[i, i], [j + w_len + 1, j + w_len + 2]] != BLANK):
                    remove_flag = False
                if i > 2 and np.all(self.cell[[i - 2, i - 1], [j + w_len, j + w_len]] != BLANK):
                    remove_flag = False
                if i < self.height - 2 and np.all(self.cell[[i + 1, i + 2], [j + w_len, j + w_len]] != BLANK):
                    remove_flag = False
                if remove_flag == True:
                    self.enable[i, j + w_len] = True
        return

    def drop(self, word=None, ori_i_j=None):
        """
        Remove the specified word from the puzzle.

        Parameters
        ----------
        word : Word or str
            The word in the puzzle to drop.
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
        if word and ori_i_j:
            raise ValueError(
                "Both 'word' and 'ori_i_j' must not be specified at the same time.")
        if word:
            if not isinstance(word, str):
                raise TypeError("'word' must be Word or str")
            word = Word(word)
            drop_idx = np.where(self.used_words == word)[0][0]
            ori = self.used_ori[drop_idx]
            i = self.used_i[drop_idx]
            j = self.used_j[drop_idx]
        if ori_i_j is not None:
            if type(ori_i_j) not in (list, tuple):
                raise TypeError(f"ori_i_j must be list or tuple")
            if len(ori_i_j) != 3:
                raise ValueError(
                    f"Length of 'ori_i_j' must be 3, not {len(ori_i_j)}")
            ori, i, j = ori_i_j
            for _ori, _i, _j, _word in zip(self.used_ori, self.used_i, self.used_j, self.used_words):
                if _ori == ori and _i == i and _j == j:
                    word = _word
                    break
        self._drop(ori, i, j, word)

    def collapse(self):
        """
        Pull out the puzzle words at random until the connectivity breaks down.
        """
        # If nwords = 0, return
        if self.nwords == 0:
            return
        # Make a random index of nwords
        random = np.arange(self.nwords)
        np.random.shuffle(random)
        # Drop words until connectivity collapses
        used_ori_random = copy.deepcopy(self.used_ori[:self.nwords][random])
        used_i_random = copy.deepcopy(self.used_i[:self.nwords][random])
        used_j_random = copy.deepcopy(self.used_j[:self.nwords][random])
        used_word_random = copy.deepcopy(self.used_words[:self.nwords][random])
        w_lens = np.vectorize(len)(used_word_random)
        for ori, i, j, word, w_len in zip(used_ori_random, used_i_random, used_j_random, used_word_random, w_lens):
            # If '2' is aligned in the cover array, the word can not be dropped
            if ori == 0:
                if not np.any(np.diff(np.where(self.cover[i:i + w_len, j] == 2)[0]) == 1):
                    self._drop(ori, i, j, word)
            if ori == 1:
                if not np.any(np.diff(np.where(self.cover[i, j:j + w_len] == 2)[0]) == 1):
                    self._drop(ori, i, j, word)
            if self.component >= 2:
                break
        return
    
    def to_json(self, indent=None):
        """
        Export puzzle answer as json.

        Parameters
        ----------
        name : str, default "out.json"
            Output file name
        indent : int, default None
            The indent in json output

        Returns
        -------
        json_dict : dict
            The json dictionary
        """
        word_list = []
        for ori, i, j, word in zip(self.used_ori[:self.nwords], self.used_i[:self.nwords], self.used_j[:self.nwords], self.used_words[:self.nwords]):
            word_list.append({"i": int(i), "j": int(j), "ori": int(ori), "word": word})
        mask = self.mask
        if mask is None:
            mask = np.full(self.cell.shape, True)
        json_dict = {
            "list": word_list,
            "mask": mask.tolist(),
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "nwords": self.nwords,
            "seed": int(self.seed),
            "epoch": self.epoch,
        }
        return json_dict

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
        with open(name, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, sort_keys=True, indent=indent, ensure_ascii=False)

    @staticmethod
    def from_json(name):
        with open(name, "rb") as f:
            data = json.load(f)
        puzzle = Puzzle(width=data["width"], height=data["height"], mask=data["mask"], name=data["name"])
        puzzle.seed = data["seed"]
        word_list = data["list"]
        for word_dict in word_list:
            puzzle.add(word_dict["ori"], word_dict["i"], word_dict["j"], word_dict["word"])
        return puzzle

    def kick(self):
        """
        Remove words other than the largest component from puzzle
        """
        # If nwords = 0, return
        if self.nwords == 0:
            return
        mask = self.cover > 0
        label, nlabel = ndimage.label(mask)
        sizes = ndimage.sum(mask, label, range(nlabel + 1))
        largest_ccl = sizes.argmax()
        # Erase elements except largest component.
        # In self._drop used_x will shrink, so pass prev_used_x in reverse order.
        prev_used_ori = self.used_ori[:self.nwords][::-1]
        prev_used_i = self.used_i[:self.nwords][::-1]
        prev_used_j = self.used_j[:self.nwords][::-1]
        prev_used_words = self.used_words[:self.nwords][::-1]
        for ori, i, j, word in zip(prev_used_ori, prev_used_i, prev_used_j, prev_used_words):
            if label[i, j] != largest_ccl:
                self._drop(ori, i, j, word, is_kick=True)
        return

    def solve(self, epoch, optimizer="local_search", objective_function=None, of=None, n=None, show=True, use_f=False):
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
        # Save initial seed number
        if self.seed == None:
            self.seed = np.random.get_state()[1][0]
        if epoch <= 0:
            raise ValueError("'epoch' must be lather than 0")
        if isinstance(optimizer, str):
            self.optimizer = Optimizer(optimizer)
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        if objective_function is of is not None:
            raise ValueError("'objective_function' and 'of' must not both be specified")
        if objective_function is None:
            objective_function = of
        if isinstance(objective_function, (list, tuple, set)):
            self.obj_func = ObjectiveFunction(objective_function)
        if isinstance(objective_function, ObjectiveFunction):
            self.obj_func = objective_function
        if self.optimizer.method == "local_search":
            return self.optimizer.optimize(self, epoch, show=show, use_f=use_f)
        if self.optimizer.method == "multi_start":
            return self.optimizer.optimize(self, epoch, n=n, show=show, use_f=use_f)

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
            if len(self.obj_func) <= 5:
                figsize = [5, 5]
            else:
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
        empty_cell = np.full(self.cell.shape, " ", dtype="unicode")
        empty_cell[self.cell == BLANK] = BLANK
        word_list = self.used_words[self.used_words != BLANK]
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
        word_list = self.used_words[self.used_words != BLANK]
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
            self.width, self.height, self.mask, self.name)
        jumped_puzzle._dic = copy.deepcopy(self._dic)
        jumped_puzzle._plc = Placeable(self.width, self.height, jumped_puzzle._dic, self.mask)
        jumped_puzzle.optimizer = copy.deepcopy(self.optimizer)
        jumped_puzzle.obj_func = copy.deepcopy(self.obj_func)
        jumped_puzzle.base_history = copy.deepcopy(self.base_history)

        if set(self.history).issubset(self.base_history) is False:
            if idx <= len(self.history):
                jumped_puzzle.base_history = copy.deepcopy(self.history)
            else:
                raise RuntimeError('This puzzle is up to date')

        for hist in jumped_puzzle.base_history[:idx]:
            if hist[0] == History.ADD:
                jumped_puzzle._add(hist[1], hist[2], hist[3], hist[4])
            elif hist[0] == History.DROP:
                jumped_puzzle._drop(hist[1], hist[2], hist[3], hist[4], is_kick=False)
            elif hist[0] == History.DROP_KICK:
                jumped_puzzle._drop(hist[1], hist[2], hist[3], hist[4], is_kick=True)
            elif hist[0] == History.MOVE:
                jumped_puzzle.move(direction=hist[1], n=hist[2])
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

        Returns
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
        name = name or f"{now}_{self.width}_{self.height}_{self.seed}_{self.epoch}.pickle"
        with open(name, mode="wb") as f:
            pickle.dump(self, f)

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
        r_min, r_max, c_min, c_max = utils.get_rect(self.cover)
        str2int = {'U': 1, 'D': 2, 'R': 3, 'L': 4}
        if isinstance(direction, str) and direction.upper() in ('U', 'D', 'R', 'L'):
            direction = str2int[direction.upper()]
        if direction not in (1, 2, 3, 4):
            raise ValueError()
        if n < 0:
            reverse = {'1': 2, '2': 1, '3': 4, '4': 3}
            direction = reverse[str(direction)]
            n = -n

        n2limit = {1: r_min, 2: self.height - (r_max + 1), 3: self.width - (c_max + 1), 4: c_min}

        if limit is True:
            n = n2limit[direction]

        if direction == 1:
            if r_min < n:
                n = n2limit[direction]
            di_dj = (-1,0)
            axis = 0
        if direction == 2:
            if self.height - (r_max + 1) < n:
                n = n2limit[direction]
            di_dj = (1,0)
            axis = 0
        if direction == 3:
            if self.width - (c_max + 1) < n:
                n = n2limit[direction]
            di_dj = (0,1)
            axis = 1
        if direction == 4:
            if c_min < n:
                n = n2limit[direction]
            di_dj = (0,-1)
            axis = 1

        for _ in range(n):
            if self.mask is not None:
                if np.any(np.roll(self.cover, sum(di_dj), axis=axis)[self.mask == True] >= 1):
                    break
            self.cell = np.roll(self.cell, sum(di_dj), axis=axis)
            self.cover = np.roll(self.cover, sum(di_dj), axis=axis)
            self.used_i += di_dj[0]
            self.used_j += di_dj[1]
            self.history.append((History.MOVE, direction, 1, None, None))
        self.enable = self.get_enable()
        return

    def get_word_indices(self, cell=None):
        """
        Returns the indices of the head of the word on the board.

        Parameters
        ----------
        cell : numpy ndarray
            cell array

        Returns
        -------
        indices : dict
            Indices of the head of the words on the board in a dictionary with "vertical" and "horizontal" keys
        """
        if cell is None:
            cell = self.cell
        width = cell.shape[1]
        height = cell.shape[0]
        jj, ii = np.meshgrid(np.arange(width), np.arange(height))

        # Vertical
        head_0 = (cell[ii[0, :], jj[0, :]] != BLANK) * (cell[ii[0, :]+1, jj[0, :]] != BLANK)
        body_0 = (cell[ii[1:-1, :]-1, jj[1:-1, :]] == BLANK) * (cell[ii[1:-1, :], jj[1:-1, :]] != BLANK) * (cell[ii[1:-1, :]+1, jj[1:-1, :]] != BLANK)
        start_0 = np.vstack([head_0, body_0])

        # Horizontal
        head_1 = (cell[ii[:, 0], jj[:, 0]] != BLANK) * (cell[ii[:, 0], jj[:, 0]+1] != BLANK)
        body_1 = (cell[ii[:, 1:-1], jj[:, 1:-1]-1] == BLANK) * (cell[ii[:, 1:-1], jj[:, 1:-1]] != BLANK) * (cell[ii[:, 1:-1], jj[:, 1:-1]+1] != BLANK)
        start_1 = np.hstack([head_1.reshape(height, 1), body_1])

        indices = {"vertical": np.where(start_0), "horizontal": np.where(start_1)}
        return indices

    def get_used_words_and_enable(self, cell=None):
        """
        Get used_words and enable from the cell.

        Parameters
        ----------
        cell : numpy ndarray
            cell array
        
        Returns
        -------
        used_words : list
        enable : np.ndarray
        """
        if cell is None:
            cell = self.cell
        indices = self.get_word_indices(cell=cell)

        used_words = []
        enable = np.ones(cell.shape).astype(bool)
        for i, j in zip(indices["vertical"][0], indices["vertical"][1]):
            # used_words
            try:
                imax = i + np.where(cell[i:, j] == '')[0][0]
            except:
                imax = self.height
            used_words.append(''.join(cell[i:imax, j]))
            # enable
            if i != 0:
                enable[i-1, j] = False
            if imax != self.height:
                enable[imax, j] = False

        for i, j in zip(indices["horizontal"][0], indices["horizontal"][1]):
            # used_words
            try:
                jmax = j + np.where(cell[i, j:] == '')[0][0]
            except:
                jmax = self.width
            used_words.append(''.join(cell[i, j:jmax]))
            # enable
            if j != 0:
                enable[i, j-1] = False
            if jmax != self.width:
                enable[i, jmax] = False
        return np.array(used_words), enable

    def get_used_words(self, cell=None):
        """
        Get used_words from the cell.

        Parameters
        ----------
        cell : numpy ndarray
            cell array
        
        Returns
        -------
        used_words : list
            used_words
        """
        if cell is None:
            cell = self.cell
        indices = self.get_word_indices(cell=cell)
        used_words = []
        for i, j in zip(indices["vertical"][0], indices["vertical"][1]):
            try:
                imax = i + np.where(cell[i:, j] == '')[0][0]
            except:
                imax = self.height
            used_words.append(''.join(cell[i:imax, j]))
        for i, j in zip(indices["horizontal"][0], indices["horizontal"][1]):
            try:
                jmax = j + np.where(cell[i, j:] == '')[0][0]
            except:
                jmax = self.width
            used_words.append(''.join(cell[i, j:jmax]))
        return np.array(used_words)
    
    def get_enable(self, cell=None):
        """
        Get enable from the cell.

        Parameters
        ----------
        cell : numpy ndarray
            cell array
        
        Returns
        -------
        enable : numpy ndarray
            enable
        """
        if cell is None:
            cell = self.cell
        indices = self.get_word_indices(cell=cell)
        enable = np.ones(cell.shape).astype(bool)
        for i, j in zip(indices["vertical"][0], indices["vertical"][1]):
            # enable
            try:
                imax = i + np.where(cell[i:, j] == '')[0][0]
            except:
                imax = self.height
            if i != 0:
                enable[i-1, j] = False
            if imax != self.height:
                enable[imax, j] = False
        for i, j in zip(indices["horizontal"][0], indices["horizontal"][1]):
            # enable
            try:
                jmax = j + np.where(cell[i, j:] == '')[0][0]
            except:
                jmax = self.width
            if j != 0:
                enable[i, j-1] = False
            if jmax != self.height:
                enable[i, jmax] = False
        return enable

    def get_cover(self, cell=None):
        """
        Calculate the cover from the cell.

        Parameters
        ----------
        cell : numpy ndarray
            cell array
        
        Returns
        -------
        cover : numpy ndarray
            cover array
        """
        if cell is None:
            cell = self.cell
        cell = np.pad(cell, [(1,1), (1,1)], mode="constant", constant_values=BLANK)
        upper = cell[:-2, 1:-1] != BLANK
        lower = cell[2:, 1:-1] != BLANK
        vertical = (upper + lower)

        left = cell[1:-1, :-2] != BLANK
        right = cell[1:-1, 2:] != BLANK
        horizontal = (left + right)
        cover = vertical.astype(int) + horizontal.astype(int)
        cover *= (cell[1:-1, 1:-1] != BLANK)
        return cover

    def update_board(self, cell=None):
        """
        Update the cover and enable to fit the cell.

        Parameters
        ----------
        cell : numpy ndarray
            cell array
        """
        if cell:
            self.cell = cell
        enable = self.get_enable(cell=self.cell)
        cover = self.get_cover(cell)
        self.enable = enable
        self.cover = cover
        return

    