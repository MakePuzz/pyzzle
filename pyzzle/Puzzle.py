import os
import copy
import datetime
import itertools
import math
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import pickle

from pyzzle.Placeable import Placeable
from pyzzle.Dictionary import Dictionary


class Puzzle:
    """
    The Almighty Puzzle Class.
    Example usage::
        from pyzzle import Puzzle
        puzzle = Puzzle(8,8)
        puzzle.add(0,0,0,'hoge')
        puzzle.add(1,0,0,'hotel')
        puzzle.add(1,0,2,'gateball')
        puzzle.add(0,3,0,'elevetor')
        puzzle.show()
        puzzle.saveProbremImage()
        puzzle.saveAnswerImage()
    """

    def __init__(self, width, height, title="Criss Cross", msg=True):
        self.width = width
        self.height = height
        self.totalWeight = 0
        self.title = title
        self.cell = np.full(width * height, "", dtype="unicode").reshape(height, width)
        self.cover = np.zeros(width * height, dtype="int").reshape(height, width)
        self.label = np.zeros(width * height, dtype="int").reshape(height, width)
        self.enable = np.ones(width * height, dtype="bool").reshape(height, width)
        self.usedWords = np.full(width * height, "", dtype=f"U{max(width, height)}")
        self.usedPlcIdx = np.full(width * height, -1, dtype="int")
        self.solSize = 0
        self.history = []
        self.baseHistory = []
        self.log = None
        self.epoch = 0
        self.nlabel = None
        self.firstSolved = False
        self.initSeed = None
        self.dic = Dictionary(msg=False)
        self.plc = Placeable(self.width, self.height, self.dic, msg=False)
        self.objFunc = None
        self.optimizer = None
        # self.fp = os.path.get_path()
        ## Message
        if msg is True:
            print(f"{self.__class__.__name__} object has made.")
            print(f" - title       : {self.title}")
            print(f" - width       : {self.width}")
            print(f" - height      : {self.height}")
            print(f" - cell' shape : (width, height) = ({self.cell.shape[0]},{self.cell.shape[1]})")

    def __str__(self):
        return self.title

    def reinit(self, all=False):
        """
        This method reinitilize Puzzle informations

        Parameters
        ----------
        all : bool default False
            Reinitilize completely if all is True
        """
        if all is True:
            self.dic = None
            self.plc = None
            self.objFunc = None
            self.optimizer = None
        self.totalWeight = 0
        self.enable = np.ones(self.width * self.height, dtype="bool").reshape(self.height, self.width)
        self.cell = np.full(self.width * self.height, "", dtype="unicode").reshape(self.height, self.width)
        self.cover = np.zeros(self.width * self.height, dtype="int").reshape(self.height, self.width)
        self.label = np.zeros(self.width * self.height, dtype="int").reshape(self.height, self.width)
        self.enable = np.ones(self.width * self.height, dtype="bool").reshape(self.height, self.width)
        self.usedWords = np.full(self.width * self.height, "", dtype=f"U{max(self.width, self.height)}")
        self.usedPlcIdx = np.full(self.width * self.height, -1, dtype="int")
        self.solSize = 0
        self.baseHistory = []
        self.history = []
        self.log = None
        self.epoch = 0
        self.firstSolved = False
        self.initSeed = None

    def in_ipynb(self):
        """
        Are we in a jupyter notebook?
        """
        try:
            return 'ZMQ' in get_ipython().__class__.__name__
        except NameError:
            return False

    def importDict(self, dictionary, msg=True):
        """
        This method imports Dictionary to Puzzle

        Parameters
        ----------
        dictionary : Dictionary
            Dictionary object imported by Puzzle
        """
        self.dic = dictionary
        self.plc = Placeable(self.width, self.height, self.dic, msg=msg)

    def isEnabledAdd(self, div, i, j, word, wLen):
        """
        This method determines if a word can be placed

        Parameters
        ----------
        div : int
            Direction of the word (0:Vertical, 1:Horizontal)
        i : int
            Row number of the word
        j : int
            Col number of the word
        word : str
            The word to be checked whether it can be added
        wLen : int
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
        if div == 0:
            empties = self.cell[i:i + wLen, j] == ""
        if div == 1:
            empties = self.cell[i, j:j + wLen] == ""

        # If 0 words used, return True
        if self.solSize is 0:
            return 0

        # If the preceding and succeeding cells are already filled
        if div == 0:
            if i > 0 and self.cell[i - 1, j] != "":
                return 1
            if i + wLen < self.height and self.cell[i + wLen, j] != "":
                return 1
        if div == 1:
            if j > 0 and self.cell[i, j - 1] != "":
                return 1
            if j + wLen < self.width and self.cell[i, j + wLen] != "":
                return 1

        # At least one place must cross other words
        if np.all(empties == True):
            return 2

        # Judge whether correct intersection
        where = np.where(empties == False)[0]
        if div == 0:
            jall = np.full(where.size, j, dtype="int")
            if np.any(self.cell[where + i, jall] != np.array(list(word))[where]):
                return 3
        if div == 1:
            iall = np.full(where.size, i, dtype="int")
            if np.any(self.cell[iall, where + j] != np.array(list(word))[where]):
                return 3

        # If the same word is in use, return False
        if word in self.usedWords:
            return 4

        # If neighbor cells are filled except at the intersection, return False
        where = np.where(empties == True)[0]
        if div == 0:
            jall = np.full(where.size, j, dtype="int")
            # Left side
            if j > 0 and np.any(self.cell[where + i, jall - 1] != ""):
                return 5
            # Right side
            if j < self.width - 1 and np.any(self.cell[where + i, jall + 1] != ""):
                return 5
        if div == 1:
            iall = np.full(where.size, i, dtype="int")
            # Upper
            if i > 0 and np.any(self.cell[iall - 1, where + j] != ""):
                return 5
            # Lower
            if i < self.height - 1 and np.any(self.cell[iall + 1, where + j] != ""):
                return 5

        # US/USA, DOMINICA/DOMINICAN problem
        if div == 0:
            if np.any(self.enable[i:i + wLen, j] == False) or np.all(empties == False):
                return 6
        if div == 1:
            if np.any(self.enable[i, j:j + wLen] == False) or np.all(empties == False):
                return 6

        # If Break through the all barrier, return True
        return 0

    def _add(self, div, i, j, k):
        """
        This internal method places a word at arbitrary positions.
        If it can not be arranged, nothing is done.

        Parameters
        ----------
        div : int
            Direction of the word (0:Vertical, 1:Horizontal)
        i : int
            Row number of the word
        j : int
            Col number of the word
        k : int
            The number of the word registered in Placeable
        """
        word = self.dic.word[k]
        weight = self.dic.weight[k]
        wLen = self.dic.wLen[k]

        # Judge whether adding is enabled
        code = self.isEnabledAdd(div, i, j, word, wLen)
        if code is not 0:
            return code

        # Put the word to puzzle
        if div == 0:
            self.cell[i:i + wLen, j] = list(word)[0:wLen]
        if div == 1:
            self.cell[i, j:j + wLen] = list(word)[0:wLen]

        # Set the prohibited cell before and after placed word
        if div == 0:
            if i > 0:
                self.enable[i - 1, j] = False
            if i + wLen < self.height:
                self.enable[i + wLen, j] = False
        if div == 1:
            if j > 0:
                self.enable[i, j - 1] = False
            if j + wLen < self.width:
                self.enable[i, j + wLen] = False

        # Update cover array
        if div == 0:
            self.cover[i:i + wLen, j] += 1
        if div == 1:
            self.cover[i, j:j + wLen] += 1

        # Update properties
        wordIdx = self.dic.word.index(word)
        self.usedPlcIdx[self.solSize] = self.plc.invP[div, i, j, wordIdx]
        self.usedWords[self.solSize] = self.dic.word[k]
        self.solSize += 1
        self.totalWeight += weight
        self.history.append((1, wordIdx, div, i, j))
        return 0

    def add(self, div, i, j, word, weight=0):
        if type(word) is int:
            k = word
        elif type(word) is str:
            self.dic.add(word, weight)
            self.plc._compute([word], self.dic.size - 1)
            k = self.dic.word.index(word)
        else:
            raise TypeError()
        self._add(div, i, j, k)

    def addToLimit(self):
        """
        This method adds the words as much as possible
        """
        # Make a random index of plc
        randomIndex = np.arange(self.plc.size)
        np.random.shuffle(randomIndex)

        # Add as much as possible
        solSizeTmp = None
        while self.solSize != solSizeTmp:
            solSizeTmp = self.solSize
            dropIdx = []
            for i, r in enumerate(randomIndex):
                code = self._add(self.plc.div[r], self.plc.i[r], self.plc.j[r], self.plc.k[r])
                if code is not 2:
                    dropIdx.append(i)
            randomIndex = np.delete(randomIndex, dropIdx)
        return

    def firstSolve(self):
        """
        This method creates an initial solution
        """
        # Check the firstSolved
        if self.firstSolved:
            raise RuntimeError("'firstSolve' method has already called")

        # Save initial seed number
        self.initSeed = np.random.get_state()[1][0]
        # Add as much as possible
        self.addToLimit()
        self.firstSolved = True

    def show(self, ndarray=None):
        """
        This method displays a puzzle

        Parameters
        ----------
        ndarray : ndarray, optional
            A Numpy.ndarray for display.
            If not specified, it thought to slef.cell
        """
        if ndarray is None:
            ndarray = self.cell
        if self.in_ipynb() is True:
            styles = [
                dict(selector="th", props=[("font-size", "90%"),
                                           ("text-align", "center"),
                                           ("color", "#ffffff"),
                                           ("background", "#777777"),
                                           ("border", "solid 1px white"),
                                           ("width", "30px"),
                                           ("height", "30px")]),
                dict(selector="td", props=[("font-size", "105%"),
                                           ("text-align", "center"),
                                           ("color", "#161616"),
                                           ("background", "#dddddd"),
                                           ("border", "solid 1px white"),
                                           ("width", "30px"),
                                           ("height", "30px")]),
                dict(selector="caption", props=[("caption-side", "bottom")])
            ]
            df = pd.DataFrame(ndarray)
            df = (df.style.set_table_styles(styles).set_caption(f"Puzzle({self.width},{self.height}), solSize:{self.solSize}, Dictionary:[{self.dic.fpath}]"))
            display(df)
        else:
            ndarray = np.where(ndarray == "", "  ", ndarray)
            print(ndarray)

    def logging(self):
        """
        This method logs the current objective function values
        """
        if self.objFunc is None:
            raise RuntimeError("Logging method must be executed after compilation method")
        if self.log is None:
            self.log = pd.DataFrame(columns=self.objFunc.getFuncs())
            self.log.index.name = "epoch"
        tmpSe = pd.Series(self.objFunc.getScore(self, all=True), index=self.objFunc.getFuncs())
        self.log = self.log.append(tmpSe, ignore_index=True)

    def _drop(self, div, i, j, k, isKick=False):
        """
        This internal method removes the specified word from the puzzle.

        Parametes
        ----------
        div : int
            Direction of the word (0:Vertical, 1:Horizontal)
        i : int
            Row number of the word
        j : int
            Col number of the word
        k : int
            The number of the word registered in Placeable
        isKick : bool default False
            If this dropping is in the kick process, it should be True.
            This information is used in making ``history``.

        Notes
        -----
        This method pulls out the specified word without taking it
        into consideration, which may break the connectivity of the puzzle
        or cause LAOS / US / USA problems.
        """
        # Get p, pidx
        p = self.plc.invP[div, i, j, k]
        pidx = np.where(self.usedPlcIdx == p)[0][0]

        wLen = self.dic.wLen[k]
        weight = self.dic.weight[k]
        # Pull out a word
        if div == 0:
            self.cover[i:i + wLen, j] -= 1
            where = np.where(self.cover[i:i + wLen, j] == 0)[0]
            jall = np.full(where.size, j, dtype="int")
            self.cell[i + where, jall] = ""
        if div == 1:
            self.cover[i, j:j + wLen] -= 1
            where = np.where(self.cover[i, j:j + wLen] == 0)[0]
            iall = np.full(where.size, i, dtype="int")
            self.cell[iall, j + where] = ""
        # Update usedWords, usedPlcIdx, solSize, totalWeight
        self.usedWords = np.delete(self.usedWords, pidx)  # delete
        self.usedWords = np.append(self.usedWords, "")  # append
        self.usedPlcIdx = np.delete(self.usedPlcIdx, pidx)  # delete
        self.usedPlcIdx = np.append(self.usedPlcIdx, -1)  # append
        self.solSize -= 1
        self.totalWeight -= weight
        # Insert data to history
        code = 3 if isKick else 2
        self.history.append((code, k, div, i, j))
        # Release prohibited cells
        removeFlag = True
        if div == 0:
            if i > 0:
                if i > 2 and np.all(self.cell[[i - 3, i - 2], [j, j]] != ""):
                    removeFlag = False
                if j > 2 and np.all(self.cell[[i - 1, i - 1], [j - 2, j - 1]] != ""):
                    removeFlag = False
                if j < self.width - 2 and np.all(self.cell[[i - 1, i - 1], [j + 1, j + 2]] != ""):
                    removeFlag = False
                if removeFlag == True:
                    self.enable[i - 1, j] = True
            if i + wLen < self.height:
                if i + wLen < self.height - 2 and np.all(self.cell[[i + wLen + 1, i + wLen + 2], [j, j]] != ""):
                    removeFlag = False
                if j > 2 and np.all(self.cell[[i + wLen, i + wLen], [j - 2, j - 1]] != ""):
                    removeFlag = False
                if j < self.width - 2 and np.all(self.cell[[i + wLen, i + wLen], [j + 1, j + 2]] != ""):
                    removeFlag = False
                if removeFlag == True:
                    self.enable[i + wLen, j] = True
        if div == 1:
            if j > 0:
                if j > 2 and np.all(self.cell[[i, i], [j - 3, j - 2]] != ""):
                    removeFlag = False
                if i > 2 and np.all(self.cell[[i - 2, i - 1], [j - 1, j - 1]] != ""):
                    removeFlag = False
                if i < self.height - 2 and np.all(self.cell[[i + 1, i + 2], [j - 1, j - 1]] != ""):
                    removeFlag = False
                if removeFlag == True:
                    self.enable[i, j - 1] = True
            if j + wLen < self.width:
                if j + wLen < self.width - 2 and np.all(self.cell[[i, i], [j + wLen + 1, j + wLen + 2]] != ""):
                    removeFlag = False
                if i > 2 and np.all(self.cell[[i - 2, i - 1], [j + wLen, j + wLen]] != ""):
                    removeFlag = False
                if i < self.height - 2 and np.all(self.cell[[i + 1, i + 2], [j + wLen, j + wLen]] != ""):
                    removeFlag = False
                if removeFlag == True:
                    self.enable[i, j + wLen] = True

    def drop(self, word=None, divij=None):
        """
        This method removes the specified word from the puzzle.

        Parametes
        ----------
        word : int or str
            The word number or word in the puzlle to drop.
        divij : tuple of int, optional
            Tuple indicating a specific word to drop.

        Notes
        -----
        This method pulls out the specified word without taking it
        into consideration, which may break the connectivity of the puzzle
        or cause LAOS / US / USA problems.
        """
        if word is None and divij is None:
            raise ValueError("'word' or 'divij' must be specified")
        if word is not None:
            if type(word) is int:
                k = word
            elif type(word) is str:
                k = self.dic.word.index(word)
            else:
                raise TypeError("'word' must be int or str")
            for p in self.usedPlcIdx:
                if self.plc.k[p] == k:
                    div = self.plc.div[p]
                    i = self.plc.i[p]
                    j = self.plc.j[p]
                    break
        if divij is not None:
            if type(divij) not in (list, tuple):
                raise TypeError(f"divij must be list or tuple")
            if len(divij) is not 3:
                raise ValueError(f"Length of 'divij' must be 3, not {len(divij)}")
            for p in self.usedPlcIdx:
                _div = self.plc.div[p]
                _i = self.plc.i[p]
                _j = self.plc.j[p]
                if _div == divij[0] and _i == divij[1] and _j == divij[2]:
                    k = puzzle.plc.k[p]
                    break
        self._drop(divij[0], divij[1], divij[2], k)

    def collapse(self):
        """
        This method collapses connectivity of the puzzle.
        """
        # If solSize = 0, return
        if self.solSize == 0:
            return

        # Make a random index of solSize
        randomIndex = np.arange(self.solSize)
        np.random.shuffle(randomIndex)

        # Drop words until connectivity collapses
        tmpUsedPlcIdx = copy.deepcopy(self.usedPlcIdx)
        for r, p in enumerate(tmpUsedPlcIdx[randomIndex]):
            # Get div, i, j, k, wLen
            div = self.plc.div[p]
            i = self.plc.i[p]
            j = self.plc.j[p]
            k = self.plc.k[p]
            wLen = self.dic.wLen[self.plc.k[p]]
            # If '2' is aligned in the cover array, the word can not be dropped
            if div == 0:
                if not np.any(np.diff(np.where(self.cover[i:i + wLen, j] == 2)[0]) == 1):
                    self._drop(div, i, j, k)
            if div == 1:
                if not np.any(np.diff(np.where(self.cover[i, j:j + wLen] == 2)[0]) == 1):
                    self._drop(div, i, j, k)

            self.label, self.nlabel = ndimage.label(self.cover)
            if self.nlabel >= 2:
                break

    def kick(self):
        """
        This method kicks elements except largest CCL.
        """
        # If solSize = 0, return
        if self.solSize == 0:
            return

        mask = self.cover > 0
        self.label, self.nlabel = ndimage.label(mask)
        sizes = ndimage.sum(mask, self.label, range(self.nlabel + 1))
        largestCCL = sizes.argmax()

        # Erase elements except CCL ('kick' in C-program)
        for idx, p in enumerate(self.usedPlcIdx[:self.solSize]):
            if p == -1:
                continue
            if self.label[self.plc.i[p], self.plc.j[p]] != largestCCL:
                self._drop(self.plc.div[p], self.plc.i[p], self.plc.j[p], self.plc.k[p], isKick=True)

    def compile(self, objFunc, optimizer, msg=True):
        """
        This method compiles the objective function and
        optimization method into the Puzzle instance.

        Parameters
        ----------
        objFunc : ObjectiveFunction
            ObjectiveFunction object for compile to Puzzle
        optimizer : Optimizer
            Optimizer object for compile to Puzzle
        """
        self.objFunc = objFunc
        self.optimizer = optimizer

        if msg is True:
            print("compile succeeded.")
            print(" --- objective functions:")
            for funcNum in range(len(objFunc)):
                print(f"  |-> {funcNum} {objFunc.registeredFuncs[funcNum]}")
            print(f" --- optimizer: {optimizer.method}")

    def solve(self, epoch):
        """
        This method repeats the solution improvement by
        the specified number of epoch.

        Parameters
        ----------
        epoch : int
            The number of epoch
        """
        if self.firstSolved is False:
            raise RuntimeError("'firstSolve' method has not called")
        if epoch is 0:
            raise ValueError("'epoch' must be lather than 0")
        exec(f"self.optimizer.{self.optimizer.method}(self, {epoch})")
        print(" --- done")

    def showLog(self, title="Objective Function's time series", grid=True, **kwargs):
        """
        This method shows log of objective functions.

        Parameters
        ----------
        title : str default "Objective Function's time series"
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

    def isSimpleSol(self):
        """
        This method determines whether it is the simple solution
        """
        rtnBool = True

        # Get word1
        for s, p in enumerate(self.usedPlcIdx[:self.solSize]):
            i = self.plc.i[p]
            j = self.plc.j[p]
            word1 = self.usedWords[s]
            if self.plc.div[p] == 0:
                crossIdx1 = np.where(self.cover[i:i + len(word1), j] == 2)[0]
            elif self.plc.div[p] == 1:
                crossIdx1 = np.where(self.cover[i, j:j + len(word1)] == 2)[0]
            # Get word2
            for t, q in enumerate(self.usedPlcIdx[s + 1:self.solSize]):
                i = self.plc.i[q]
                j = self.plc.j[q]
                word2 = self.usedWords[s + t + 1]
                if len(word1) != len(word2):  # If word1 and word2 have different lengths, they can not be replaced
                    continue
                if self.plc.div[q] == 0:
                    crossIdx2 = np.where(self.cover[i:i + len(word2), j] == 2)[0]
                if self.plc.div[q] == 1:
                    crossIdx2 = np.where(self.cover[i, j:j + len(word2)] == 2)[0]
                replaceable = True
                # Check cross part from word1
                for w1idx in crossIdx1:
                    if word1[w1idx] != word2[w1idx]:
                        replaceable = False
                        break
                # Check cross part from word2
                if replaceable is True:
                    for w2idx in crossIdx2:
                        if word2[w2idx] != word1[w2idx]:
                            replaceable = False
                            break
                # If word1 and word2 are replaceable, this puzzle doesn't have a simple solution -> return False
                if replaceable is True:
                    print(f" - words '{word1}' and '{word2}' are replaceable")
                    rtnBool = False
        return rtnBool

    def saveImage(self, data, fpath, list_label="[Word List]", dpi=100):
        """
        This method generates and returns a puzzle image with a word list
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
        ax1_table = ax1.table(cellText=df.values, cellColours=colors, cellLoc="center", bbox=[0, 0, 1, 1])
        ax1_table.auto_set_font_size(False)
        ax1_table.set_fontsize(18)
        ax1.set_title(label="*** " + self.title + " ***", size=20)

        # Draw word list
        words = [word for word in self.usedWords if word != ""]
        if words == []:
            words = [""]
        words.sort()
        words = sorted(words, key=len)

        rows = self.height
        cols = math.ceil(len(words) / rows)
        padnum = cols * rows - len(words)
        words += [''] * padnum
        words = np.array(words).reshape(cols, rows).T

        ax2_table = ax2.table(cellText=words, cellColours=None, cellLoc="left", edges="open", bbox=[0, 0, 1, 1])
        ax2.set_title(label=list_label, size=20)
        ax2_table.auto_set_font_size(False)
        ax2_table.set_fontsize(18)
        plt.tight_layout()
        plt.savefig(fpath, dpi=dpi)
        plt.close()

    def saveProblemImage(self, fpath="problem.png", list_label="[Word List]", dpi=100):
        """
        This method generates and returns a puzzle problem with a word list
        """
        data = np.full(self.width * self.height, "", dtype="unicode").reshape(self.height, self.width)
        self.saveImage(data, fpath, list_label, dpi)

    def saveAnswerImage(self, fpath="answer.png", list_label="[Word List]", dpi=100):
        """
        This method generates and returns a puzzle answer with a word list.
        """
        data = self.cell
        self.saveImage(data, fpath, list_label, dpi)

    def jump(self, idx):
        tmp_puzzle = self.__class__(self.width, self.height, self.title, msg=False)
        tmp_puzzle.dic = copy.deepcopy(self.dic)
        tmp_puzzle.plc = Placeable(self.width, self.height, tmp_puzzle.dic, msg=False)
        tmp_puzzle.optimizer = copy.deepcopy(self.optimizer)
        tmp_puzzle.objFunc = copy.deepcopy(self.objFunc)
        tmp_puzzle.baseHistory = copy.deepcopy(self.baseHistory)

        if set(self.history).issubset(self.baseHistory) is False:
            if idx <= len(self.history):
                tmp_puzzle.baseHistory = copy.deepcopy(self.history)
            else:
                raise RuntimeError('This puzzle is up to date')

        for code, k, div, i, j in tmp_puzzle.baseHistory[:idx]:
            if code == 1:
                tmp_puzzle._add(div, i, j, k)
            elif code == 2:
                tmp_puzzle._drop(div, i, j, k, isKick=False)
            elif code == 3:
                tmp_puzzle._drop(div, i, j, k, isKick=True)
        tmp_puzzle.firstSolved = True
        return tmp_puzzle

    def getPrev(self, n=1):
        if len(self.history) - n < 0:
            return self.jump(0)
        return self.jump(len(self.history) - n)

    def getNext(self, n=1):
        if len(self.history) + n > len(self.baseHistory):
            return self.getLatest()
        return self.jump(len(self.history) + n)

    def getLatest(self):
        return self.jump(len(self.baseHistory))

    def toPickle(self, name=None, msg=True):
        """
        This method saves Puzzle object as a binary file
        """
        now = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
        name = name or f"{now}_{self.dic.name}_{self.width}_{self.height}_{self.initSeed}_{self.epoch}.pickle"
        with open(name, mode="wb") as f:
            pickle.dump(self, f)
        if msg is True:
            print(f"Puzzle has pickled to the path '{name}'")

    def getRect(self):
        rows = np.any(self.cover, axis=1)
        cols = np.any(self.cover, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def move(self, direction, n=0, limit=False):
        rmin, rmax, cmin, cmax = self.getRect()
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
            n2limit = {1: rmin, 2: self.height - (rmax + 1), 3: cmin, 4: self.width - (cmax + 1)}
            n = n2limit[direction]

        if direction is 1:
            if rmin < n:
                raise RuntimeError()
            self.cell = np.roll(self.cell, -n, axis=0)
            self.cover = np.roll(self.cover, -n, axis=0)
            self.label = np.roll(self.label, -n, axis=0)
            self.enable = np.roll(self.enable, -n, axis=0)
            for i, p in enumerate(self.usedPlcIdx[:self.solSize]):
                self.usedPlcIdx[i] = self.plc.invP[self.plc.div[p], self.plc.i[p] - n, self.plc.j[p], self.plc.k[p]]
        if direction is 2:
            if self.height - (rmax + 1) < n:
                raise RuntimeError()
            self.cell = np.roll(self.cell, n, axis=0)
            self.cover = np.roll(self.cover, n, axis=0)
            self.label = np.roll(self.label, n, axis=0)
            self.enable = np.roll(self.enable, n, axis=0)
            for i, p in enumerate(self.usedPlcIdx[:self.solSize]):
                self.usedPlcIdx[i] = self.plc.invP[self.plc.div[p], self.plc.i[p] + n, self.plc.j[p], self.plc.k[p]]
        if direction is 3:
            if cmin < n:
                raise RuntimeError()
            self.cell = np.roll(self.cell, -n, axis=1)
            self.cover = np.roll(self.cover, -n, axis=1)
            self.label = np.roll(self.label, -n, axis=1)
            self.enable = np.roll(self.enable, -n, axis=1)
            for i, p in enumerate(self.usedPlcIdx[:self.solSize]):
                self.usedPlcIdx[i] = self.plc.invP[self.plc.div[p], self.plc.i[p], self.plc.j[p] - n, self.plc.k[p]]
        if direction is 4:
            if self.width - (cmax + 1) < n:
                raise RuntimeError()
            self.cell = np.roll(self.cell, n, axis=1)
            self.cover = np.roll(self.cover, n, axis=1)
            self.label = np.roll(self.label, n, axis=1)
            self.enable = np.roll(self.enable, n, axis=1)
            for i, p in enumerate(self.usedPlcIdx[:self.solSize]):
                self.usedPlcIdx[i] = self.plc.invP[self.plc.div[p], self.plc.i[p], self.plc.j[p] + n, self.plc.k[p]]

        self.history.append((4, direction, n))
