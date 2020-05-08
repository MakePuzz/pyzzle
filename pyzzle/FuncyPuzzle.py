import os
import sys
import copy
import datetime
import time
import math
import itertools
import unicodedata
import collections
import pickle
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from IPython.display import display, HTML
import matplotlib.pyplot as plt

sys.path.append('../python')
from pyzzle import Puzzle, Dictionary, Placeable, ObjectiveFunction, Optimizer
from src import utils

class FancyPuzzle(Puzzle):
    def __init__(self, width, height, mask=None, title="スケルトンパズル", msg=True):
        if mask is None:
            mask = np.ones(width*height, dtype="bool").reshape(height, width)
        self.mask = mask
        
        super().__init__(width, height, title, msg)

    def isEnabledAdd(self, div, i, j, word, wLen):
        """
        This method determines if a word can be placed
        """
        if div == 0:
            if np.any(self.mask[i:i+wLen, j] == False):
                return 7
        if div == 1:
            if np.any(self.mask[i, j:j+wLen] == False):
                return 7
    
        return super().isEnabledAdd(div, i, j, word, wLen)

    def saveImage(self, data, fpath, list_label="[Word List]", dpi=100):
        """
        This method generates and returns a puzzle image with a word list
        """
        # Generate puzzle image
        colors = np.where(self.cover<1, "#000000", "#FFFFFF")
        df = pd.DataFrame(data)

        fig=plt.figure(figsize=(16, 8), dpi=dpi)
        ax1=fig.add_subplot(121) # puzzle
        ax2=fig.add_subplot(122) # word list
        ax1.axis("off")
        ax2.axis("off")
        fig.set_facecolor('#EEEEEE')
        
        # Draw puzzle
        ax1_table = ax1.table(cellText=df.values, cellColours=colors, cellLoc="center", bbox=[0, 0, 1, 1], fontsize=20)
        ax1.set_title(label=f"*** {self.title} ***", size=20)
        
        # delete unmasked cells
        mask = np.where(puzzle.mask == False)
        for i, j in list(zip(mask[0], mask[1])):
            del ax1_table._cells[i, j]

        # Draw word list
        words = [word for word in self.usedWords if word != ""]
        if words == []:
            words = [""]
        words.sort()
        words = sorted(words, key=len)

        rows = self.height
        cols = math.ceil(len(words)/rows)
        padnum = cols*rows - len(words)
        words += ['']*padnum
        words = np.array(words).reshape(cols, rows).T

        ax2_table = ax2.table(cellText=words, cellColours=None, cellLoc="left", edges="open", bbox=[0, 0, 1, 1])
        ax2.set_title(label=list_label, size=20)
        for _, cell in ax2_table.get_celld().items():
            cell.set_text_props(size=18)
        plt.tight_layout()
        plt.savefig(fpath, dpi=dpi)
        plt.close()
    
    def jump(self, idx):
        tmp_puzzle = self.__class__(self.width, self.height, self.mask, self.title, msg=False)
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
            elif code in (2,3):
                tmp_puzzle._drop(div, i, j, k)
        tmp_puzzle.initSol = True
        return tmp_puzzle

    def move(self, direction, n=0, limit=False):
        super().move(direction, n, limit)