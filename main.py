# coding: utf-8
"""
Crossword Local Search
"""
# In[]
import os
import numpy as np
from matplotlib.font_manager import FontProperties

#os.chdir("/Users/taiga/Crossword-LocalSearch/Python")
from pyzzle import Puzzle, Dictionary, ObjectiveFunction, Optimizer

# In[]
# Set variables
fpath = "../dict/pokemon.txt"  # countries hokkaido animals kotowaza birds dinosaurs fishes sports
width = 15
height = 15
seed = 1
withweight = False

fp = FontProperties(fname="../fonts/SourceHanCodeJP.ttc", size=14)
np.random.seed(seed=seed)

# In[]
# Make instances
puzzle = Puzzle(width, height)
dic = Dictionary(fpath)
objFunc = ObjectiveFunction()
optimizer = Optimizer()

# In[]
puzzle.importDict(dic)
# Register and set method and compile
objFunc.register(["totalWeight", "solSize", "crossCount", "fillCount", "maxConnectedEmpties"])
optimizer.setMethod("localSearch")
puzzle.compile(objFunc=objFunc, optimizer=optimizer)

# In[]
# Solve
puzzle.firstSolve()
puzzle.solve(epoch=5)
print(f"SimpleSolution: {puzzle.isSimpleSol()}")
print(puzzle.cell)
print(f"単語リスト：{puzzle.usedWords[:puzzle.solSize]}")
puzzle.saveAnswerImage(f"fig/{dic.name}_w{width}_h{height}_r{seed}.png")
