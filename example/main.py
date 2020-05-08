# coding: utf-8
"""
Crossword Local Search
"""
# In[]
import os, sys
import numpy as np

#os.chdir("/Users/taiga/Crossword-LocalSearch/Python")
sys.path.append("../")
from pyzzle import Puzzle, Dictionary, ObjectiveFunction, Optimizer

# In[]
# Set variables
fpath = "../dict/pokemon.txt"  # countries hokkaido animals kotowaza birds dinosaurs fishes sports
width = 15
height = 15
seed = 1
with_weight = False

np.random.seed(seed=seed)

# In[]
# Make instances
puzzle = Puzzle(width, height)
dic = Dictionary(fpath)
obj_func = ObjectiveFunction()
optimizer = Optimizer()

# In[]
puzzle.import_dict(dic)
# Register and set method and compile
obj_func.register(["total_weight", "sol_size", "cross_count", "fill_count", "max_connected_empties"])
optimizer.set_method("local_search")
puzzle.compile(obj_func=obj_func, optimizer=optimizer)

# In[]
# Solve
puzzle.first_solve()
puzzle.solve(epoch=5)
print(f"SimpleSolution: {puzzle.is_simple_sol()}")
print(puzzle.cell)
print(f"単語リスト：{puzzle.used_words[:puzzle.sol_size]}")
puzzle.save_answer_image(f"fig/{dic.name}_w{width}_h{height}_r{seed}.png")
