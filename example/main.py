# coding: utf-8
"""
Crossword Local Search
"""
# In[]
import os, sys
import numpy as np

#os.chdir("/Users/taiga/Crossword-LocalSearch/Python")
sys.path.append("../")
from pyzzle import Puzzle, FancyPuzzle, Dictionary, ObjectiveFunction, Optimizer

# In[]
# Set variables
fpath = "../dict/pokemon.txt"  # countries hokkaido animals kotowaza birds dinosaurs fishes sports
width = 15
height = 15
seed = 1
with_weight = False

np.random.seed(seed=seed)

# In[]
## Make instances
### FuncyPuzzle
mask = np.array([
    [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
], dtype="bool")
puzzle = FancyPuzzle(mask, "ドーナツパズル")

### Puzzle (normal)
# puzzle = Puzzle(width, height)

### Dictionary, ObjectiveFunction, Optimizer
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

# In[]
puzzle.solve(epoch=2)

# In[]
print(f"unique solution: {puzzle.is_unique}")
print(puzzle.cell)
print(f"単語リスト：{puzzle.used_words[:puzzle.sol_size]}")
puzzle.save_answer_image(f"fig/{dic.name}_w{width}_h{height}_r{seed}.png")

puzzle.export_json(f"json/{dic.name}_w{width}_h{height}_r{seed}.json")