# coding: utf-8
# In[]
import sys
import numpy as np

sys.path.append("../")
from pyzzle import Puzzle, FancyPuzzle, Dictionary, ObjectiveFunction, Optimizer, Mask
# In[]
# Set variables
width = 15
height = 15
with_weight = False

seed = 5
np.random.seed(seed=seed)
# In[]
## Make instances
### FuncyPuzzle
puzzle = FancyPuzzle(Mask.donut_s, "Donut Puzzle")

### Puzzle (normal)
# puzzle = Puzzle(width, height)

### Dictionary, ObjectiveFunction, Optimizer
dic = Dictionary.dataset.logo
puzzle.import_dict(dic)
# In[]

obj_func = [
            "circulation",
            "weight",
            "nwords", 
            "cross_count", 
            "fill_count", 
            "max_connected_empties", 
            "difficulty"
        ]

# In[]
puzzle.first_solve()

# In[]
puzzle.solve(epoch=5, optimizer="local_search", of=obj_func)
print(f"unique solution: {puzzle.is_unique}")

# In[]
print(puzzle.cell)
print(f"単語リスト：{puzzle.used_words[:puzzle.nwords]}")
oname = f"{dic.name}_w{puzzle.width}_h{puzzle.height}_ep{puzzle.epoch}_seed{puzzle.seed}.png"
puzzle.save_answer_image(f"fig/answer_{oname}")
puzzle.save_problem_image(f"fig/problem_{oname}")
puzzle.export_json(f"json/{oname[:-4]}.json")
# puzzle.to_pickle(f"pickle/{oname[:-4]}.pickle")

# In[]
import matplotlib.pyplot as plt
puzzle.show_log()
plt.savefig(f"fig/log_{puzzle.epoch}ep.png")

# %%
puzzle.show()

# %%
