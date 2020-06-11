# coding: utf-8
# In[]
import matplotlib.pyplot as plt
from pyzzle import Puzzle, FancyPuzzle, Dictionary, Mask
import sys
import numpy as np

sys.path.append("../")
# In[]
# Set parameters
width = 15
height = 15
mask = Mask.donut_s # 不要ならNone
dic = Dictionary.dataset.logo
name = "Pyzzle"
seed = 5


np.random.seed(seed=seed)
# In[]
# Make instances
puzzle = Puzzle(mask=mask, name=name)

# In[]
# Dictionary
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
puzzle.first_solve(use_f=False)

# In[]
puzzle.solve(epoch=5, optimizer="local_search", of=obj_func, use_f=True)
print(f"unique solution: {puzzle.is_unique}")

# In[]
print(puzzle.cell)
print(f"単語リスト：{puzzle.used_words[:puzzle.nwords]}")
oname = f"{dic.name}_w{puzzle.width}_h{puzzle.height}_ep{puzzle.epoch}_seed{puzzle.seed}"
puzzle.save_answer_image(f"fig/answer_{oname}_answer.png")
puzzle.save_problem_image(f"fig/problem_{oname}_problem.png")
puzzle.export_json(f"json/{oname}.json")
# puzzle.to_pickle(f"pickle/{oname}.pickle")

# In[]
puzzle.show_log()
plt.savefig(f"fig/{oname}_log.png")

# %%
puzzle.show()

# %%
