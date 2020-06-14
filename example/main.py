# coding: utf-8
# In[]
import sys
from pathlib import PurePath

import numpy as np
import matplotlib.pyplot as plt

from pyzzle import Puzzle, FancyPuzzle, Dictionary, Mask, Gravity

sys.path.append("../")
# In[]
# Set parameters
width = 15
height = 15
mask = Mask.donut_s # 不要ならNone
# gravity = Gravity.hokkaido # 不要ならNone
dic = Dictionary.dataset.logo
name = "Pyzzle"
seed = 5
epoch = 5


np.random.seed(seed=seed)
# In[]
# Make instances
puzzle = Puzzle(width=width, height=height, mask=mask, name=name) #, gravity=gravity

# In[]
# Dictionary
puzzle.import_dict(dic)

# In[]
obj_func = [
    "gravity",
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
puzzle.solve(epoch=epoch, optimizer="local_search", of=obj_func, show=False, use_f=True)

# In[]
# puzzle.solve(epoch=300, optimizer="local_search", of=obj_func, show=False, use_f=True)

# In[]
print(f"unique solution: {puzzle.is_unique}")

# In[]
base_dir = str(PurePath(__file__).parent)
oname = f"{dic.name}_w{puzzle.width}_h{puzzle.height}_ep{puzzle.epoch}_seed{puzzle.seed}"
puzzle.save_answer_image(f"{base_dir}/fig/{oname}_answer.png")
puzzle.save_problem_image(f"{base_dir}/fig/{oname}_problem.png")
puzzle.export_json(f"{base_dir}/json/{oname}.json")
# puzzle.to_pickle(f"pickle/{oname}.pickle")

# In[]
puzzle.show_log()
plt.savefig(f"{base_dir}/fig/{oname}_log.png")

# %%
puzzle.show()

# %%
# from pyzzle import PyzzleAPI
# api = PyzzleAPI()
# a = api.get_all_puzzles()
# a.text
# %%
