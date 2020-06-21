# coding: utf-8
# In[]
import sys
from pathlib import PurePath

import numpy as np
import matplotlib.pyplot as plt

from pyzzle import Puzzle, Dictionary, Mask, Gravity
from pyzzle import utils

utils.debug_on()
# In[]
# Set parameters
width = 15
height = 15
mask = None #Mask.donut_s # 不要ならNone
gravity = None # 不要ならNone
dic = Dictionary.dataset["nikoli105"]
name = "Nikoli 105 contest"
epoch = 50

seed = 0
np.random.seed(seed=seed)
# In[]
# Make instances
puzzle = Puzzle(width=width, height=height, mask=mask, name=name) #, gravity=gravity

# In[]
# Dictionary
puzzle.import_dict(dic)

# In[]
obj_func = [
    "weight", "nwords"
]

# In[]
puzzle = puzzle.solve(epoch=epoch, n=3, optimizer="multi_start", of=obj_func, show=False, use_f=True)

# In[]
puzzle.show()

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

# In[]
puzzle.show()
