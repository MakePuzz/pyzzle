# coding: utf-8
# In[]
import sys
from pathlib import PurePath

import numpy as np
import matplotlib.pyplot as plt

from pyzzle import Puzzle, Dictionary, Mask, Gravity
from pyzzle import utils

# Set parameters
width = 15
height = 15
mask = None # Mask.donut_s # 不要ならNone
gravity = None # 不要ならNone
dic = Dictionary("../../dictionaries/pokemon.txt")
name = "Pyzzle"
epoch = 3

seed = 0
np.random.seed(seed=seed)
# In[]
# Make instances
puzzle = Puzzle(width=width, height=height, mask=mask, name=name) #, gravity=gravity

# In[]
puzzle.import_dict(dic)

# In[]
obj_func = [
    "uniqueness", "weight", "nwords", "area_rect", "ease_r"
]

# In[]
utils.debug_on()
puzzle = puzzle.solve(epoch=epoch, time_limit=None, n=1, optimizer="multi_start", of=obj_func, show=False, use_f=True)
utils.logging_off()

# In[]
# puzzle.solve(epoch=300, optimizer="local_search", of=obj_func, show=False, use_f=True)
# In[]
puzzle.show()
print(f"component should be 1: {puzzle.component}")
print(f"unique solution: {puzzle.is_unique}")

# In[]
base_dir = str(PurePath(__file__).parent)
oname = f"{puzzle.name}_w{puzzle.width}_h{puzzle.height}_ep{puzzle.epoch}_seed{puzzle.seed}"
puzzle.save_answer_image(f"{base_dir}/fig/{oname}_answer.png")
puzzle.save_problem_image(f"{base_dir}/fig/{oname}_problem.png")
puzzle.export_json(f"{base_dir}/json/{oname}.json")

if width == height == 15:
    if name.isalnum(): # 英数字ならTrue
        title = f"Theme：{name}"
    else:
        title = f"テーマ：{name}"
    utils.export_image(puzzle.cell, puzzle.uwords[puzzle.uwords!=""], title=title, oname=f"{base_dir}/fig/twitter_answer_{oname}.png", answer=True)
# puzzle.to_pickle(f"pickle/{oname}.pickle")

# In[]
puzzle.show_log()
plt.savefig(f"{base_dir}/fig/{oname}_log.png")

# %%
# from pyzzle import PyzzleAPI
# api = PyzzleAPI()
# a = api.get_all_puzzles()
# a.text


# # %%
# from pyzzle import Puzzle
# puzzle = Puzzle.from_json(f"{base_dir}/json/{oname}.json")

# # %%
# puzzle.show()

# %%
# from pyzzle import Puzzle, utils
# puzzle = Puzzle.from_json(f"json/ファイル名.json")
# utils.export_image(puzzle.cell, puzzle.uwords[puzzle.uwords!=""], title="テーマ：テスト", oname=f"fig/twitter_test.png", answer=True)
# %%
# %%
