# coding: utf-8
# In[]
from pathlib import PurePath

import numpy as np
import matplotlib.pyplot as plt

from pyzzle import Puzzle, Dictionary, Mask, Gravity, Optimizer
from pyzzle import utils

width = 15
height = 15
mask = None # Mask.donut_s # 不要ならNone
gravity = None # 不要ならNone
dic = Dictionary("../../dictionaries/pokemon.txt")
name = "Pyzzle"
epoch = 3
seed = 0
obj_func = [
    "uniqueness", "weight", "nwords", "area_rect", "ease_r"
]
# In[]
puzzle = Puzzle(width=width, height=height, mask=mask, name=name, seed=seed) #, gravity=gravity
puzzle.import_dict(dic)

# In[]
utils.debug_on()

optimizer = Optimizer.MultiStart(n=2, show=False, shrink=False, use_f=True)
puzzle = puzzle.solve(epoch=epoch, optimizer=optimizer, of=obj_func, time_limit=None)

# utils.logging_off()
# In[]
puzzle.show()
print(f"Component: {puzzle.component==1}")
print(f"Uniqueness: {puzzle.is_unique}")

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

    height_inch = 7
    width_inch = 6
    plt.rcParams.update({"figure.subplot.left": 0, "figure.subplot.bottom": 0, "figure.subplot.right": 1, "figure.subplot.top": 1})
    fig, [axl, axr] = plt.subplots(1, 2, figsize=(9*height_inch/7.5+width_inch, height_inch))
    utils.export_image(fig, axl, axr, puzzle.cell, puzzle.uwords[puzzle.uwords!=""], width, height, title=title, fontsize=18, oname=f"{base_dir}/fig/twitter_answer_{oname}.png", answer=True)
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