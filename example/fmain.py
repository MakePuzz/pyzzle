# coding: utf-8
# In[]
import matplotlib.pyplot as plt
from pyzzle import Puzzle, Dictionary, Mask, utils
import sys
import numpy as np

sys.path.append("../")
# In[]
# Set parameters
width = 15
height = 15
mask = Mask.infinity_m
 # 不要ならNone
dic = Dictionary.dataset.pokemon
name = "Pyzzle"
seed = 5


np.random.seed(seed=seed)
# In[]
# Make instances
puzzle = Puzzle(width=3, height=6, mask=mask, name=name)

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
blank = "*"
cell = np.where(puzzle.cell == "", blank, puzzle.cell)
cell = np.array(list(map(lambda x: ord(x), cell.ravel()))).reshape(cell.shape)
cell = np.asfortranarray(cell.astype(np.int32).T)

height = puzzle.height
width = puzzle.width

n = puzzle.plc.size
w_len_max = max(puzzle.dic.w_len)

ori_s = puzzle.plc.ori
i_s = np.array(puzzle.plc.i) + 1 # fortran は 1から
j_s = np.array(puzzle.plc.j) + 1 # fortran は 1から
k_s = np.array(puzzle.plc.k)

words = list(map(lambda k: puzzle.dic.word[k], puzzle.plc.k))
words_int = np.full([n, w_len_max], ord(blank), dtype=np.int32)
w_lens = np.zeros(n, dtype=np.int32, order="F")
for i in range(n):
    word = puzzle.dic.word[puzzle.plc.k[i]]
    w_lens[i] = len(word)
    words_int[i,:w_lens[i]] = list(map(ord, word))
words_int = np.asfortranarray(words_int)
enable = np.asfortranarray(puzzle.enable.astype(np.int32))

# In[]
from pyzzle.Puzzle import add_to_limit
is_used = add_to_limit(height, width, n, w_len_max, ord(blank), ori_s, i_s, j_s, k_s, words_int, w_lens, cell, enable)

# In[]
# ocell = np.array(list(map(lambda x: chr(x), cell.ravel()))).reshape(puzzle.cell.shape)
cell = np.array(list(map(lambda x: list(map(chr, x)), cell)))
ocell = np.where(cell == blank, "", cell)
puzzle.cell = ocell.T
puzzle.show()


# In[]

# print(ocell)

# In[]
"""
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
puzzle.show_log()
plt.savefig(f"fig/log_{puzzle.epoch}ep.png")

# %%
puzzle.show()

# %%
"""