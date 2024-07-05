# coding: utf-8
"""
Crossword Local Search by command line

引数：
 1. 辞書ファイルのパス
 2. パズルの横幅
 3. パズルの縦幅
 4. シード値（-sまたは--seedオプションで指定. デフォルトは66666）
 5. エポック数（-eまたは--epochオプションで指定. デフォルトは10）
 6. パズルのタイトル（-n,または--nameオプションで指定. デフォルトは{辞書名}_w{width}_h{height}_r{seed}_ep{epoch}）
 7. 重みを考慮するかどうか（-wまたは--weightオプションでフラグとして指定. デフォルトはFalse）
 8. 出力ファイル名（-oまたは--outputオプションで指定. デフォルトは{name}.png）
実行例：
python main_cli.py ../src/pyzzle/dict/animals.txt 6 6 -s 1 -e 5 --use_f
"""
# In[]
import os
import argparse

import numpy as np

import pyzzle
# In[]
parser = argparse.ArgumentParser(description="make a puzzle with given parameters")
parser.add_argument("dict_path", type=str,
                    help="file path of a dictionary")
parser.add_argument("width", type=int,
                    help="width of the puzzle")
parser.add_argument("height", type=int,
                    help="height of the puzzle")
parser.add_argument("-s", "--seed", type=int, default=66666,
                    help="random seed value, default=66666")
parser.add_argument("-e", "--epoch", type=int, default=10,
                    help="epoch number of local search, default=10")
parser.add_argument("-n", "--name", type=str,
                    help="name of the puzzle")
parser.add_argument("-w", "--weight", action="store_true",
                    help="flag of consider the weight, default=False")
parser.add_argument("-o", "--output", type=str,
                    help="name of the output image file")
parser.add_argument("-f", "--use_f", action="store_true",
                    help="use Fortran function in local search")
args = parser.parse_args()

# settings
dict_path = args.dict_path # countries hokkaido animals kotowaza birds dinosaurs fishes sports pokemon typhoon cats s_and_p100
width = args.width
height = args.height
seed = args.seed
epoch = args.epoch
name = args.name
with_weight = args.weight
output = args.output

np.random.seed(seed=seed)

# In[]
puzzle = pyzzle.Puzzle(width, height)
print(puzzle)

dic = pyzzle.Dictionary(dict_path)

if name is None:
    name = os.path.basename(dict_path).split(".")[0]
puzzle.puzzle_name = name
if output is None:
    output = name + ".png"
if with_weight:
    obj_func = ["weight", "nwords"]
else:
    obj_func = ["nwords"]

puzzle.import_dict(dic)
optimizer = pyzzle.Optimizer.LocalSearch(show=False, shrink=False, use_f=args.use_f)
puzzle = puzzle.solve(epoch=epoch, optimizer=optimizer, of=obj_func)
puzzle.export_json(f"json/{puzzle.name}_w{width}_h{height}_r{seed}.json")

def is_halfwidth(char):
    return len(char.encode('utf-8')) <= 1

if is_halfwidth(dic[0]["word"][0]): # 半角ならTrue
    halfspace = True
else:
    halfspace = False

if width == height == 15:
    if halfspace: # 英数字ならTrue
        title = f"Theme：{name}"
    else:
        title = f"テーマ：{name}"
    oname_problem = f"fig/twitter_problem_{puzzle.name}_w{width}_h{height}_r{seed}.png"
    oname_answer = f"fig/twitter_answer_{puzzle.name}_w{width}_h{height}_r{seed}.png"
    pyzzle.utils.export_image(puzzle.cell, puzzle.uwords[puzzle.uwords!=""], title=title, oname=oname_problem, dpi=144, answer=False)
    pyzzle.utils.export_image(puzzle.cell, puzzle.uwords[puzzle.uwords!=""], title=title, oname=oname_answer, dpi=144, answer=True)
    print(f"Saved as {oname_answer} and {oname_problem}")


pyzzle.utils.show_2Darray(puzzle.cell, puzzle.mask, halfspace=halfspace)