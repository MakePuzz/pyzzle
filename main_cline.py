# coding: utf-8
"""
Crossword Local Search by command line

引数：
 1. 辞書ファイルのパス
 2. パズルの横幅
 3. パズルの縦幅
 4. シード値（-sまたは--seedオプションで指定. デフォルトは66666）
 5. エポック数（-eまたは--epochオプションで指定. デフォルトは10）
 6. パズルのタイトル（-tまたは--titleオプションで指定. デフォルトは{辞書名}_w{width}_h{height}_r{seed}_ep{epoch}）
 7. 重みを考慮するかどうか（-wまたは--weightオプションでフラグとして指定. デフォルトはFalse）
 8. 出力ファイル名（-oまたは--outputオプションで指定. デフォルトは{title}.png）

実行例：
python get_puzzle.py dict/pokemon.txt -w 15 -h 15 -s 1 -e 15
"""
# In[]
import os
import argparse
import numpy as np
from matplotlib.font_manager import FontProperties

#os.chdir("/Users/taiga/Crossword-LocalSearch/Python")
from pyzzle import Puzzle, Dictionary, ObjectiveFunction, Optimizer

# In[]
parser = argparse.ArgumentParser(description="make a puzzle with given parameters")
parser.add_argument("fpath", type=str,
                    help="file path of a dictionary")
parser.add_argument("width", type=int,
                    help="witdh of the puzzle")
parser.add_argument("height", type=int,
                    help="height of the puzzle")
parser.add_argument("-s", "--seed", type=int, default=66666,
                    help="random seed value, default=66666")
parser.add_argument("-e", "--epoch", type=int, default=10,
                    help="epoch number of local search, default=10")
parser.add_argument("-t", "--title", type=str,
                    help="title of the puzzle")
parser.add_argument("-w", "--weight", action="store_true",
                    help="flag of consider the weight, default=False")
parser.add_argument("-o", "--output", type=str,
                    help="name of the output image file")
args = parser.parse_args()

# settings
fpath = args.fpath # countries hokkaido animals kotowaza birds dinosaurs fishes sports pokemon typhoon cats s_and_p100
width = args.width
height = args.height
seed = args.seed
epoch = args.epoch
title = args.title
withWeight = args.weight
output = args.output

fp = FontProperties(fname="../jupyter/fonts/SourceHanCodeJP.ttc", size=14)
np.random.seed(seed=seed)

# In[]
# Make instances
puzzle = Puzzle(width, height, msg=False)
dic = Dictionary(fpath, msg=False)
objFunc = ObjectiveFunction(msg=False)
optimizer = Optimizer(msg=False)

if title is None:
    title = f"{dic.name}_w{width}_h{height}_r{seed}_ep{epoch}"
puzzle.puzzleTitle = title
if output is None:
    output = title + ".png"

# In[]
puzzle.importDict(dic, msg=False)
# Register and set method and compile
if withWeight is True:
    objFunc.register(["totalWeight","solSize", "crossCount", "fillCount", "maxConnectedEmpties"], msg=False)
else:
    objFunc.register(["solSize", "crossCount", "fillCount", "maxConnectedEmpties"], msg=False)
optimizer.setMethod("localSearch", msg=False)
puzzle.compile(objFunc=objFunc, optimizer=optimizer, msg=False)

# In[]
# Solve
puzzle.firstSolve()
puzzle.solve(epoch=epoch)
print(f"SimpleSolution: {puzzle.isSimpleSol()}")
print(puzzle.cell)
print(f"単語リスト：{puzzle.usedWords[:puzzle.solSize]}")
puzzle.saveAnswerImage(f"fig/{output}_answer", fp=fp)
puzzle.saveProblemImage(f"fig/{output}_problem", fp=fp)
