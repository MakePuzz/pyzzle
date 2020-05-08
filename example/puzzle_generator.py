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

出力：
 出力ファイルのパス, 唯一解判定の結果

実行例：
    python puzzle_generator.py dict/pokemon.txt -w 15 -h 15 -s 1 -e 15

importして使う場合：
    import puzzle_generator
    ans, prob, simple = puzzle_generator.get(fpath, width, height, seed, epoch, title, with_weight, output)
"""
# In[]
import os
import argparse
import numpy as np
from matplotlib.font_manager import FontProperties

#os.chdir("/Users/taiga/Crossword-LocalSearch/Python")
from pyzzle import Puzzle, Dictionary, ObjectiveFunction, Optimizer

# In[]
def get(fpath, width, height, seed, epoch, title, with_weight, output):
    np.random.seed(seed=seed)
    # Make instances
    puzzle = Puzzle(width, height, msg=False)
    dic = Dictionary(fpath, msg=False)
    obj_func = ObjectiveFunction(msg=False)
    optimizer = Optimizer(msg=False)

    if title is None:
        title = f"{dic.name}_w{width}_h{height}_r{seed}_ep{epoch}"
    puzzle.puzzle_title = title
    if output is None:
        output = title + ".png"

    puzzle.import_dict(dic, msg=False)
    # Register and set method and compile
    if with_weight is True:
        obj_func.register(["total_weight","sol_size", "cross_count", "fill_count", "max_connected_empties"], msg=False)
    else:
        obj_func.register(["sol_size", "cross_count", "fill_count", "max_connected_empties"], msg=False)
    optimizer.set_method("local_search", msg=False)
    puzzle.compile(obj_func=obj_func, optimizer=optimizer, msg=False)

    # Solve
    puzzle.first_solve()
    puzzle.solve(epoch=epoch)
    is_simple = puzzle.is_simple_sol()
    pass_answer = f"fig/{output}_answer.png"
    pass_problem = f"fig/{output}_problem.png"
    puzzle.save_answer_image(pass_answer)
    puzzle.save_problem_image(pass_problem)
    return pass_problem, pass_answer, is_simple

if __name__ == '__main__':
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
    with_weight = args.weight
    output = args.output
    print(get(fpath, width, height, seed, epoch, title, with_weight, output))