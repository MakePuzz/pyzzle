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
 9. ツイートするテキスト (-tまたは--textオプションで指定. ) 
10. Twitter API key (-kまたは--keyオプションで指定. 形式は，CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECREの順にカンマで繋ぐ. )
実行例：
python main_cli.py ../pyzzle/dict/pokemon.txt 15 15 -s 1 -e 5 -t 'Hello World!' -k CONSUMER_KEY,CONSUMER_SECRET,ACCESS_TOKEN,ACCESS_SECRE
"""
# In[]
import argparse

import numpy as np

from pyzzle import Puzzle, Dictionary, utils

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
parser.add_argument("-t", "--text", type=str,
                    help="Tweet text")
parser.add_argument("-k", "--key", type=str,
                    help="Tweitter API keys: CONSUMER_KEY,CONSUMER_SECRET,ACCESS_TOKEN,ACCESS_SECRET")

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
puzzle = Puzzle(width, height)
dic = Dictionary(dict_path)

if name is None:
    name = f"w{width}_h{height}_r{seed}_ep{epoch}"
puzzle.puzzle_name = name
if output is None:
    output = name + ".png"
if with_weight is True:
    obj_func = ["weight", "nwords"]
else:
    obj_func = ["nwords"]

puzzle.import_dict(dic)
puzzle = puzzle.solve(epoch=epoch, optimizer="local_search", of=obj_func, use_f=False)
puzzle.export_json(f"json/{puzzle.name}_w{width}_h{height}_r{seed}.json")
if width == height == 15:
    if name.isalnum(): # 英数字ならTrue
        title = f"Theme：{name}"
    else:
        title = f"テーマ：{name}"
    utils.export_image(puzzle.cell, puzzle.uwords[puzzle.uwords!=""], title=title, oname=f"fig/twitter_probrem_{puzzle.name}_w{width}_h{height}_r{seed}.png", dpi=144, answer=False)
    utils.export_image(puzzle.cell, puzzle.uwords[puzzle.uwords!=""], title=title, oname=f"fig/twitter_answer_{puzzle.name}_w{width}_h{height}_r{seed}.png", dpi=144, answer=True)
    
    
## Twitter below: To post a tweet with images.
import sys
import json
import requests
from requests_oauthlib import OAuth1

api_key = args.key.split(",")
text = args.text
image_path_list = [f"fig/twitter_probrem_{puzzle.name}_w{width}_h{height}_r{seed}.png", f"fig/twitter_answer_{puzzle.name}_w{width}_h{height}_r{seed}.png"]
twitter_oauth = OAuth1(api_key[0], api_key[1], api_key[2], api_key[3])  # Consumer Key, Consumer Secret, Access Token, Access Token Secert
url_media = "https://upload.twitter.com/1.1/media/upload.json"
url_text = "https://api.twitter.com/1.1/statuses/update.json"
media_ids = ""

for image_path in image_path_list:
    files = {"media":open(image_path, "rb")}
    # Post a image
    request = requests.post(url_media, files=files, auth=twitter_oauth)
    # Check the response to image
    if request.status_code != 200:
        # Upload failed
        print("Upload failed.")
        sys.exit(0)

    media_id = json.loads(request.text)["media_id"]
    media_id_string = json.loads(request.text)["media_id_string"]

    if media_ids == "":
        media_ids += media_id_string
    else:
        media_ids = media_ids + "," + media_id_string


params = {"status": text, "media_ids": [media_ids]}
# Post a tweet with images
request = requests.post(url_text, params = params, auth = twitter_oauth)

# Check the response to tweet text. Success = 200
print(request.status_code)
