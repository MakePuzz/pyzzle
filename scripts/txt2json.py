"""
* 特定のディレクトリ内の辞書ファイル(.txt)をまとめて1つのjsonファイルに変換するプログラム
* 以下のパラメータを設定して実行する
  dir: 辞書の入っているディレクトリのパス
  output: 出力ファイル名
"""
import glob
import os
import json
import argparse

parser = argparse.ArgumentParser(description="generate json from text in <dir>")
parser.add_argument("dir", type=str, help="directory of text files")
parser.add_argument("-o", "--output", type=str, default="dict.json", help="name of output file")
args = parser.parse_args()

files = glob.glob(f"{args.dir}/*.txt")
dicts = {}
for file_path in files:
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, mode="r", encoding="utf-8") as text:
        words = [w.rstrip("\n") for w in text.readlines() if w != "" or w != "\n"]
        if ' ' in words[0]:
            words = [{"word": w.split()[0], "weight": int(w.split()[1])} for w in words if ' ' in w]
    dicts[file_name] = words
with open(args.output, mode="w", encoding="utf-8") as j:
    json.dump(dicts, j, indent=2, ensure_ascii=False)


