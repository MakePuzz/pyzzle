"""
* 特定のディレクトリ内の画像を動画に変換するプログラム
* 入力画像ファイルの拡張子は設定可能
* 以下のパラメータを設定して実行する
  imagedir: 入力画像の入っているディレクトリのパス
  input_ext: 入力画像の拡張子
  codec: 動画コーデックの設定は http://www.fourcc.org/codecs.php を参照
  Output: codecに対応した拡張子を持つ出力ファイル名
  fps: 出力する動画の1秒あたりのフレーム数
"""
import sys
import glob
import argparse

import cv2

parser = argparse.ArgumentParser(description="animate the image in <imagedir> at fps <fps>")
parser.add_argument("imagedir", type=str,
                    help="animate the image in a given directory")
parser.add_argument("-f", "--fps", type=int, default=6,
                    help="animate with a given fps, default=6")
parser.add_argument("-e", "--ext", type=str, default='png',
                    help="extention of images, default='png'")
parser.add_argument("-c", "--codec", type=str, default='mp4v',
                    help="codec of movie, default='mp4v'")
parser.add_argument("-o", "--output", type=str, default='out.mov',
                    help="name of output file, default='out.mov'")
args = parser.parse_args()

# Settings
imagedir = args.imagedir
fps = args.fps
input_ext = args.ext
codec = [args.codec[0],args.codec[1],args.codec[2],args.codec[3]]
output = args.output

# Open images
files = sorted(glob.glob(imagedir+'/*.'+ input_ext))
images = list(map(lambda file: cv2.imread(file), files))

fourcc = cv2.VideoWriter_fourcc(codec[0], codec[1], codec[2], codec[3])
video = cv2.VideoWriter(output, fourcc, fps, (images[0].shape[1], images[0].shape[0]))

# Sage as movie
for img in images:
    video.write(img)

video.release()
