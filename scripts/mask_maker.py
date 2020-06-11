import argparse

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="Export mask array from input image")
parser.add_argument("img", type=str,
                    help="path to image")
parser.add_argument("-r", "--rate", type=float, default=1.0,
                    help="scale rate, default=1.0")
parser.add_argument("-o", "--output", type=str, default="mask.txt",
                    help="name of output file, default='mask.txt'")
args = parser.parse_args()

path = args.img
rate = args.rate
oname = args.output

gray_img = Image.open(path).convert("L")
gray_img = np.array(gray_img.resize((int(gray_img.width*rate), int(gray_img.height*rate))))

thresh = np.mean(gray_img)
gray_01 = np.where(gray_img>=thresh, 0, 1).reshape(gray_img.shape).astype("int")

gray_str = gray_01.astype("str")
for i in range(gray_str.shape[0]):
    gray_str[i,0] = "["+gray_str[i,0]
    gray_str[i,-1] = gray_str[i,-1]+"],"
gray_str[0,0] = "np.array([["+gray_str[0,0][1]
gray_str[-1,-1] = gray_str[-1,-1][0]+"]], dtype=bool)"

np.savetxt(oname, gray_str, delimiter=",", fmt="%s")
print("Successfully exported mask as '"+oname+"'. shape:"+str(gray_str.shape))