import argparse
import os
import cv2
import pandas as pd
from blur.blur import BlurDetection, viz_blur_stats
from tools.viz import  image_gallery
from hash.hash import *
import matplotlib.pyplot as plt




if __name__ == "__main__":
    data_dir = input("Enter filename in data/ that contains downloaded images: ")
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data", data_dir)

    if not os.path.exists (data_dir):
        raise FileNotFoundError("Check to make sure this directory exists")
    # hash_size = int(input("Hash size in bytes:"))
    # rsz_1 = int(input("Resize x, y (square):"))
    images = []
    titles = []
    extensions = [".jpg", ".png"]
    for f, _, d in os.walk(data_dir):
        for fil in d:
            fullpath = os.path.join(f, fil)
            if os.path.splitext(fullpath)[1] in extensions:
                im = cv2.imread(fullpath)
                images.append(im)
                titles.append(fil)
    image_gallery(images, titles)
    viz_blur_stats(images, titles, threshs=[0.01, 600, 245])