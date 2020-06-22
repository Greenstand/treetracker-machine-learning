import argparse
import os
import cv2
import pandas as pd
from blur.blur import BlurDetection
from hash.hash import *
import matplotlib.pyplot as plt


def blur_stats(dir, viz=False):
    blurrer = BlurDetection()
    df = {}

    if viz:
        fig, axarr = plt.subplots(2, figsize=(20,10))
        plt.suptitle(dir)
        [a.grid() for a in axarr]
        axarr[0].set_xlabel("Var")
        axarr[0].set_ylabel("Max")
        axarr[0].set_title("Laplacian Var and Max of Image Set")
        axarr[1].set_xlabel("Skew")
        axarr[1].set_ylabel("Kurtosis")
        axarr[1].set_title("Grayscale Skew/Kurtosis")

    for f, _, d in os.walk(dir):
        for file in d:
            idt, ext = os.path.splitext(file)
            fullpath = os.path.join(f, idt + ext)
            if ext in (".jpg", ".png") :
                im = cv2.imread(fullpath)
                varpp, maxe = blurrer.lp_variance_max(im)
                exc_sk, exc_kr = blurrer.hist_stats(im)

                if viz:
                    if varpp > 2e-2:
                        c = "r"
                    else:
                        c = "g"
                    axarr[0].scatter(varpp, maxe, c=c)
                    axarr[0].annotate(idt, (varpp, maxe))
                    if exc_sk > 1.5 or exc_kr > 1.8:
                        c = "r"
                    else:
                        c = "g"
                    axarr[1].scatter(exc_sk, exc_kr, c=c)
                    axarr[1].annotate(idt, (exc_sk, exc_kr))

                df[int(idt)] = [fullpath, maxe, varpp, exc_sk, exc_kr]
    plt.show()
    df = pd.DataFrame.from_dict(df).T
    df.columns = ["path", "lpmax", "varpp", "skew", "kurtosis"]
    return df


if __name__ == "__main__":
    data_dir = input("Enter filename in data/ that contains downloaded images: ")
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data", data_dir)

    if not os.path.exists (data_dir):
        raise FileNotFoundError("Check to make sure this directory exists")
    hash_size = int(input("Hash size in bytes:"))
    rsz_1 = int(input("Resize x, y (square):"))

    blur_stats(data_dir, viz=True)