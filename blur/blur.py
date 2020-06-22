import cv2
import os
from scipy.stats import skew, kurtosis
import numpy as np
from data.data_management import *
import matplotlib.pyplot as plt
from matplotlib import colors



class BlurDetection ():
    def __init__(self):
        pass
    def lp_variance_max(self, img):
        '''
        Tutorial from https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
        LoG operator returns edges, so low variance means more blurred due to fewer edges.
        Returns per variance per pixel, allowing larger images to not be penalized as blurry.
        :param img: image to process
        :return:
        '''
        if img.ndim > 2:
            print ("Stats measured in grayscale, converting image to grayscale")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lapl = cv2.Laplacian(img, cv2.CV_64F)
        return lapl.var() / np.prod(img.shape), np.max(lapl)

    def fft_filter(self, img, bw):
        '''
        Low pass filter using 2D FFT
        :param img: image to process
        :param bw: bw (int) low-pass filter bandwidth = freq.shape - bw * 2
        :return: filtered img in [0-1] intensity range.
        '''
        freq = np.fft.fft2(img / np.max(img)) # normalizes image for numerical stability
        mask = np.zeros(freq.shape)
        mask [bw: -bw, bw: -bw] = 1.0

        # mask [mask.shape[0] // 2 - bw : mask.shape[0] // 2 + bw, mask.shape[1] // 2 - bw : mask.shape[1] // 2 + bw] = 1.0
        freq = np.multiply(freq, mask)
        return np.fft.ifft2(freq)


    def max_laplacian(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.max(cv2.convertScaleAbs(cv2.Laplacian(gray, 3)))

    def hist_stats(self, img):
        if img.ndim > 2:
            print ("Histogram stats measured in grayscale, converting image to grayscale")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        counts, bars = np.histogram(img.flatten(), bins=np.arange(0,256),density=True)
        sk = skew(counts)
        kr = kurtosis(counts)
        return sk, kr

    def entropy_pp(self, img):
        """
        per pixel entropy of a histogram
        :param img: grayscale image to analyze
        :return:
        """
        if img.ndim > 2:
            print ("Entropy per pixel measured in grayscale, converting image to grayscale")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = np.histogram(img.flatten(), bins=np.arange(0,255), density=True)
        return -(hist[0] * np.log(np.abs(hist[0]) + 1e-8)).sum()


if __name__=="__main__":
    homepath = os.getcwd()[:-4]
    data = GreenstandDataset('nov11data.csv')
    random_ids = np.loadtxt(os.path.join(homepath, "data/onepercentids.txt"))
    randoms = [data.read_image_from_db(os.path.join(homepath, 'data/random_zeroone_percent_db/'), key=int(r)) for r in
               random_ids] # len 20
    blurrer = BlurDetection()
    vars = []
    maxes = []
    f, axarr = plt.subplots(5, 4, figsize=(20, 20))
    for i in range(0, 5):
        for j in range(0, 4):
            axarr[i, j].set_title(int(random_ids[4 * i + j]))
            axarr[i, j].imshow(randoms[4 * i + j], cmap="Reds")
    plt.show()

    f, ax = plt.subplots()
    ax.grid()
    ax.set_title("Max by Variance Laplacian")
    ax.set_xlabel("Variance")
    ax.set_ylabel("Max")
    for i in range(5):
        for j in range(4):
            var, max = blurrer.lp_variance_max(randoms[4 * i + j])# 0.01 seems like a fair threshold for this: over this variance is
            name = random_ids[4 * i + j]
            ax.scatter(var, max)
            ax.annotate(int(name), (var, max))
    plt.show()
    exit()
    f, axarr = plt.subplots(5,4, figsize=(25,25))
    for i in range(0,5):
        for j in range(0,4):
            var, max = np.round(blurrer.lp_variance_max(randoms[4 * i + j]), 3) # 0.01 seems like a fair threshold for this: over this variance is
            s = cv2.convertScaleAbs(cv2.Laplacian(cv2.cvtColor(randoms[4  * i + j], cv2.COLOR_BGR2GRAY), 3)).flatten()
            counts, bars, patches = axarr[i, j].hist(s, bins=np.arange(0,256), density=True)
            sk = np.round(skew(counts), 2)
            kr = np.round(kurtosis(counts), 2)
            axarr[i, j].set_title(f"ppvar: {var}, kurtosis: {kr}", fontsize=20)
            # We'll color code by height, but you could use any scalar
            fracs = counts / counts.max()
            # we need to normalize the data to 0..1 for the full range of the colormap
            norm = colors.Normalize(fracs.min(), fracs.max())
            if np.abs(kr) <= 10:
                cm = plt.cm.Greens
            else:
                cm = plt.cm.Reds
            # Now, we'll loop through our objects and set the color of each accordingly
            for thisfrac, thispatch in zip(fracs, patches):
                color =cm(thisfrac)
                thispatch.set_facecolor(color)

    plt.suptitle("Laplacian Statistics", fontsize=24)
    plt.show()
    f, axarr = plt.subplots(5,4, figsize=(25,25))
    for i in range(0,5):
        for j in range(0,4):
            gray = cv2.cvtColor(randoms[4  * i + j], cv2.COLOR_BGR2GRAY)
            entropy = blurrer.entropy_pp(gray)
            s = cv2.convertScaleAbs(gray).flatten()
            counts, bars, patches = axarr[i, j].hist(s, bins=np.arange(0,256), density=True)
            sk = np.round(skew(counts), 2)
            kr = np.round(kurtosis(counts), 2)
            axarr[i, j].set_title(f"skew: {sk}, kurtosis: {kr} ", fontsize=20)
            fracs = counts / counts.max()
            # we need to normalize the data to 0..1 for the full range of the colormap
            norm = colors.Normalize(fracs.min(), fracs.max())
            if np.abs(kr) <= 5:
                cm = plt.cm.Greens
            else:
                cm = plt.cm.Reds
            # Now, we'll loop through our objects and set the color of each accordingly
            for thisfrac, thispatch in zip(fracs, patches):
                color =cm(thisfrac)
                thispatch.set_facecolor(color)

    plt.suptitle("Grayscale Statistics", fontsize=24)
    plt.show()

