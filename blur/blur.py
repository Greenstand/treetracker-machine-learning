import cv2
import os
from scipy.stats import skew, kurtosis
import numpy as np
from data.data_management import *
import matplotlib.pyplot as plt



class BlurDetection ():
    def __init__(self, thresh):
        self.thresh = thresh

    def log_var(self, img):
        '''
        Tutorial from https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
        LoG operator returns edges, so low variance means more blurred due to fewer edges.
        Returns per variance per pixel, allowing larger images to not be penalized as blurry.
        :param img: image to process
        :return:
        '''

        return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() / np.prod(img.shape)

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

def grayscale(img):
    """
    shorthand way to convert to grayscale
    :param img:
    :return:
    """
    return img.dot([0.07, 0.72, 0.21])

def entropy_pp(img):
    """
    per pixel entropy of a histogram
    :param img: grayscale image to analyze
    :return:
    """
    hist = np.histogram(img.flatten(), bins=np.arange(0,255), density=True)
    return -(hist[0] * np.log(np.abs(hist[0]) + 1e-8)).sum()


if __name__=="__main__":
    homepath = os.getcwd()[:-4]
    data = GreenstandDataset('nov11data.csv')
    random_ids = np.loadtxt(os.path.join(homepath, "data/onepercentids.txt"))
    randoms = [data.read_image_from_db(os.path.join(homepath, 'data/random_zeroone_percent_db/'), key=int(r)) for r in
               random_ids] # len 20
    f, axarr = plt.subplots(5,4, figsize=(20,20))
    for i in range(0,5):
        for j in range(0,4):
            gray = cv2.cvtColor(randoms[4  * i + j], cv2.COLOR_BGR2GRAY)
            entropy = entropy_pp(gray)
            axarr[i, j].set_title(f"{entropy:.2E}", fontsize=20)
            s = cv2.convertScaleAbs(gray)
            axarr[i, j].imshow(s, cmap="gray")
    plt.show()

    f, axarr = plt.subplots(5,4, figsize=(20,20))
    blurrer = BlurDetection(100)
    for i in range(0,5):
        for j in range(0,4):
            var = np.round(blurrer.log_var(randoms[4 * i + j]), 3) # 0.01 seems like a fair threshold for this: over this variance is
            s = cv2.convertScaleAbs(cv2.Laplacian(cv2.cvtColor(randoms[4  * i + j], cv2.COLOR_BGR2GRAY), 3)).flatten()
            counts, bars, patches = axarr[i, j].hist(s, bins=np.arange(0,256), density=True)
            sk = np.round(skew(counts), 2)
            kr = np.round(kurtosis(counts), 2)
            axarr[i, j].set_title(f"ppvar: {var}, kurtosis: {kr}", fontsize=20)
    plt.suptitle("Laplacian Statistics", fontsize=24)
    plt.show()
    f, axarr = plt.subplots(5,4, figsize=(20,20))
    for i in range(0,5):
        for j in range(0,4):
            gray = cv2.cvtColor(randoms[4  * i + j], cv2.COLOR_BGR2GRAY)
            entropy = entropy_pp(gray)
            s = cv2.convertScaleAbs(gray).flatten()
            counts, bars, patches = axarr[i, j].hist(s, bins=np.arange(0,256), density=True)
            sk = np.round(skew(counts), 2)
            kr = np.round(kurtosis(counts), 2)
            axarr[i, j].set_title(f"skew: {sk}, kurtosis: {kr} ", fontsize=20)
    plt.suptitle("Grayscale Statistics", fontsize=24)
    plt.show()

