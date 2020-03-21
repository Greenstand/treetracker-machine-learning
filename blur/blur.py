import cv2
import os
import numpy as np
from data.data_management import *
import matplotlib.pyplot as plt



class BlurDetection ():
    def __init__(self, thresh):
        self.thresh = thresh

    def log_var(self, img):
        '''
        Tutorial from https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
        LoG operator returns edges, so low variance means more blurred due to fewer edges
        :param img: image to process
        :return:
        '''

        return cv2.Laplacian(grayscale(img), cv2.CV_64F).var()

    def fft_filter(self, img, bw):
        '''
        Low pass filter using 2D FFT
        :param img: image to process
        :param bw: bw (int) low-pass filter bandwidth
        :return: filtered img
        '''
        freq = np.fft.fft2(img)
        # mask = np.ones(freq.shape)
        # mask [mask.shape[0] // 2 - bw : mask.shape[0] // 2 + bw, mask.shape[1] // 2 - bw : mask.shape[1] // 2 + bw] = 1
        # freq = np.multiply(freq, mask)
        return np.fft.ifft2(freq)



def grayscale(img):
    return img.dot([0.07, 0.72, 0.21])


if __name__=="__main__":
    homepath = os.getcwd()[:-4]
    data = GreenstandDataset('nov11data.csv')
    random_ids = np.loadtxt(os.path.join(homepath, "data/onepercentids.txt"))
    randoms = [data.read_image_from_db(os.path.join(homepath, 'data/random_zeroone_percent_db/'), key=int(r)) for r in
               random_ids] # len 20
    f, axarr = plt.subplots(5,4, figsize=(20,20))
    blurrer = BlurDetection(100)
    for i in range(0,5):
        for j in range(0,4):
            var = blurrer.log_var(randoms[4 * i + j])
            axarr[i, j].set_title(var)
            axarr[i, j].imshow(grayscale(randoms[4  * i + j]), cmap='gray')
    plt.show()
    f, axarr = plt.subplots(5,4, figsize=(20,20))
    for i in range(0,5):
        for j in range(0,4):
            var = blurrer.log_var(cv2.GaussianBlur(randoms[4 * i + j], (9,9), 0))
            axarr[i, j].set_title(var)
            axarr[i, j].imshow(cv2.GaussianBlur(grayscale(randoms[4 * i + j]), (7,7), 0), cmap='gray')

    plt.show()
    f, axarr = plt.subplots(5,4, figsize=(20,20))

    for i in range(0,5):
        for j in range(0,4):
            imgfft = blurrer.fft_filter(randoms[4 * i + j], 3 * randoms[4 * i + j].shape[0] // 4)
            print (np.min(imgfft), np.max(imgfft))
            axarr[i, j].imshow(np.abs(imgfft), cmap='gray')

    plt.show()