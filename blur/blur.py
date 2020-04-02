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
        LoG operator returns edges, so low variance means more blurred due to fewer edges.
        Returns per variance per pixel, allowing larger images to not be penalized as blurry.
        :param img: image to process
        :return:
        '''

        return cv2.Laplacian(grayscale(img), cv2.CV_64F).var() / np.prod(img.shape)

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
            var = blurrer.log_var(randoms[4 * i + j]) # 0.01 seems like a fair threshold for this: over this variance is
            axarr[i, j].set_title(f"{var:.2E}")
            axarr[i, j].imshow(grayscale(randoms[4  * i + j]), cmap="gray")
    plt.show()
    f, axarr = plt.subplots(5,4, figsize=(20,20))
    for i in range(0,5):
        for j in range(0,4):
            fourier = np.fft.fft2(grayscale(randoms[4 * i + j]), axes=(0,1))
            mask = np.zeros_like(fourier)
            k = 100
            mask[0:k, 0:k] = 1

            fourier = np.fft.ifft2(np.multiply(fourier, mask))
            var = np.var(fourier)
            axarr[i, j].set_title(f"{var:.2E}")

            r = axarr[i, j].imshow((np.abs(fourier)).astype(np.int), cmap="gray")
            # axarr[i, j].imshow(cv2.GaussianBlur(grayscale(randoms[4 * i + j]), (3,3), sigmaX=0), cmap='gray')
    plt.show()
    # f, axarr = plt.subplots(5,4, figsize=(20,20))

    # for i in range(0,5):
    #     for j in range(0,4):
    #         imgfft = blurrer.fft_filter(randoms[4 * i + j], 200)
    #         axarr[i, j].imshow(np.real(imgfft), cmap='gray')
    #
    # plt.show()