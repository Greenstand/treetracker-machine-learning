import cv2
import numpy as np


class ImageHasher():
    def __init__(self, hash_size):
        self.size = hash_size

    def hamming_distance(self,img1_hash, img2_hash)
        '''
        Count number of bits that are different between two binary hashes
        :param img1_hash: (int) self.hash_size bit hash of first candidate image
        :param img2_hash: (int) self.hash_size bit hash of second candidate image
        :return: (int) number of bits that are not equal in provided hashes
        '''
        return np.count_nonzero(img1_hash != img2_hash)


    def average_hash(self, img):
        resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (self.size + 1, self.size))
        x_large = resized > np.mean(resized)
        return np.sum(np.exp2(np.flatnonzero(x_large)))


    def difference_hash(self, img):
        '''
        Credit to https://www.pyimagesearch.com/2017/11/27/image-hashing-opencv-python/
        for implementation tutorial.

        :param img: (np.ndarray) the RGB image to find a difference hash function of
        :param hash_size: (int) the number of bits in the hash (ex. setting to 8 yields 2**8=64 bit address)
        :return: (int) 2 ** hash_size bit image hash function
        '''
        resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (self.size + 1, self.size))
        x_diff = resized[:, 1:] > resized[:, :-1]
        # return sum of 2-power
        return np.sum(np.exp2(np.flatnonzero(x_diff)))


