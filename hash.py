import cv2
import numpy as np


def hamming_distance(img1_hash, img2_hash):
    '''
    Count number of bits that are different between two binary hashes
    :param img1_hash: (int) self.hash_size bit hash of first candidate image. Expects decimal input.
    :param img2_hash: (int) self.hash_size bit hash of second candidate image. Expects decimal input.
    :return: (int) number of bits that are not equal in provided hashes
    '''
    if type(img1_hash) is not int or type(img2_hash) is not int:
        raise InvalidA
    return np.binary_repr(img1_hash ^ img2_hash).count("1")


class HashCode():
    """
    Object representation to convert between hash code binary, hex, and int
    representations and compute distance.
    See https://github.com/JohannesBuchner/imagehash for the open-source implementation that
    this class derives from.
    """

    def __init__(self, binary_array):
        self.hash = binary_array

    def __str__(self):
        pass


class ImageHasher():
    """
    Customized image hashing dervied from ImageHash open source library, allowing us to modify for
    Greenstand purposes.
    See https://github.com/JohannesBuchner/imagehash for the open-source implementation.
    """
    def __init__(self, hash_size, ):
        self.size = hash_size




    def average_hash(self, img):
        '''
        Averge hashing implementation to return hash of bits greater than average of
        pixel intensities.
        :param img: (np.ndarray) the RGB image to find a difference hash function of
        :return:
        '''
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
        return int(np.sum(np.exp2(np.flatnonzero(x_diff))))


    def perceptual_hash(self, img):
        '''
        :param img: (np.ndarray) the RGB image to find a difference hash function of
        :param hash_size: (int) the number of bits in the hash (ex. setting to 8 yields 2**8=64 bit address)
        :return: (int) 2 ** hash_size bit image hash function
        '''
        # TODO: Implement
