import cv2
import numpy as np
from scipy.fft import dctn
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter

def hamming_sort(candidate_image_hash, compared_image_hashes):
    '''
    A helper function to sort a list of image hashes (represented as integers) by Hamming distance from the
    specified candidate image hash
    :param candidate_image_hash: Hash to look for
    :param compared_image_hashes: List of integers consisting of the image hashes to compare to
    :return: tuple(np.ndarray, np.ndarray) of sorted arguments and the corresponding hamming distance
    '''
    hammings = [hamming_distance(candidate_image_hash, h) for h in compared_image_hashes]
    idxs = np.argsort(hammings)
    return idxs, np.sort(hammings)


def binary_array_to_int(arr):
    '''
    Helper function to take binary bitmask and return sum of 2 ** x for each index x in the flattened arr
    such that arr[x] = 1.
    :param arr: binary array
    :return: int
    '''
    return int(np.sum(np.exp2(np.flatnonzero(arr))))


def hamming_distance(img1_hash, img2_hash):
    '''
    Count number of bits that are different between two binary hashes. In test.py, we show
    this method is ~ 2.5x faster than the ImageHash version
    :param img1_hash: (int) self.hash_size bit hash of first candidate image. Expects decimal input.
    :param img2_hash: (int) self.hash_size bit hash of second candidate image. Expects decimal input.
    :return: (int) number of bits that are not equal in provided hashes
    '''
    if type(img1_hash) != int or type(img2_hash) != int:
        raise ValueError("arguments must be decimal integers, arguments provided are %s, %s" %(type(img1_hash), type(img2_hash)))
    return np.binary_repr(img1_hash ^ img2_hash).count("1")


def preprocess(images, size, interp=cv2.INTER_AREA):
    '''
    Short method to apply several transformations before hashing procedure.
    :param images: (list) images to transform
    :return: (list) of transformed images corresponding to passed images
    '''
    return [cv2.resize(
            cv2.cvtColor(
            np.uint8(
            cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)) ,cv2.COLOR_BGR2GRAY), (size,size), interpolation=interp)for img in images]  # or convert) for img in images]



class ImageHasher():
    """
    Customized image hashing derived from ImageHash open source library, allowing us to modify for
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
        resized = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        # following one liner avoids unnecessary variable assignment. The average hash takes the mean grayscal pixel intensity
        # and generates hash from (flattened) indices greater than the mean
        return resized > np.mean(resized)

    def difference_hash(self, img):
        '''
        Credit to https://www.pyimagesearch.com/2017/11/27/image-hashing-opencv-python/
        for implementation tutorial.

        :param img: (np.ndarray) the RGB image to find a difference hash function from
        :param hash_size: (int) the number of bits in the hash (ex. setting to 8 yields 2**8=64 bit address)
        :return: (int) 2 ** hash_size bit image hash function returned as int (not hex or binary)
        '''
        resized = cv2.resize(img, (self.size + 1, self.size), interpolation=cv2.INTER_AREA)
        # following one liner avoids unnecessary variable assignment. The difference hash takes the left-right difference
        # of grayscale pixel intensities and generates a hash from the indices
        return resized[:, 1:] > resized[:, :-1]


    def dct_hash(self, img, blur_dim=7):
        '''
        :param img: (np.ndarray) the RGB image to find a difference hash function of
        :param hash_size: (int) the number of bits in the hash (ex. setting to 8 yields 2**8=64 bit address)
        :param blur_dim(int) size of square mean-filter
        :return: (tuple(np.ndarray, np.ndarray)) 2 ** hash_size bit image hash binary array and DCT matrix output
        '''
        if self.size != 8:
            print("Original DCT was 8 x 8 so size parameter of ImageHash object may be off")
        dct_matx = dctn(cv2.blur(img, (blur_dim, blur_dim)), type=2,norm="ortho")
        tr_matx = dct_matx[:self.size,:self.size].flatten()# original algorithm selects top 8x8=64
        return tr_matx > np.median(tr_matx), dct_matx



    def marr_hildreth_hash(self, img, std=3, ksize=3):
        '''
        TODO: Edge-detector based hash
        :param img:
        :param std:
        :param ksize:
        :return:
        '''
        resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = cv2.Laplacian(cv2.GaussianBlur(img, (std,std), 0), kernel_size=ksize)
        # TODO: Finish marr hildreth edge-based hash if others don't work


    def histo_hash(self, img, nbins=64, filter_size=5):
        '''
        Use histogram of pixel intensity values to generate non-positional hash
        :param nbins: Number of bits = number of bins
        :param filter_size: size of square max/min filtering operations
        :return: (np.ndarray) binary array
        '''
        diff = maximum_filter(img, size=(filter_size, filter_size)) - minimum_filter(img, size=(filter_size,filter_size))
        histo = np.histogram(diff, bins=nbins)[0]
        # histo = np.log10(histo + 1)
        return  histo > np.median(histo)


    def histo_avg_hash(self, img, nbins=32, filter_size=5, thresh=None):
        '''
        Use histogram of pixel intensity values to generate non-positional hash and concatenate to
        average hash so positional and nonpositional information is accounted for.
        :param nbins: Number of bits = number of bins
        :param filter_size: size of square max/min filtering operations
        :return:
        '''
        diff = maximum_filter(img, size=(filter_size, filter_size))
        histo = np.histogram(diff, bins=nbins, range=(0, 255))[0]
        resized = cv2.resize(img, (8,8))
        if thresh is None:
            thresh = np.sum(histo) / nbins
        return np.concatenate([histo > thresh, (resized > np.mean(resized)).flatten()])
