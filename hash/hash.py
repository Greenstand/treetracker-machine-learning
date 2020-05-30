import cv2
import numpy as np
from scipy.fft import dctn
import os
from PIL import Image
import matplotlib.pyplot as plt

def chi_squared_distance(h1, h2, eps=1e-8):
    '''
    Returns the chi-squared distance between two histograms, as defined by Li and Jain in the
    Handbook of Face Recognition. If the two histograms aren't of the same shape, zeros are added
    to the shortest one. Thus, each histogram should share the same first bin.
    :param h1: (np.array) comparison histogram
    :param h2: (np.array) candidate histogram
    :param eps: (float) small factor to prevent numerical overflow in case of 0/0
    :return: (float) chi-squared distance between h1 and h2
    '''
    if h1.shape[0] < h2.shape[0]:
        h1 = np.concatenate([h1, np.zeros(shape=h2.shape[0] - h1.shape[0])])
    elif h2.shape[0] < h1.shape[0]:
        h2 = np.concatenate([h2, np.zeros(shape=h1.shape[0] - h2.shape[0])])
    return np.sum(np.divide(np.square(h1 - h2), (h1 + h2 + eps)))

def hamming_sort(candidate_image_hash, compared_image_hashes):
    '''
    A helper function to sort a list of image hashes (represented as integers) by Hamming distance from the
    specified candidate image hash
    :param candidate_image_hash: (int) Hash to look for
    :param compared_image_hashes: (list(int)) consisting of the image hashes to compare to
    :return: tuple(np.ndarray, np.ndarray) of sorted arguments and the corresponding hamming distance
    '''
    hammings = np.array([hamming_distance(candidate_image_hash, h) for h in compared_image_hashes])
    return np.argsort(hammings), np.sort(hammings)


def binary_array_to_int(arr):
    '''
    Helper function to take binary bitmask and return sum of 2 ** x for each index x in the flattened arr
    such that arr[x] = 1.
    :param arr: (np.ndarray) binary array
    :return: (int) operation result
    '''
    return int(np.sum(np.exp2(np.flatnonzero(arr))))


def hamming_distance(img1_hash, img2_hash):
    '''
    Count number of bits that are different between two binary hashes. In test_hash_calcs.py, we show
    this method is ~ 2.5x faster than the ImageHash version
    :param img1_hash: (int) self.hash_size bit hash of first candidate image. Expects decimal input.
    :param img2_hash: (int) self.hash_size bit hash of second candidate image. Expects decimal input.
    :return: (int) number of bits that are not equal in provided hashes
    '''
    if type(img1_hash) != int or type(img2_hash) != int:
        raise ValueError("arguments must be decimal integers, arguments provided are %s, %s" %(type(img1_hash), type(img2_hash)))
    return np.binary_repr(img1_hash ^ img2_hash).count("1")


def preprocess(images, size, interp=cv2.INTER_AREA, ksize=3):
    '''
    Short method to apply several transformations before hashing procedure.
    :param images: (list) images to transform
    :return: (list) of transformed images corresponding to passed images
    '''
    ret = []
    if type(images) is not list:
        images = [images]

    for img in images:
        col = cv2.cvtColor(np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)) , cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(col, (ksize,ksize), 0)
        resized = cv2.resize(blur, (size,size), interpolation=interp)
        ret.append(cv2.equalizeHist(resized))
    return ret

def print_groups(adj):
    """
    Helper function to print out which subsets are duplicates
    :param adj: adjacency matrix
    :return: list of ndarrays each containing indices of duplicates (size=1 indicates no duplicates)
    """
    unaccounteds = list(range(adj.shape[0]))
    clusters = []
    i = -1
    while len(unaccounteds) != 0 and i < adj.shape[0]:
        i = unaccounteds[0]
        connected = np.sort(np.flatnonzero(adj[i,:])) # get connected pixel by indexs
        clusters.append(connected)
        for val in connected:
            unaccounteds.remove(val)
    return clusters

def hash_accuracy(index, adj, hashes_by_index, hammings):
    """
    A heuristic to decide whether a hashing algorithm is working.
    We want all sorted hashes of the same image to be together and all others to be far.
    :param index: the index that is used to compute relative similarity
    :param adj: adjacency matrix to decide which images are actually duplicates
    :param hashes_by_index: the sorted indices of most similarity to teh queried index
    :param hammings: the corresponding Hamming distances to hashes_by_index
    :return:
    """
    error = 0 # the average hamming distance after the first misclassified
    correct = 0
    normalized_hd = np.array(hammings) / np.max(hammings) # allows even comparison of hashes with different ranges
    if hashes_by_index[0] != index:
        print ("Same image does not show minimal hash!")
        return np.infty

    duplicates = np.flatnonzero(adj[index,:])
    thresh = -1
    for j in range(len(hashes_by_index)):
        if hashes_by_index[j] in duplicates and thresh == -1:
            correct += 1
        elif thresh == -1 and not hashes_by_index[j] in duplicates:
            # the first occurrence of a non-adjacent image in the sorted hashes
            thresh = j
        elif thresh != -1 and hashes_by_index[j] in duplicates:
            # penalize an actual duplicate coming after the first non-adjacent(nonduplicate)
            # image
            mistake_bound = j - thresh / len(hashes_by_index) # how far off first misclassification we are
            error += mistake_bound * normalized_hd[j]
            # this promotes small hamming distance in case the set of
            # all images compared are similar to begin with.
            # consider one similar nonduplicate coming before a tougher
            # duplicate
        else:
            pass # do nothing for unrelated images

    # correct = thresh
    return correct, error



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
        # diff = maximum_filter(img, size=(filter_size, filter_size)) - minimum_filter(img, size=(filter_size,filter_size))
        histo = np.histogram(img, bins=nbins)[0]
        return  histo > np.median(histo)


    def histo_avg_hash(self, img, nbins=32, filter_size=5, thresh=None):
        '''
        Use histogram of pixel intensity values to generate non-positional hash and concatenate to
        average hash so positional and nonpositional information is accounted for.
        :param nbins: Number of bits = number of bins
        :param filter_size: size of square max/min filtering operations
        :return:
        '''
        # diff = maximum_filter(img, size=(filter_size, filter_size))
        histo = np.histogram(img, bins=nbins, range=(0, 255))[0]
        resized = cv2.resize(img, (self.size, self.size))
        if thresh is None:
            thresh = np.sum(histo) / nbins
        return np.concatenate([histo > thresh, (resized > np.mean(resized)).flatten()])

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data", "kilema_tanzania")
    hasher = ImageHasher(8) # will create 28 bit hash
    print (data_dir)
    hashes = {}
    for f, _, d in os.walk(data_dir):
        for fil in d:
            fullpath = os.path.join(f, fil)
            if os.path.splitext(fullpath)[1] == ".jpg":
                im = cv2.imread(fullpath)
                preprocess(im, size=200)
                hash = hasher.histo_avg_hash(im)
                hashes[os.path.split(fullpath)[1]] = binary_array_to_int(hash)
    thresh = 15
    keys = list(hashes.keys())
    seens = dict(zip(keys, [False for _ in keys]))
    for k in range(len(keys)):
        root = keys[k]
        args, hs = hamming_sort(hashes[root], hashes.values())
        j = hs[(hs >= 0) & (hs < thresh)].shape[0]
        if j > 1:
            # print ("Potential matches %d thresh for %s:"%(thresh, k))
            matches = np.array(list(hashes.keys()))[args][:j]
            print (root + " had %d matches"%(matches.shape[0]))
            for m in matches[1:]:
                seens[m] = True
            if not seens[root]:
                f, axarr = plt.subplots(1, len(matches), figsize=(20,20))
                plt.suptitle(root + " approx matches, thresh=%d"%thresh, fontsize=32, fontweight="bold")
                for i in range(min(len(matches), 4)):
                    fname = os.path.splitext(matches[i])[0]
                    axarr[i].imshow(Image.open(os.path.join(data_dir, fname, matches[i])))
                    axarr[i].set_title(matches[i] + " Distance:" + str(hs[i]), fontsize=24, fontweight="bold")
                plt.show()

