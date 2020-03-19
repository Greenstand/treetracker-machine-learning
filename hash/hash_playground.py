from data.data_management import GreenstandDataset
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import hsv_to_rgb
import cv2
import numpy as np
import imagehash
import os
from scipy.ndimage import maximum_filter, minimum_filter
from hash import *
from sklearn.decomposition import PCA

homepath = os.getcwd()[:-4]
print (homepath)
# to 93457 to 93481 duplicates
data = GreenstandDataset('nov11data.csv')
key1 = 93457
key2 = 93481
random_ids = np.loadtxt(os.path.join(homepath, "data/onepercentids.txt"))
randoms = [data.read_image_from_db(os.path.join(homepath, 'data/random_zeroone_percent_db/'), key=int(r)) for r in random_ids]
#Comparison test
a = data.read_image_from_db(os.path.join(homepath, 'data/duplicates_db/'), key=key1)
b = data.read_image_from_db(os.path.join(homepath, 'data/duplicates_db/'), key=key2)
dups = [data.read_image_from_db(os.path.join(homepath, 'data/1573_duplicates/'), key=int(r)) for r in range(20)]

images = [a, b] + dups + randoms
# known duplicates
adjacency_matrix = np.identity(len(images)) * 2
adjacency_matrix[0,1] = 1
adjacency_matrix[1,0] = 1
adjacency_matrix[2:len(dups) + 3, 2: len(dups) + 3] = 1

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






print (len(images), " images")
images = preprocess(images, size=100)
pics = np.array(images).reshape((6,7, 100,100))
images = np.array(images).reshape((-1, 100, 100))


hsize = 8
my_hasher = ImageHasher(hsize)
# difference hashes
imagehash_diff_hashes = [imagehash.dhash(Image.fromarray(img), hash_size=hsize) for img in images]
imagehash_avg_hashes = [imagehash.average_hash(Image.fromarray(img), hash_size=hsize) for img in images]
imagehash_dct_hashes = [imagehash.phash(Image.fromarray(img), hash_size=8, highfreq_factor=4) for img in images]
diff_hashes = [binary_array_to_int(my_hasher.difference_hash(img)) for img in images]
avg_hashes = [binary_array_to_int(my_hasher.average_hash(img)) for img in images]
dct_hashes = [binary_array_to_int(my_hasher.dct_hash(img)[0]) for img in images]
dct_matrices = [my_hasher.dct_hash(img)[1] for img in images] # DCT matrices not hashes
histo_hashes = [binary_array_to_int(my_hasher.histo_hash(img, 64, 5)) for img in images]
histo_hybrid_hashes = [binary_array_to_int(my_hasher.histo_avg_hash(img, filter_size=3,thresh=None)) for img in images]



print ("IMAGE 0")
print ("AVERAGE HASH")
idxs, hams = hamming_sort(avg_hashes[0], avg_hashes)
print (idxs)
print (hams)
correct, error = hash_accuracy(0, adjacency_matrix, idxs, hams)
print ("Correct: ", correct)
print ("Error: ", error)


f, axarr = plt.subplots(2,2)
plt.suptitle("Top 4 matches of average hashing image 0")
axarr[0, 0].imshow(images[idxs[0]])
axarr[1, 0].imshow(images[idxs[1]])
axarr[0, 1].imshow(images[idxs[2]])
axarr[1, 1].imshow(images[idxs[3]])
plt.show()


print ("DIFFERENCE HASH")
idxs, hams = hamming_sort(diff_hashes[0], diff_hashes)
print (idxs)
print (hams)
correct, error = hash_accuracy(0, adjacency_matrix, idxs, hams)
print ("Correct: ", correct)
print ("Error: ", error)

f, axarr = plt.subplots(2,2)
plt.suptitle("Top 4 matches of difference hashing image 0")
axarr[0, 0].imshow(images[idxs[0]])
axarr[1, 0].imshow(images[idxs[1]])
axarr[0, 1].imshow(images[idxs[2]])
axarr[1, 1].imshow(images[idxs[3]])
plt.show()


print ("PROPOSED HISTOGRAM HASH image 0")
idxs, hams = hamming_sort(histo_hashes[0], histo_hashes)
print (idxs)
print (hams)
correct, error = hash_accuracy(0, adjacency_matrix, idxs, hams)
print ("Correct: ", correct)
print ("Error: ", error)


f, axarr = plt.subplots(2,2)
plt.suptitle("Top 4 matches of histogram hashing image 0")
axarr[0, 0].imshow(images[idxs[0]])
axarr[1, 0].imshow(images[idxs[1]])
axarr[0, 1].imshow(images[idxs[2]])
axarr[1, 1].imshow(images[idxs[3]])
plt.show()


print ("PROPOSED MODIFIED HISTOGRAM HASH image 0")
idxs, hams = hamming_sort(histo_hybrid_hashes[0], histo_hybrid_hashes)
print (idxs)
print (hams)
correct, error = hash_accuracy(0, adjacency_matrix, idxs, hams)
print ("Correct: ", correct)
print ("Error: ", error)




f, axarr = plt.subplots(2,2)
plt.suptitle("Top 4 matches of modified histogram hashing image 0")
axarr[0, 0].imshow(images[idxs[0]])
axarr[1, 0].imshow(images[idxs[1]])
axarr[0, 1].imshow(images[idxs[2]])
axarr[1, 1].imshow(images[idxs[3]])
plt.show()




print ("IMAGE 3")
print ("AVERAGE HASH")
idxs, hams = hamming_sort(avg_hashes[3], avg_hashes)
print (idxs)
print (hams)
correct, error = hash_accuracy(3, adjacency_matrix, idxs, hams)
print ("Correct: ", correct)
print ("Error: ", error)


f, axarr = plt.subplots(2,2)
plt.suptitle("Top 4 matches of average hashing image 3")
axarr[0, 0].imshow(images[idxs[0]])
axarr[1, 0].imshow(images[idxs[1]])
axarr[0, 1].imshow(images[idxs[2]])
axarr[1, 1].imshow(images[idxs[3]])
plt.show()


print ("DIFFERENCE HASH")
idxs, hams = hamming_sort(diff_hashes[3], diff_hashes)
print (idxs)
print (hams)
correct, error = hash_accuracy(3, adjacency_matrix, idxs, hams)
print ("Correct: ", correct)
print ("Error: ", error)


f, axarr = plt.subplots(2,2)
plt.suptitle("Top 4 matches of difference hashing image 3")
axarr[0, 0].imshow(images[idxs[0]])
axarr[1, 0].imshow(images[idxs[1]])
axarr[0, 1].imshow(images[idxs[2]])
axarr[1, 1].imshow(images[idxs[3]])
plt.show()


print ("PROPOSED HISTOGRAM HASH image 3")
idxs, hams = hamming_sort(histo_hashes[3], histo_hashes)
print (idxs)
print (hams)
correct, error = hash_accuracy(3, adjacency_matrix, idxs, hams)
print ("Correct: ", correct)
print ("Error: ", error)


f, axarr = plt.subplots(2,2)
plt.suptitle("Top 4 matches of histogram hashing image 3")
axarr[0, 0].imshow(images[idxs[0]])
axarr[1, 0].imshow(images[idxs[1]])
axarr[0, 1].imshow(images[idxs[2]])
axarr[1, 1].imshow(images[idxs[3]])
plt.show()


print ("PROPOSED MODIFIED HISTOGRAM HASH image 3")
idxs, hams = hamming_sort(histo_hybrid_hashes[3], histo_hybrid_hashes)
print (idxs)
print (hams)
correct, error = hash_accuracy(3, adjacency_matrix, idxs, hams)
print ("Correct: ", correct)
print ("Error: ", error)

f, axarr = plt.subplots(2,2)
plt.suptitle("Top 4 matches of modified histogram hashing image 3")
axarr[0, 0].imshow(images[idxs[0]])
axarr[1, 0].imshow(images[idxs[1]])
axarr[0, 1].imshow(images[idxs[2]])
axarr[1, 1].imshow(images[idxs[3]])
plt.show()

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

print (print_groups(adjacency_matrix))


