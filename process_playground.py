from data_management import GreenstandDataset
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import hsv_to_rgb
import cv2
import numpy as np
import imagehash
import os
from scipy.ndimage import maximum_filter, minimum_filter
from hash import *

# to 93457 to 93481 duplicates
data = GreenstandDataset('nov11data.csv')
key1 = 93457
key2 = 93481
random_ids = np.loadtxt("onepercentids.txt")
randoms = [data.read_image_from_db('random_zeroone_percent_db/', key=int(r)) for r in random_ids]

# Comparison test
a = data.read_image_from_db('duplicates_db/', key=key1)
b = data.read_image_from_db('duplicates_db/', key=key2)
images = [a, b] + randoms
images = preprocess(images, size=100)

# if doing hash comparison is slow consider switching to greyscale and
# making images smaller
# fig, axarr = plt.subplots(2,2, figsize=(8,8))
# plt.suptitle("Sample images A=%d, B=%d, C=%d, D=%d"%(key1,key2,random_ids[0],random_ids[1]))
# axarr[0,0].imshow(images[0], cmap='gray')
# axarr[0,0].set_title("A")
# axarr[1,0].imshow(images[1], cmap='gray')
# axarr[1,0].set_title("B")
# axarr[0,1].imshow(images[2], cmap='gray')
# axarr[0,1].set_title("C")
# axarr[1,1].imshow(images[3], cmap='gray')
# axarr[1,1].set_title("D")
# plt.savefig("playground_images/bw_histo_image_samples.jpg")
# plt.show()
# plt.figure()

hsize = 8
my_hasher = ImageHasher(hsize)
# difference hashes
imagehash_diff_hashes = [imagehash.dhash(Image.fromarray(img), hash_size=hsize) for img in images]
imagehash_avg_hashes = [imagehash.average_hash(Image.fromarray(img), hash_size=hsize) for img in images]
imagehash_dct_hashes = [imagehash.phash(Image.fromarray(img), hash_size=8, highfreq_factor=4) for img in images]
diff_hashes = [binary_array_to_int(my_hasher.difference_hash(img)) for img in images]
avg_hashes = [binary_array_to_int(my_hasher.average_hash(img)) for img in images]
dct_hashes = [binary_array_to_int(my_hasher.dct_hash(img)[0]) for img in images]
dcts = [my_hasher.dct_hash(img)[1] for img in images]

# n = 8
# f, axarr = plt.subplots(2,2)
# plt.suptitle("Sample DCTS A=%d, B=%d, C=%d, D=%d"%(key1,key2,key3,key4))
# axarr[0,0].imshow(dcts[0][:n,:n])
# axarr[0,0].set_title("A")
# axarr[1,0].imshow(dcts[1][:n,:n])
# axarr[1,0].set_title("B")
# axarr[0,1].imshow(dcts[2][:n,:n])
# axarr[0,1].set_title("C")
# axarr[1,1].imshow(dcts[3][:n,:n])
# axarr[1,1].set_title("D")
# plt.savefig("playground_images/dcts.jpg")
# plt.show()
# plt.figure()

print ("Comparison to ImageHash library")
for i in range(len(images)):
    print (i)
    print ("-" * 50)
    print ("AVERAGES")
    print (str(imagehash_avg_hashes[i]) == hex(avg_hashes[i]))
    print (str(imagehash_avg_hashes[i]))
    print (hex(avg_hashes[i]))
    print ("DIFF")
    print (str(imagehash_diff_hashes[i]) == hex(diff_hashes[i]))
    print (str(imagehash_diff_hashes[i]))
    print (hex(diff_hashes[i]))
    print ("DCT")
    print (str(imagehash_dct_hashes[i]) == hex(dct_hashes[i]))
    print (str(imagehash_dct_hashes[i]))
    print (hex(dct_hashes[i]))


print ("DIFFERENCE HASHES")
for j in range(len(diff_hashes)):
    print (hamming_distance(diff_hashes[0], diff_hashes[j]))
    print (imagehash_diff_hashes[0] - imagehash_diff_hashes[j])
print ("AVERAGE  HASHES")
for j in range(len(avg_hashes)):
    print (hamming_distance(avg_hashes[0], avg_hashes[j]))
    print (imagehash_avg_hashes[0] - imagehash_avg_hashes[j])


print ("DCT HASHES")
for j in range(len(dct_hashes)):
    print (hamming_distance(dct_hashes[0], dct_hashes[j]))
    print (imagehash_dct_hashes[0] - imagehash_dct_hashes[j])



a_fake = a.copy()
a_fake[0,:] = 0
# print (a[0,0])
# print (a_fake[0,0])
a_hash = imagehash.dhash(Image.fromarray(a), hash_size=hsize)
a_fake_hash = imagehash.dhash(Image.fromarray(a_fake),hash_size=hsize)
b_hash = imagehash.dhash(Image.fromarray(b), hash_size=hsize)

print ("A vs A_fake vs B")
print (a_hash)
print (a_fake_hash)
print (b_hash)
print (a_hash - b_hash)

def ext_viz(img, filter_size, nbins=8):
    max_filtered = maximum_filter(img, size=(filter_size, filter_size))
    min_filtered = minimum_filter(img, size=(filter_size,filter_size))
    f, axarr = plt.subplots(2,3)
    axarr[0,0].imshow(max_filtered)
    axarr[0,1].imshow(min_filtered)
    axarr[0,2].imshow(max_filtered - min_filtered)
    axarr[1,0].plot(np.histogram(max_filtered, bins=nbins)[0])
    axarr[1,0].grid()
    axarr[1,1].plot(np.histogram(min_filtered, bins=nbins)[0])
    axarr[1,1].grid()
    axarr[1,2].plot(np.histogram(max_filtered - min_filtered, bins=8)[0])
    axarr[1,2].grid()
    plt.show()
    return max_filtered, min_filtered
#
# max0, min0 = ext_viz(images[0], 8, nbins=8)
# max1, min1 = ext_viz(images[1], 8, nbins=8)
# max2, min2 = ext_viz(images[2], 8)
# max3, min3 = ext_viz(images[3], 8)



histo_hashes = [binary_array_to_int(my_hasher.histo_hash(img, 64, 5)) for img in images]
histo_hybrid_hashes = [binary_array_to_int(my_hasher.histo_avg_hash(img)) for img in images]




print ("AVERAGE HASH")
idxs, hams = hamming_sort(avg_hashes[0], avg_hashes)
print (idxs)
print (hams)

f, axarr = plt.subplots(2,2)
plt.suptitle("Top 4 matches of average hashing")
axarr[0, 0].imshow(images[idxs[0]])
axarr[1, 0].imshow(images[idxs[1]])
axarr[0, 1].imshow(images[idxs[2]])
axarr[1, 1].imshow(images[idxs[3]])
plt.show()


print ("DIFFERENCE HASH")
idxs, hams = hamming_sort(diff_hashes[0], diff_hashes)
print (idxs)
print (hams)

f, axarr = plt.subplots(2,2)
plt.suptitle("Top 4 matches of difference hashing")
axarr[0, 0].imshow(images[idxs[0]])
axarr[1, 0].imshow(images[idxs[1]])
axarr[0, 1].imshow(images[idxs[2]])
axarr[1, 1].imshow(images[idxs[3]])
plt.show()


print ("PROPOSED HISTOGRAM HASH")
idxs, hams = hamming_sort(histo_hashes[0], histo_hashes)
print (idxs)
print (hams)

f, axarr = plt.subplots(2,2)
plt.suptitle("Top 4 matches of histogram hashing")
axarr[0, 0].imshow(images[idxs[0]])
axarr[1, 0].imshow(images[idxs[1]])
axarr[0, 1].imshow(images[idxs[2]])
axarr[1, 1].imshow(images[idxs[3]])
plt.show()


print ("PROPOSED MODIFIED HISTOGRAM HASH")
idxs, hams = hamming_sort(histo_hybrid_hashes[0], histo_hybrid_hashes)
print (idxs)
print (hams)



f, axarr = plt.subplots(2,2)
plt.suptitle("Top 4 matches of modified histogram hashing")
axarr[0, 0].imshow(images[idxs[0]])
axarr[1, 0].imshow(images[idxs[1]])
axarr[0, 1].imshow(images[idxs[2]])
axarr[1, 1].imshow(images[idxs[3]])
plt.show()