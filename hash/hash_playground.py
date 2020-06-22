from data.data_management import GreenstandDataset
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from matplotlib.colors import hsv_to_rgb
import cv2
import numpy as np
import imagehash
import os
from hash import *
from sklearn.cluster import SpectralClustering


# This file is meant as a place to try out ideas on the fly. None of this is
# ready to be put into production of any kind.

homepath = os.getcwd()[:-4]
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
clustering = SpectralClustering(n_clusters=2,
                                assign_labels="discretize",
                                random_state=0,
                                eigen_solver="arpack")

# known duplicates
adjacency_matrix = np.identity(len(images)) * 2
adjacency_matrix[0,1] = 1
adjacency_matrix[1,0] = 1
adjacency_matrix[2:len(dups) + 3, 2: len(dups) + 3] = 1

def show_images(dims, title,imgs, idxs):
    f, axarr = plt.subplots(dims[0], dims[1], figsize=(10,10))
    if idxs is None:
        idxs = list(range(len(imgs)))
    assert len(idxs) == len(imgs)
    c = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            axarr[i,j].imshow(imgs[idxs[c]])
            c += 1
    plt.suptitle(title)
    plt.show()

def hash_algorithm_display(hasher, image_idx, images, hash_algo, verbose=False):
    if verbose:
        print ("IMAGE %d" %image_idx)
    img = images[image_idx]
    hash_algo = hash_algo.upper()
    if hash_algo == "DIFF":
        hashes = [binary_array_to_int(hasher.difference_hash(img)) for img in images]
    elif hash_algo == "AVG":
        hashes = [binary_array_to_int(hasher.average_hash(img)) for img in images]
    elif hash_algo == "DCT":
        hashes = [binary_array_to_int(hasher.dct_hash(img)[0]) for img in images]
        dct_matrices = [hasher.dct_hash(img)[1] for img in images]  # DCT matrices not hashes
    elif hash_algo == "HISTO":
        hashes = [binary_array_to_int(hasher.histo_hash(img, 64, 5)) for img in images]
    elif hash_algo == "MOD HISTO":
        hashes = [binary_array_to_int(hasher.histo_avg_hash(img, filter_size=3, thresh=None)) for img in
                           images]
    else:
        hashes = None
        raise ValueError("Unknown hash algorithm argument")

    idxs, hams = hamming_sort(hashes[image_idx], hashes)
    if verbose:
        print(idxs)
        print(hams)
    correct, error = hash_accuracy(0, adjacency_matrix, idxs, hams)
    if verbose:
        print("Correct: ", correct)
        print("Error: ", error)
    t = "Top 4 matches of difference hashing image 0"
    show_images((2, 2), t, images, idxs)
    return hashes, idxs, hams


# 42 images in total
#
# plt.title("Original")
# plt.imshow(images[0])
# plt.show()
#
#
# plt.title("Resize")
# plt.imshow(cv2.resize(images[0],(rsz, rsz)))
# plt.show()
#
# plt.title("Gauss")
# plt.imshow(cv2.GaussianBlur(images[0], (ks, ks),0))
# plt.show()
#
# plt.title("Gauss Resize")
# plt.imshow(cv2.resize(cv2.GaussianBlur(images[0], (ks, ks),0), (rsz,rsz)))
# plt.show()



rsz = 200
ks = 5


for skip in [4]:
    resizes = [cv2.resize(i, (rsz // skip, rsz // skip)) for i in images]
    blurs = [cv2.GaussianBlur(i, (ks, ks), 0) for i in resizes]
    hsvs = [rgb_to_hsv(a) for a in blurs]
    orig = None
    f, axarr = plt.subplots (6,7,figsize=(20,20))
    pics = np.array(resizes).reshape((6,7, rsz // skip, rsz // skip, 3))
    for i in range(pics.shape[0]):
        for j in range(pics.shape[1]):
            im = pics[i, j]
            # print (d)
            axarr[i,j].imshow(im)
    plt.show()

for skip in [4]:
    resizes = [cv2.resize(i, (rsz // skip, rsz // skip)) for i in images]
    blurs = [cv2.GaussianBlur(i, (ks, ks), 0) for i in resizes]
    hsvs = [rgb_to_hsv(a) for a in blurs]
    orig = None
    f, axarr = plt.subplots (6,7,figsize=(20,20))
    pics = np.array(hsvs).reshape((6,7, rsz // skip, rsz // skip, 3))
    for i in range(pics.shape[0]):
        for j in range(pics.shape[1]):
            im = pics[i, j]
            low_val_thresh = np.percentile(pics[i,j, :, :, 2], 10, interpolation='midpoint')
            high_val_thresh = np.percentile(pics[i,j, :, :, 2], 90, interpolation='midpoint')
            val_mask = np.where((pics[i, j, :, :, 2] < low_val_thresh) | (pics[i, j, :, :, 2] > high_val_thresh))
            low_hue_thresh = np.percentile(pics[i,j, :, :, 0], 20, interpolation='midpoint')
            high_hue_thresh = np.percentile(pics[i,j, :, :, 0], 80, interpolation='midpoint')
            hue_mask = np.where((pics[i, j, :, :, 0] < low_hue_thresh) | (pics[i, j, :, :, 0] > high_hue_thresh))
            im[val_mask] = 0
            im[hue_mask] = 0
            im [im < 0] = 0
            if im[:,:, 0].any() < 0:
                print (im[:, :, 0])
            f = np.bincount((im[:, :, 0].ravel() * 180).astype(np.uint8))
            binstart = np.min((im[:, :, 0].ravel() * 180).astype(np.uint8))
            assert binstart == 0 # they should all start at 0
            if i == 0 and j == 0:
                orig = f
            d = np.round(chi_squared_distance(orig, f), 2)
            # print (d)
            axarr[i, j].bar(list(range(binstart, binstart + len(f))), f)
            axarr[i, j].set_title("X^2 Distance from Orig: %d"%d)
    plt.show()


for skip in [4]:
    resizes = [cv2.resize(i, (rsz // skip, rsz // skip)) for i in images]
    blurs = [cv2.GaussianBlur(i, (ks, ks), 0) for i in resizes]
    hsvs = [rgb_to_hsv(a) for a in blurs]
    orig = None
    plt.suptitle("Clustering")
    f, axarr = plt.subplots (6,7,figsize=(20,20))
    pics = np.array(hsvs).reshape((6,7, rsz // skip, rsz // skip, 3))
    for i in range(pics.shape[0]):
        for j in range(pics.shape[1]):
            im = pics[i, j]
            low_val_thresh = np.percentile(pics[i,j, :, :, 2], 10, interpolation='midpoint')
            high_val_thresh = np.percentile(pics[i,j, :, :, 2], 90, interpolation='midpoint')
            val_mask = np.where((pics[i, j, :, :, 2] < low_val_thresh) | (pics[i, j, :, :, 2] > high_val_thresh))
            low_hue_thresh = np.percentile(pics[i,j, :, :, 0], 20, interpolation='midpoint')
            high_hue_thresh = np.percentile(pics[i,j, :, :, 0], 80, interpolation='midpoint')
            hue_mask = np.where((pics[i, j, :, :, 0] < low_hue_thresh) | (pics[i, j, :, :, 0] > high_hue_thresh))
            im[val_mask] = 0
            im[hue_mask] = 0
            im [im < 0] = 0
            xx, yy = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
            xx = xx[..., np.newaxis]
            yy = yy[..., np.newaxis]
            stack = np.concatenate([im, xx, yy], axis=2)
            clusters = clustering.fit_predict(stack.reshape(-1, stack.shape[-1]))
            print ("clustering pic %d"%(i * pics.shape[1] + j))
            cluster_im = clusters.reshape(stack.shape[0], stack.shape[1])
            axarr[i, j].imshow(cluster_im)

    plt.show()
# images = preprocess(images, size=rsz,ksize=ks)
# pics = np.array(images).reshape((6,7, rsz, rsz))
# f, axarr = plt.subplots (6,7,figsize=(20,20))
# for i in range(pics.shape[0]):
#     for j in range(pics.shape[1]):
#         axarr[i,j].imshow(pics[i, j])
# plt.show()
# images = np.array(images).reshape((-1, rsz, rsz))
#
#
# hsize = 8
# my_hasher = ImageHasher(hsize)
# # hash_algorithm_display(my_hasher, 0, images, "AVG", True)

