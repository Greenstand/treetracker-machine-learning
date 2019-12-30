from data_management import GreenstandDataset
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import hsv_to_rgb
import cv2
import numpy as np
import imagehash
import os
from hash import ImageHasher, hamming_distance, preprocess


# to 93457 to 93481 duplicates
data = GreenstandDataset('nov11data.csv')
key1 = 93457
key2 = 93481
key3 = int(7.718300000000000000e+04)
key4 = int(1.654190000000000000e+05)

# Comparison test
a = data.read_image_from_db('duplicates_db/', key=key1)
b = data.read_image_from_db('duplicates_db/', key=key2)
c = data.read_image_from_db(os.path.join(os.getcwd(), 'random_zeroone_percent_db/'), key=key3)
d = data.read_image_from_db('random_zeroone_percent_db/', key=key4)
images = [a,b,c,d]
images = preprocess(images)
size = 120
images = [cv2.resize(img,(size,size), interpolation=cv2.INTER_AREA) for img in images]

# if doing hash comparison is slow consider switching to greyscale and
# making images smaller
fig, axarr = plt.subplots(2,2, figsize=(8,8))
plt.suptitle("Sample images A=%d, B=%d, C=%d, D=%d"%(key1,key2,key3,key4))
axarr[0,0].imshow(images[0], cmap='gray')
axarr[0,0].set_title("A")
axarr[1,0].imshow(images[1], cmap='gray')
axarr[1,0].set_title("B")
axarr[0,1].imshow(images[2], cmap='gray')
axarr[0,1].set_title("C")
axarr[1,1].imshow(images[3], cmap='gray')
axarr[1,1].set_title("D")
plt.savefig("playground_images/bw_histo_image_samples.jpg")
plt.show()
plt.figure()

hsize = 8
my_hasher = ImageHasher(hsize)
# difference hashes
imagehash_diff_hashes = [imagehash.dhash(Image.fromarray(img), hash_size=hsize) for img in images]
imagehash_avg_hashes = [imagehash.average_hash(Image.fromarray(img), hash_size=hsize) for img in images]
imagehash_dct_hashes = [imagehash.phash(Image.fromarray(img), hash_size=8, highfreq_factor=4) for img in images]
diff_hashes = [my_hasher.difference_hash(img) for img in images]
avg_hashes = [my_hasher.average_hash(img) for img in images]
dct_hashes = [my_hasher.dct_hash(img) for img in images]

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


# print (a_hash)
#
# print (a_hash - b_hash)
# print (np.count_nonzero((a_hash)!= (b_hash)))
#
# print (np.count_nonzero([0,0,1] != [0,0,0]))

# orb = cv2.ORB()
# print ("ORB Initialized")
# kp, des = orb.detectAndCompute(a, None)

# print ("HERE")
# draw only keypoints location,not size and orientation
# img2 = cv2.drawKeypoints(a, kp, color=(0,255,0), flags=0)
# plt.imshow(img2)
# plt.show()
#
# plt.imshow(a)
# plt.show()