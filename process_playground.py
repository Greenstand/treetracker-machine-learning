from data_management import GreenstandDataset
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import hsv_to_rgb
import cv2
import numpy as np
import imagehash
import os
from hash import ImageHasher, hamming_distance


print ("OpenCV version: ", cv2.__version__)
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

size = 120
images = [cv2.resize(img,(size,size)) for img in images]

# if doing hash comparison is slow consider switching to greyscale and
# making images smaller
fig, axarr = plt.subplots(2,2, figsize=(8,8))
plt.suptitle("Sample images A=%d, B=%d, C=%d, D=%d"%(key1,key2,key3,key4))
axarr[0,0].imshow(a)
axarr[0,0].set_title("A")
axarr[1,0].imshow(b)
axarr[1,0].set_title("B")
axarr[0,1].imshow(c)
axarr[0,1].set_title("C")
axarr[1,1].imshow(d)
axarr[1,1].set_title("D")
plt.show()





hsize = 9
my_hasher = ImageHasher(hsize)
hashes = [imagehash.dhash(Image.fromarray(img), hash_size=hsize) for img in images]
my_hashes = [my_hasher.difference_hash(img) for img in images]

print ("IMAGEHASH")
for h in hashes:
    print ("Open Source Hamming")
    print (hashes[0] - h) # difference between image A and this image
    # print (int(str(hashes[0]),16))
    # print (int(str(h),16))
    print ("Custom Hamming")
    print (np.binary_repr(np.bitwise_xor(int(str(hashes[0]),16), int(str(h),16))).count("1"))
    print (hamming_distance(int(str(h),16), int(str(hashes[0]),16)))

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


a_hash = imagehash.dhash(Image.fromarray(a), hash_size=hsize)
b_hash = imagehash.dhash(Image.fromarray(b), hash_size=hsize)
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