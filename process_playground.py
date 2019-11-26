from data_management import GreenstandDataset
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import hsv_to_rgb
import cv2
import numpy as np
import imagehash

print (cv2.__version__)

# to 93457 to 93481 duplicates
data = GreenstandDataset('nov11data.csv')
key1 = 93457
key2 = 93481

a = data.read_image_from_db('duplicates_db/', key=key1)
b = data.read_image_from_db('duplicates_db/', key=key2)
plt.imshow(b)
plt.show()

# plt.show()
#
# fig, axarr = plt.subplots(1,2)
# axarr[0].imshow(a)
# axarr[0].set_title(str(key1))
# axarr[1].imshow(b)
# axarr[0].set_title(str(key2))
# plt.show()

frame_LAB = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.uint8)
# HSV Color values are from internet. Green is [120,180]. Saturation is
# amount of gray (grayer near 0, 1 is near primary color).
# Value is brightness

#
# plt.hist(frame_LAB[:,:,0].flatten(),bins=25)
# plt.title('L')
# plt.show()
#
#
# plt.hist(frame_LAB[:,:,1].flatten(),bins=25)
# plt.title('A')
# plt.show()
#
#
# plt.hist(frame_LAB[:,:,2].flatten(),bins=25)
# plt.title('B')
# plt.show()

frame_threshold = cv2.inRange(frame_LAB,(0,30,150),(255,125,255)).astype(np.uint8)
# plt.imshow(frame_threshold)
# plt.show()
#
# res = cv2.bitwise_and(frame_LAB,frame_LAB,mask=frame_threshold)
# plt.imshow(cv2.cvtColor(res,cv2.COLOR_LAB2RGB))
# plt.show()

def difference_hash(img, hash_size):
    '''
    Credit to https://www.pyimagesearch.com/2017/11/27/image-hashing-opencv-python/
    for implementation tutorial.

    :param img: (np.ndarray) the RGB image to find a difference hash function of
    :param hash_size: (int) the number of bits in the hash (ex. setting to 8 yields 2**8=64 bit address)
    :return: (int) 2 ** hash_size bit image hash function
    '''
    resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (hash_size + 1, hash_size))
    x_diff = resized[:, 1:] > resized[:, :-1]
    # return sum of 2-power
    return np.sum(np.exp2(np.flatnonzero(x_diff)))


def phash(img, resize_dim):


    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, resize_dim)
    # TODO: Finish method
    return


a_hash = imagehash.dhash(Image.fromarray(a), hash_size=9)
b_hash = imagehash.dhash(Image.fromarray(b), hash_size=9)
print (a_hash)

print (a_hash - b_hash)
# print (np.count_nonzero((a_hash)!= (b_hash)))
#
# print (np.count_nonzero([0,0,1] != [0,0,0]))

# orb = cv2.ORB()
# print ("ORB Initialized")
# kp, des = orb.detectAndCompute(a, None)

print ("HERE")
# draw only keypoints location,not size and orientation
# img2 = cv2.drawKeypoints(a, kp, color=(0,255,0), flags=0)
# plt.imshow(img2)
# plt.show()
#
# plt.imshow(a)
# plt.show()