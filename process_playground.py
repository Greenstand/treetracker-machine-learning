from data_management import GreenstandDataset
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import cv2
import numpy as np

print (cv2.__version__)

# to 93457 to 93481 duplicates
data = GreenstandDataset('nov11data.csv')
key1 = 93457
key2 = 93481

a = data.read_image_from_db('duplicates_db/', key=key1)
b = data.read_image_from_db('duplicates_db/', key=key2)
plt.imshow(b)
plt.show()
xsize, ysize = 200,200
a = cv2.resize(a,dsize=(xsize, ysize), interpolation=cv2.INTER_NEAREST)
b = cv2.resize(b,dsize=(xsize, ysize), interpolation=cv2.INTER_NEAREST)

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


orb = cv2.ORB()
print ("ORB Initialized")
kp, des = orb.detectAndCompute(a, None)

print ("HERE")
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(a,kp,color=(0,255,0), flags=0)
plt.imshow(img2)
plt.show()

plt.imshow(a)
plt.show()