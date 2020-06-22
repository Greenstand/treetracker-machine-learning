import unittest
import imagehash
import numpy as np
import time
import imagehash
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from hash.hash import hamming_distance, binary_array_to_int, preprocess, ImageHasher

class CorrectHamming(unittest.TestCase):
    verified = ( ((1, 1),0), # same two inputs
                 ((0, 150), 4), # zero input
                 ((0x0d5585f2ed36cc3758b4d, 0x0d5585f2ed36cc3758b4d), 0),
                 ((1, 1), 0), # TODO: add tests
                 ((1, 1), 0),
                 ((1, 1), 0),
                 )

    timed_example = (4901245960912984, 703295019458591)

    def test_hamming(self):
        for hashes, expected in self.verified:
            self.assertEqual(hamming_distance(hashes[0],hashes[1]), expected)

    def test_time(self):
        start = time.perf_counter()
        for j in range(10000):
            hamming_distance(self.timed_example[0], self.timed_example[1])
        print(((time.perf_counter() - start) / 10000) * 10**6, " ms average custom hamming time")

    def test_imagehash_time(self):
        timea = imagehash.ImageHash(np.array(list(np.binary_repr(self.timed_example[0])))[:50])
        timeb = imagehash.ImageHash(np.array(list(np.binary_repr(self.timed_example[1])))[:50])
        start = time.perf_counter()
        for j in range(10000):
            timea - timeb
        print(((time.perf_counter() - start) / 10000) * 10**6, " ms average open source imagehash hamming time")

    def test_against_imagehash(self):
        hsize = 8
        images = [np.random.randint(0,255,(100,100)) for _ in range(10)]
        my_hasher = ImageHasher(hsize)
        imagehash_diff_hashes = [imagehash.dhash(Image.fromarray(img), hash_size=hsize) for img in images]
        imagehash_avg_hashes = [imagehash.average_hash(Image.fromarray(img), hash_size=hsize) for img in images]
        imagehash_dct_hashes = [imagehash.phash(Image.fromarray(img), hash_size=8, highfreq_factor=4) for img in images]
        diff_hashes = [binary_array_to_int(my_hasher.difference_hash(img)) for img in images]
        avg_hashes = [binary_array_to_int(my_hasher.average_hash(img)) for img in images]
        dct_hashes = [binary_array_to_int(my_hasher.dct_hash(img)[0]) for img in images]
        dcts = [my_hasher.dct_hash(img)[1] for img in images]
        # TODO: See what accounts for differences in hash computations
        print("Comparison to imagehash library")
        for i in range(len(images)):
            print(i)
            print("-" * 50)
            print("AVERAGES")
            print(str(imagehash_avg_hashes[i]) == hex(avg_hashes[i]))
            print(str(imagehash_avg_hashes[i]))
            print(hex(avg_hashes[i]))
            print("DIFF")
            print(str(imagehash_diff_hashes[i]) == hex(diff_hashes[i]))
            print(str(imagehash_diff_hashes[i]))
            print(hex(diff_hashes[i]))
            print("DCT")
            print(str(imagehash_dct_hashes[i]) == hex(dct_hashes[i]))
            print(str(imagehash_dct_hashes[i]))
            print(hex(dct_hashes[i]))

        print("DIFFERENCE HASHES")
        for j in range(len(diff_hashes)):
            print(hamming_distance(diff_hashes[0], diff_hashes[j]))
            print(imagehash_diff_hashes[0] - imagehash_diff_hashes[j])
        print("AVERAGE  HASHES")
        for j in range(len(avg_hashes)):
            print(hamming_distance(avg_hashes[0], avg_hashes[j]))
            print(imagehash_avg_hashes[0] - imagehash_avg_hashes[j])

        print("DCT HASHES")
        for j in range(len(dct_hashes)):
            print(hamming_distance(dct_hashes[0], dct_hashes[j]))
            print(imagehash_dct_hashes[0] - imagehash_dct_hashes[j])

        a_fake = a.copy()
        a_fake[0, :] = 0
        # print (a[0,0])
        # print (a_fake[0,0])
        a_hash = imagehash.dhash(Image.fromarray(a), hash_size=hsize)
        a_fake_hash = imagehash.dhash(Image.fromarray(a_fake), hash_size=hsize)
        b_hash = imagehash.dhash(Image.fromarray(b), hash_size=hsize)

        print("A vs A_fake vs B")
        print(a_hash)
        print(a_fake_hash)
        print(b_hash)
        print(a_hash - b_hash)

    def test_transformations(self):
        # Linear pixel intensity transformations (ex. brightness) is handled
        n = 100
        f, axarr = plt.subplots (3, figsize=(20,20))
        ims = [np.random.uniform(0, 1, (200,200,3)) for j in range (n)]
        hasher = ImageHasher(8)
        original_hashes = [binary_array_to_int(hasher.average_hash(im)) for im in ims]
        vars = [1e-2, 1e-1, 1, 5, 10, 100]
        avg_dists = []
        for var in vars:
            dists = []
            for j in range(len(ims)):
                varim = ims[j] + np.random.normal(0, scale=var, size=ims[j].shape)
                varim_hash = binary_array_to_int(hasher.average_hash(varim))
                dists.append(hamming_distance(original_hashes[j], varim_hash))
            avg_dists.append(np.mean(dists))
        axarr[0].grid()
        axarr[0].plot(vars, avg_dists, linewidth=5)
        axarr[0].set_xlabel("Variance of Gaussian Noise", fontsize=18)
        axarr[0].set_ylabel("Average Hamming Distance of %d Random Images"%n, fontsize=18)
        axarr[0].set_title("Hamming distance by Gaussian noise variance", fontsize=18)

        scales = [0.125, 0.25, 0.5, 0.75, 1,  1.5, 2, 4, 8]
        avg_dists = []
        for scale in scales:
            dists = []
            for j in range(len(ims)):
                scaleim = cv2.resize(ims[j], (int(scale * ims[j].shape[0]), int(scale * ims[j].shape[1])))
                scale_hash = binary_array_to_int(hasher.average_hash(scaleim))
                dists.append(hamming_distance(original_hashes[j], scale_hash))
            avg_dists.append(np.mean(dists))
        axarr[1].grid()
        axarr[1].plot(scales, avg_dists, linewidth=5)
        axarr[1].set_xlabel("Scale factor", fontsize=18)
        axarr[1].set_ylabel("Hamming Distance of %d Random Images"%n, fontsize=18)
        axarr[1].set_title("Hamming distance by scale", fontsize=18)

        def rotate_image(image, angle):
            # source: https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            return result

        thetas = np.arange(0, 181, 10)
        avg_dists = []
        for theta in thetas:
            dists = []
            for j in range(len(ims)):
                rotim = rotate_image(ims[j], theta)
                rot_hash = binary_array_to_int(hasher.average_hash(rotim))
                dists.append(hamming_distance(original_hashes[j], rot_hash))
            avg_dists.append(np.mean(dists))
        axarr[2].grid()
        axarr[2].plot(thetas, avg_dists, linewidth=5)
        axarr[2].set_xlabel("Rotation Angle (degree)", fontsize=18)
        axarr[2].set_ylabel("Average Hamming Distance of %d Random Images"%n, fontsize=18)
        axarr[2].set_title("Hamming distance by Rotation", fontsize=18)
        plt.show()




if __name__ == "__main__":
    unittest.main()