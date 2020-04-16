import unittest
import imagehash
import numpy as np
import time
import imagehash
from PIL import Image

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


if __name__ == "__main__":
    unittest.main()