from hash import ImageHasher, hamming_distance
import imagehash
import unittest
import numpy as np
import time

class CorrectHamming(unittest.TestCase):
    verified = ( ((1, 1),0), # same two inputs
                 ((0, 150), 4), # zero input
                 ((0x0d5585f2ed36cc3758b4d, 0x0d5585f2ed36cc3758b4d), 0),
                 ((1, 1), 0),
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


if __name__ == "__main__":
    unittest.main()