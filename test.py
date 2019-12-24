from hash import ImageHasher, hamming_distance
import imagehash
import unittest

class CorrectHamming(unittest.TestCase):
    verified = ( ((1, 1),0), # same two inputs
                 ((0, 150), 0), # zero input
                 ((1, 1), 0),
                 ((1, 1), 0),
                 ((1, 1), 0),
                 ((1, 1), 0),

                 )