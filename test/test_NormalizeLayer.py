
import os

import numpy as np

import unittest
import tensorflow as tf

import model
from model import NormalizeLayer


class test_NormalizeLayer(unittest.TestCase):

    def test_NormalizeLayer(self):
        layer = NormalizeLayer.NormalizeLayer(2, 2, 0, 1)
        
        self.assertEqual(layer(2.).numpy(),  0.)
        self.assertEqual(layer(4.).numpy(), +1.)
        self.assertEqual(layer(0.).numpy(), -1.)
        
            



if __name__ == '__main__':   
    unittest.main()

