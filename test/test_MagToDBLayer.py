
import os

import numpy as np

import unittest
import tensorflow as tf

import model
from model import MagToDBLayer


class test_MagToDBLayer(unittest.TestCase):

    def test_magToDBLayer(self):
        layer = MagToDBLayer.MagToDBLayer()
        
        self.assertEqual(layer(tf.complex(1., 0.)).numpy(), 0.)
        self.assertAlmostEqual(layer(tf.complex(1., 1.)).numpy(), 3.01, delta=10E-3)
        self.assertAlmostEqual(layer(tf.constant(-2.)).numpy(),   6.02, delta=10E-3)
            



if __name__ == '__main__':   
    unittest.main()

