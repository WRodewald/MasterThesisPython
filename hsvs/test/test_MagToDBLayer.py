
import os

import numpy as np

import unittest
import tensorflow as tf

import hsvs.model
from hsvs.model import util


class test_MagToDBLayer(unittest.TestCase):

    def test_magToDBLayer(self):
        layer = util.MagToDBLayer()
        
        self.assertEqual(layer(tf.complex(1., 0.)).numpy(), 0.)
        self.assertAlmostEqual(layer(tf.complex(1., 1.)).numpy(), 3.01, delta=10E-3)
        self.assertAlmostEqual(layer(tf.constant(-2.)).numpy(),   6.02, delta=10E-3)
            
    def test_dbToMagLayer(self):
        layer = util.DBToMagLayer()
        
        self.assertEqual(layer(0.).numpy(), 1.)
        self.assertAlmostEqual(layer(  3.01).numpy(),  np.sqrt(2), delta=10E-3)
        self.assertAlmostEqual(layer( -6.00).numpy(),  0.5,        delta=10E-3)



if __name__ == '__main__':   
    unittest.main()

