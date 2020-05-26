
import os

import numpy as np

import os
import sys
sys.path.append(os.getcwd())

import hsvs.model
from hsvs.model import ZPKToMagLayer

import unittest
import tensorflow as tf


class test_NormalizeLayer(unittest.TestCase):

    def test_ZPKTransferFunction(self):
        fs = 44100
        layer = ZPKToMagLayer.ZPKToMagLayer(fs, 1)

        p = tf.complex(0.25, 0.25)
        k = 4.
        z = tf.complex(-1., 0.)
            
        w0 = 2 * np.pi * tf.constant([100., 500., 1000., 5000., 10000.]) / fs

        z  = tf.reshape(z, [1,1])
        p  = tf.reshape(p, [1,1])
        k  = tf.reshape(k, [1])

        w0 = tf.reshape(w0, [1, w0.shape[0]])
        
        H, _, _, _ = layer.response(z, p, k, w0)

        H = tf.abs(H)

        self.assertAlmostEqual(H.numpy()[0,0], 25.598, delta=0.01)
        self.assertAlmostEqual(H.numpy()[0,1], 25.557, delta=0.01)
        self.assertAlmostEqual(H.numpy()[0,2], 25.427, delta=0.01)
        self.assertAlmostEqual(H.numpy()[0,3], 20.937, delta=0.01)
        self.assertAlmostEqual(H.numpy()[0,4],  9.867, delta=0.01)


if __name__ == '__main__':   
    unittest.main()

