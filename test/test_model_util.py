
import os

import numpy as np

import os
import sys
sys.path.append(os.getcwd())

import model
from model import ZPKToMagLayer

import unittest
import tensorflow as tf

import model.util as util


class test_model_util(unittest.TestCase):

    def test_mag2db(self):
       
        exp_0dB = util.mag2db(tf.constant(1.)).numpy()
        self.assertAlmostEqual(exp_0dB, 0., delta=10E-6)
        
        exp_6dB = util.mag2db(tf.constant(2.)).numpy()
        self.assertAlmostEqual(exp_6dB, 6.0206, delta=10E-4)
        
        exp_n120dB = util.mag2db(tf.constant(10E-6)).numpy()
        self.assertAlmostEqual(exp_n120dB, -100., delta=10E-6)

        # complex numbers
        exp_3dB = util.mag2db(tf.complex(1., 1.)).numpy()
        self.assertAlmostEqual(exp_3dB, 3.0103, delta=10E-4)


    def test_mag2db(self):
       
        exp_1 = util.db2mag(tf.constant(0.)).numpy()
        self.assertAlmostEqual(exp_1, 1., delta=10E-6)
        
        exp_2 = util.db2mag(tf.constant(6.0206)).numpy()
        self.assertAlmostEqual(exp_2, 2, delta=10E-6)
        
        exp_10En6 = util.db2mag(tf.constant(-100.)).numpy()
        self.assertAlmostEqual(exp_10En6, 10E-6, delta=10E-7)

    def test_lin_scale(self):

        self.assertAlmostEqual(util.lin_scale(-20., -20., -10., -50., +50.), -50, delta=10E-6)
        self.assertAlmostEqual(util.lin_scale(-15., -20., -10., -50., +50.),   0, delta=10E-6)

if __name__ == '__main__':   
    unittest.main()

