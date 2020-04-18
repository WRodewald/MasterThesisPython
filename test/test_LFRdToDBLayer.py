
import os

import numpy as np

import os
import sys
sys.path.append(os.getcwd())

import model
from model import LFRdToDBLayer

import unittest
import tensorflow as tf

import model.util as util


class test_LFRdToDBLayer(unittest.TestCase):

    def test_call_regression(self):
       
        # create layer with 10 overtones in output
        layer = LFRdToDBLayer.LFRdToDBLayer(10)

        # Rd = 0.3
        out = layer(0.3)

        expected = [-29.851988, -26.06776 , -24.043396, -25.213526, -26.523418,
                    -27.89702 , -29.282688, -30.64777 , -31.973135, -33.248634]

        print(out.shape)
        
        for i in range(len(expected)):
            self.assertAlmostEqual(out[0,i], expected[i], delta=10E-3)

        # Rd = 2.7
        out = layer(2.7)      
        
        expected = [-7.6034913, -26.286062 , -31.55112  , -37.182903 , -41.352654 ,
                    -44.67904  , -47.452766 , -49.834618 , -51.92332  , -53.78408]
       
        for i in range(len(expected)):
            self.assertAlmostEqual(out[0,i], expected[i], delta=10E-3)
              


if __name__ == '__main__':   
    unittest.main()

