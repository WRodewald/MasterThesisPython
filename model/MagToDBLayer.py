
import tensorflow as tf
import numpy as np
import math

from model import util

class MagToDBLayer(tf.keras.layers.Layer):
    """Converts magnitudes to decibel"""


    def __init__(self,  **kwargs):
        super(MagToDBLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(MagToDBLayer, self).build(input_shape)

    def call(self, input):
        return util.mag2db(input)
