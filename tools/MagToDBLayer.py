
import tensorflow as tf
import numpy as np
import math

class MagToDBLayer(tf.keras.layers.Layer):
    """Converts magnitudes to decibel"""


    def __init__(self,  **kwargs):
        super(MagToDBLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(MagToDBLayer, self).build(input_shape)

    def call(self, input):

        factor = tf.constant(20 / math.log(10))
        return tf.multiply(factor, tf.math.log(tf.abs(input)))
