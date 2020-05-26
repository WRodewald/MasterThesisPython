
import tensorflow as tf
import numpy as np
import math

class NormalizeLayer(tf.keras.layers.Layer):

    def __init__(self,  in_mean, in_std, out_mean = 0.5, out_std = 1., **kwargs):
        super(NormalizeLayer, self).__init__(**kwargs)

        self.factor = tf.constant(out_std/in_std, dtype=tf.float32)
        self.offset = tf.constant(out_mean - in_mean * self.factor, dtype=tf.float32)


    def build(self, input_shape):
        super(NormalizeLayer, self).build(input_shape)

    def call(self, input):
        return tf.add(tf.multiply(input, self.factor),self.offset)
