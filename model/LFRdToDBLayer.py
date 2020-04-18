
import tensorflow as tf
import numpy as np
import math

from model import util

class LFRdToDBLayer(tf.keras.layers.Layer):
    """Approxmates an LF-Rd magnitude response using a 4th order bandpass respponse"""


    def __init__(self, output_size,  **kwargs):
        super(LFRdToDBLayer, self).__init__(**kwargs)
        self.output_size = output_size;

        self.s = tf.complex(0., tf.range(3., output_size+1, dtype='float32'))
        self.s = tf.reshape(self.s, [1, output_size-2])

    def build(self, input_shape):
        super(LFRdToDBLayer, self).build(input_shape)

    def call(self, input):
        Rd = input
        # polynomial approximation of filter parameters
        Q  = 0.52 * Rd;
        g = (-0.1626 * Rd  + 0.985) / (Rd + 0.00937);

        # calculate low level filter coeffs from parameters
        k = tf.complex(1./Q, 0.)

        # calculate frequency response 
        s = self.s
        H = tf.abs(s / (s * s + k * s + tf.complex(1., 0.)))
        ot6 = util.mag2db(H * H * g)

        # polynomial approximation for harmonics 1 and 2    
        p1 = tf.constant([    0.7362,    1.2460], shape=[1,2])
        p2 = tf.constant([   -3.7175,  -12.2749], shape=[1,2])
        p3 = tf.constant([    4.0541,    0.2927], shape=[1,2])
        p4 = tf.constant([  -21.9442,  -10.8228], shape=[1,2])
        q1 = tf.constant([    0.4049,    0.1529], shape=[1,2])

        ot15 = (p1 * Rd * Rd * Rd + p2 * Rd * Rd + p3 * Rd + p4) / (q1 + Rd)

        return tf.concat([ot15, ot6], 1)
