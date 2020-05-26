
import tensorflow as tf
import numpy as np
import math

from . import util

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

        #a = tf.constant([-0.6004, -7.9514, -6.5671, 0.2403, 1.5771, -1.1222, -9.1857, 0.0254, 3.2368, 2.7025, -8.8023, -0.9560, 2.4130, 3.9092, -1.3690, -2.5269, -1.3453, 3.6957, 2.8644, -2.1918, -10.8530, 1.3129, 4.1825, -0.4516, -18.2814, -5.8255, 3.2353, 0.5224, -7.8534, -19.9245, -1.2931, 4.4018, -0.1311, -20.5924, -15.6537, 2.0059, 0.4917, -2.7046, -23.8272, -6.9569],shape=[1,40])
        #b = tf.constant([3.6578, -6.6685, -16.7305, -23.4105, -22.0545, -26.2011, -26.9520, -35.2314, -35.4254, -35.7124, -35.5698, 1.8516, -45.9318, -46.1479, -46.8850, 9.1236, -50.6755, -53.4893, -53.7630, 8.7649, -47.4303, -57.7351, -59.0493, -57.9440, -43.3852, -56.4816, -62.9547, 0.9517, -56.7403, -45.5120, -63.6512, -66.9056, 3.8231, -47.5074, -53.5038, -68.8057, 2.6530, -66.4998, -46.2038, -64.5889],shape=[1,40])
        #c = tf.constant([-18.0145, -4.2112, -1.0069, -5.3725, -11.3833, -6.8484, -0.1307, -3.3757, -8.4110, -9.1802, 0.9363, -45.9644, -3.0069, -5.5746, -0.5417, -56.4671, 1.1069, -2.2872, -2.0655, -60.2739, 4.0197, 0.9714, -1.4581, 1.4875, 4.5815, 4.1845, 0.6312, -61.1359, 4.5894, 5.2196, 3.5983, 0.3995, -66.1897, 5.6634, 6.0862, 2.7026, -67.7105, 4.2986, 5.4138, 5.8769],shape=[1,40])
        #k = tf.constant([-0.4791, -1.1135, -1.1196, -0.6542, -0.3961, -0.6105, -1.0474, -0.7135, -0.4964, -0.4874, -1.0142, 0.1913, -0.6766, -0.5819, -0.8059, 0.2687, -0.8557, -0.6808, -0.6983, 0.2711, -1.0936, -0.8057, -0.7011, -0.8422, -1.2203, -1.0001, -0.7707, 0.2268, -1.0343, -1.2471, -0.9103, -0.7532, 0.2435, -1.2559, -1.1883, -0.8419, 0.2394, -0.9420, -1.2840, -1.0430],shape=[1,40])

        ot15 = (p1 * Rd * Rd * Rd + p2 * Rd * Rd + p3 * Rd + p4) / (q1 + Rd)

        #ot = (a*Rd*Rd + b*Rd + c) * (tf.pow(Rd,k))

        return tf.concat([ot15, ot6], 1)
