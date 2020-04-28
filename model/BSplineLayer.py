

import numpy as np
import tensorflow as tf
from scipy.interpolate import splev

class BSplineLayer(tf.keras.layers.Layer):

    def __init__(self, in_size, out_size, **kwargs):
        super(BSplineLayer, self).__init__(**kwargs)

        self.in_size = in_size
        self.out_size = out_size

    def build(self, input_shape):
        super(BSplineLayer, self).build(input_shape)
        
        #construct the base spline matrix used for interpolation
        mat = np.zeros([self.out_size, self.in_size, 1])

        time = np.linspace(0., self.in_size-2, self.out_size)    
        
        # knots [0, 0, 0, 1, 2, ..., M-3, M-2, M-2, M-2]
        knots = np.arange(0, self.in_size - 1)
        knots = np.concatenate(([0, 0], knots, [self.in_size-2, self.in_size-2]))

        for m in range(self.in_size):

            coeffs= np.zeros([self.in_size])
            coeffs[m] = 1.
            
            mat[:,m,0] = splev(time, (knots, coeffs, 2))

        self.mat = tf.constant(mat, dtype=self.dtype)


    def call(self, input):
        
        return tf.reduce_sum( tf.expand_dims(input, 0) * self.mat, 1)



