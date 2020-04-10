
import tensorflow as tf
import numpy as np


def linearScale(x, x_min, x_max, y_min, y_max):
    normalized = (x - x_min) / (x_max - x_min)

    return normalized * (y_max - y_min) + y_min

def db2mag(x):
    return tf.math.pow(10., x * 0.05)

class ParZPKToMagLayer(tf.keras.layers.Layer):
    """Converts pole - zero - gain parameters to a magnitude response """


    def __init__(self, sample_rate, output_size, **kwargs):
        super(ParZPKToMagLayer, self).__init__(**kwargs)

        self.sample_rate = sample_rate
        self.output_size = output_size;


        # indice of overtones
        self.k = tf.constant(tf.range(1,output_size+1, 1.), shape = [1, 1, output_size])
        
        self.w0 = tf.constant(2 * np.pi / sample_rate);


    def build(self, input_shapes):
        if not isinstance(input_shapes, list):
            raise ValueError('This layer should be called on a list of inputs.')

        super(ParZPKToMagLayer, self).build(input_shapes[0]) 


    def extract_zpk_parameters(self, zpk):

        length = zpk.shape[1]

        k  = zpk[:,0:length+1:5]
        pw = zpk[:,1:length+1:5]
        pr = zpk[:,2:length+1:5]
        zw = zpk[:,3:length+1:5]
        zr = zpk[:,4:length+1:5]

        # parameter shaping
        pr = tf.sigmoid(pr)
        zr = tf.sigmoid(zr)
        k  = tf.sigmoid(k)
        k   = db2mag(linearScale(k, 0, 1, -100, +50))
        pr  = 1. - 1. / db2mag(linearScale(pr, -1, 1,  0, 60))
        zr  = 1. - 1. / db2mag(linearScale(zr, -1, 1,  0, 60))
        
        pi = tf.constant(np.pi);

        #p0 = pr * tf.exp(tf.complex(0., pw * pi))
        #z0 = zr * tf.exp(tf.complex(0., zw * pi))

        p0 = tf.multiply(tf.complex(pr,0.), tf.complex(tf.math.cos(pw * pi), tf.math.sin(pw * pi)), name="p0")
        z0 = tf.multiply(tf.complex(zr,0.), tf.complex(tf.math.cos(zw * pi), tf.math.sin(zw * pi)), name="z0")

        return z0, p0, k


    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, self.output_size])

    def response(self, z0, p0, k, w):
        
        #z = tf.exp(tf.complex(0., w))
        z = tf.complex(tf.math.cos(w), tf.math.sin(w), name="z")
        
        # adjust dimensionality

        z0 = tf.expand_dims(z0, 2);
        p0 = tf.expand_dims(p0, 2);
        k  = tf.expand_dims(k,  2);

        # feed forward and feed back 
        Hff = (z - z0) * (z - tf.math.conj(z0))        
        Hfb = (z - p0) * (z - tf.math.conj(p0))

        # calculate transfer function
        H = tf.reduce_sum( tf.complex(k, 0.) * Hff / Hfb, 1)

        return H

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('This layer should be called on a list of inputs.')

        input_tensor = inputs[0]

        # sampling positions
        f0 = tf.expand_dims(inputs[1],1);
        w  = f0 * self.k * self.w0;

        # filter coeff extraction
        z0, p0, k = self.extract_zpk_parameters(input_tensor)


        return self.response(z0, p0, k, w)
