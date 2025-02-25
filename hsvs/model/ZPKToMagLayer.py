
import tensorflow as tf
import numpy as np

from . import util


class ZPKToMagLayer(tf.keras.layers.Layer):
    """Converts pole - zero - gain parameters to a magnitude response """


    def __init__(self, sample_rate, output_size, **kwargs):
        super(ZPKToMagLayer, self).__init__(**kwargs)

        self.sample_rate = sample_rate
        self.output_size = output_size


        # indice of overtones
        self.k = tf.constant(tf.range(1,output_size+1, 1.), shape = [1, 1, output_size])
        
        self.w0 = tf.constant(2 * np.pi / sample_rate)


    def build(self, input_shapes):
        if not isinstance(input_shapes, list):
            raise ValueError('This layer should be called on a list of inputs.')

        super(ZPKToMagLayer, self).build(input_shapes[0]) 


    def extract_zpk_parameters(self, zpk):

        length = zpk.shape[1]

        if(length % 4 == 0):
            # 4*N parameters, we assume konstant k = 1            
            pw = zpk[:,0:length+1:4]
            pr = zpk[:,1:length+1:4]
            zw = zpk[:,2:length+1:4]
            zr = zpk[:,3:length+1:4]

            k = tf.ones_like(zpk[:,0])

        else:
            # 4*N+1 parameters, we have a specific k
            k  = zpk[:,0]
            pw = zpk[:,1:length+1:4]
            pr = zpk[:,2:length+1:4]
            zw = zpk[:,3:length+1:4]
            zr = zpk[:,4:length+1:4]
                
            k  = tf.sigmoid(k)
            k   = util.db2mag(util.lin_scale(k, 0, 1, -100, 0))


        max_f = tf.constant(8000./self.sample_rate)
        zw = max_f * tf.sigmoid(zw)
        pw = max_f * tf.sigmoid(pw)

        # parameter shaping
        pr = tf.sigmoid(pr)
        zr = tf.sigmoid(zr)
        pr  = 1. - 1. / util.db2mag(util.lin_scale(pr, 0, 1,  0, 90))
        zr  = 1. - 1. / util.db2mag(util.lin_scale(zr, 0, 1,  0, 90))
        

        #p0 = pr * tf.exp(tf.complex(0., pw * pi))
        #z0 = zr * tf.exp(tf.complex(0., zw * pi))

        p0 = tf.multiply(tf.complex(pr,0.), tf.complex(tf.math.cos(pw * 2 * np.pi), tf.math.sin(pw *  2 * np.pi)), name="p0")
        z0 = tf.multiply(tf.complex(zr,0.), tf.complex(tf.math.cos(zw * 2 * np.pi), tf.math.sin(zw *  2 * np.pi)), name="z0")

        return z0, p0, k


    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, self.output_size])

    def response(self, z0, p0, k, w):
        
        #z = tf.exp(tf.complex(0., w))
        z = tf.complex(tf.math.cos(w), tf.math.sin(w), name="z")
        
        # adjust dimensionality

        z0 = tf.expand_dims(z0, 2)
        p0 = tf.expand_dims(p0, 2)
        k  = tf.expand_dims(k, 1)

        # feed forward and feed back 
        Hff = (z - z0) * (z - tf.math.conj(z0))        
        Hfb = (z - p0) * (z - tf.math.conj(p0))

        # calculate transfer function
        H = tf.complex(k, 0.) * tf.reduce_prod(Hff / Hfb, 1)

        return H, z0, p0, k

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('This layer should be called on a list of inputs.')

        input_tensor = inputs[0]

        # sampling positions
        f0 = tf.expand_dims(inputs[1],1)
        w  = f0 * self.k * self.w0

        # filter coeff extraction
        z0, p0, k = self.extract_zpk_parameters(input_tensor)


        return self.response(z0, p0, k, w)
