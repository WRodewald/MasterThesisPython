
import tensorflow as tf
import numpy as np

class ZPKToMagLayer(tf.keras.layers.Layer):
    """Converts pole - zero - gain parameters to a magnitude response """


    def __init__(self, sample_rate, output_size, **kwargs):
        super(ZPKToMagLayer, self).__init__(**kwargs)

        self.sample_rate = sample_rate
        self.output_size = output_size;


        # indice of overtones
        self.k = tf.constant(tf.range(1,output_size+1, 1.), shape = [1, 1, output_size])
        
        self.w0 = tf.constant(np.pi / sample_rate);

    def build(self, input_shapes):
        if not isinstance(input_shapes, list):
            raise ValueError('This layer should be called on a list of inputs.')
        #build step
        super(ZPKToMagLayer, self).build(input_shapes[0])  # Be sure to call this somewhere!

    def extract_zpk_parameters(self, zpk):

        length = zpk.shape[1]

        k  = zpk[:,0]
        pw = zpk[:,1:length+1:4]
        pr = tf.complex(zpk[:,2:length+1:4],0.)
        zw = zpk[:,3:length+1:4]
        zr = tf.complex(zpk[:,4:length+1:4],0.)
        
        pi = tf.constant(np.pi);

        #p0 = tf.multiply(tf.complex(pr, 0.), tf.exp(tf.complex(0., pw)))
        #z0 = tf.multiply(tf.complex(zr, 0.), tf.exp(tf.complex(0., zw)))

        p0 = tf.multiply(pr, tf.complex(tf.math.cos(pw * pi), tf.math.sin(pw * pi)), name="p0")
        z0 = tf.multiply(zr, tf.complex(tf.math.cos(zw * pi), tf.math.sin(zw * pi)), name="z0")

        return z0, p0, k

    def compute_output_shape(self, input_shape):
        print("Did you even ask me?")
        return tf.TensorShape([None, self.output_size])

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('This layer should be called on a list of inputs.')
        zpk = inputs[0]
        f0  = tf.expand_dims(inputs[1],1);

        # sampling positions
        f = tf.multiply(f0, self.k);
        w = tf.multiply(f, self.w0);

        z = tf.complex(tf.math.cos(w), tf.math.sin(w), name="z")

        # filter coeff extraction
        z0, p0, gain = self.extract_zpk_parameters(zpk)

        z0   = tf.expand_dims(z0, 2);
        p0   = tf.expand_dims(p0, 2);
        gain = tf.expand_dims(gain, 1);

        # feed forward
        H = tf.subtract(z,z0)
        H = tf.multiply(H, tf.subtract(z, tf.math.conj(z0)))
        
        # feed back
        H = tf.divide(H, tf.subtract(z, p0))
        H = tf.divide(H, tf.subtract(z, tf.math.conj(p0)))

        # product and gain
        H = tf.reduce_prod(H, 1)
        H = tf.multiply(tf.abs(H), gain)



        return H
