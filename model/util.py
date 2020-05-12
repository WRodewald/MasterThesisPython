import tensorflow as tf
import math
import matplotlib.pyplot as plt

# utility functions for tensorflow tensors 


def log10(x):
    factor = tf.constant(1. / math.log(10.))
    return factor * tf.math.log(tf.abs(x))

def log2(x):
    factor = tf.constant(1. / math.log(2.))
    return factor * tf.math.log(tf.abs(x))

def mag2db(x):
    factor = tf.constant(10. / math.log(10.))
    return factor * tf.math.log(tf.square(x))

def db2mag(x):
    return tf.math.pow(10., x * 0.05)

def lin_scale(x, x_min, x_max, y_min, y_max):
    normalized = (x - x_min) / (x_max - x_min)
    return normalized * (y_max - y_min) + y_min


def weighted_mse_loss(y_true, y_pred, y_weight):
    return tf.reduce_mean(y_weight * tf.square(y_true-y_pred), 1)


def loss_ddsp(y_true, y_pred, frequencies, cutoff = 3000., alpha = 1.):
    
    # 6dB low pass
    lpf = tf.cast(tf.abs(1. / (1. + 1j * frequencies/cutoff)), dtype=y_true.dtype)

    # -6dB interation, 
    lpf = tf.cast(tf.abs(1. / (1j * frequencies/cutoff)), dtype=y_true.dtype)
    lpf = lpf / tf.reduce_mean(lpf)
    

    mag_true = db2mag(y_true) + 10E-7
    mag_pred = db2mag(y_pred) + 10E-7

    return lpf * (tf.abs(mag_true - mag_pred) + alpha * tf.abs(tf.math.log(mag_true) - tf.math.log(mag_pred)))


def filtered_diff_loss(y_true, y_pred, frequencies, cutoff, noise_floor = -80):
    # currently expecting decibel values as input, so we need to convert back to mag

    db_min = noise_floor

    diff = tf.cast(tf.abs(db2mag(y_true) - db2mag(y_pred)), dtype=y_true.dtype)
    lpf  = tf.cast(tf.abs(1. / (1. + 1j * frequencies/cutoff)), dtype=y_true.dtype)

    return tf.reduce_mean(diff)




class GainLayer(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(GainLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(GainLayer, self).build(input_shape)
        
        # allocate trainable variables for N samples / K parameters
        self.gain = tf.Variable(initial_value=0., trainable=False)
        self.increment = tf.Variable(initial_value=0., trainable=False)
    def call(self, input):
        
        new_gain = tf.clip_by_value(self.gain + self.increment, 0., 1.)
        self.gain.assign(new_gain)


        return input * self.gain







