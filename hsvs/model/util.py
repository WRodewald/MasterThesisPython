import tensorflow as tf
import math
import numpy as np
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


# loss expects a list of pole / zero positions
def pole_zero_loss(y_true, y_pred, gain_weight, angle_weight):

    #angle
    angle_true = tf.math.angle(y_true)
    angle_pred = tf.math.angle(y_pred)

    angle_loss = tf.square(angle_weight * (angle_true - angle_pred))

    # gain
    gain_true = mag2db(1. / (1.-tf.math.abs(y_true)))
    gain_pred = mag2db(1. / (1.-tf.math.abs(y_pred)))

    gain_loss = tf.square(gain_weight * (gain_true-gain_pred))

    return tf.reduce_mean(angle_loss + gain_loss, 1)


def weighted_mse_loss(y_true, y_pred, y_weight):
    return tf.reduce_mean(y_weight * tf.square(y_true-y_pred), 1)

def weighted_mse_loss_phase(y_true, y_pred, y_weight, y_weight_phase, phase_weight):

    db_diff = mag2db(tf.abs(y_true)) - mag2db(tf.abs(y_pred))
    phase_diff = tf.math.angle(y_true) - tf.math.angle(y_pred)

    # phase loss is maximum (phase_weight^2) at a 180° difference

    loss = y_weight * tf.square(db_diff) \
         + y_weight_phase * tf.square(phase_weight * tf.sin(0.5*phase_diff))

    return tf.reduce_mean(loss, 1)


def convert_to_time_domain(ot_series):

    # empty for DC and nyquist
    empty = tf.reduce_mean(tf.zeros_like(ot_series), axis=1, keepdims=True)
    
    # flipped complex conjugate vector for symmetry
    ot_conj = tf.reverse(tf.math.conj(ot_series), axis= [1])

    # create spectrum
    spectrum = tf.concat([empty, ot_series, empty, ot_conj], axis=1)

    return tf.signal.ifft(spectrum)



def loss_time_domain(y_true, y_pred):

    y_true = convert_to_time_domain(y_true)
    y_pred = convert_to_time_domain(y_pred)

    diff = y_true - y_pred
    
    return mag2db(tf.sqrt(tf.reduce_mean(tf.square(tf.abs(diff)))))


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



class MagToDBLayer(tf.keras.layers.Layer):
    """Converts magnitudes to decibel"""


    def __init__(self,  **kwargs):
        super(MagToDBLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(MagToDBLayer, self).build(input_shape)

    def call(self, input):
        return mag2db(tf.abs(input))

class DBToMagLayer(tf.keras.layers.Layer):
    """Converts decibel to magnitude"""


    def __init__(self,  **kwargs):
        super(DBToMagLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(DBToMagLayer, self).build(input_shape)

    def call(self, input):
        return db2mag(input)
        
class RelativePhaseLayer(tf.keras.layers.Layer):
    """Converts complex overtone spectrum to relative phase"""

    def __init__(self,  **kwargs):
        super(RelativePhaseLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(RelativePhaseLayer, self).build(input_shape)
        num_overtones = input_shape[1]
        self.k = tf.constant(tf.range(1,num_overtones+1, 1.), shape = [1, num_overtones])

    def call(self, input):
        
        # extract fundamental phase
        d0t = tf.math.angle(input[:,0])

        # adjust for frequency ratio 
        dkt = tf.expand_dims(d0t, 1) * self.k

        # adjust original phases with relative phases
        return input * tf.exp(tf.complex(0., -dkt))



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



class ExponentialGainLayer(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(ExponentialGainLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(ExponentialGainLayer, self).build(input_shape)
        
        # allocate trainable variables for N samples / K parameters
        self.gain = tf.Variable(initial_value=0., trainable=False)
        self.increment = tf.Variable(initial_value=0., trainable=False)
    def call(self, input):
        
        new_gain = tf.clip_by_value(self.gain + self.increment, 0., 1.)
        self.gain.assign(new_gain)


        return tf.pow(input, tf.cast(self.gain, dtype=input.dtype))






