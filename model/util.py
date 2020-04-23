import tensorflow as tf
import math

# utility functions for tensorflow tensors 

def mag2db(x):
    factor = tf.constant(20. / math.log(10.))
    return factor * tf.math.log(tf.abs(x))


def log10(x):
    factor = tf.constant(1. / math.log(10.))
    return factor * tf.math.log(tf.abs(x))

def log2(x):
    factor = tf.constant(1. / math.log(2.))
    return factor * tf.math.log(tf.abs(x))

def db2mag(x):
    return tf.math.pow(10., x * 0.05)

def lin_scale(x, x_min, x_max, y_min, y_max):
    normalized = (x - x_min) / (x_max - x_min)
    return normalized * (y_max - y_min) + y_min


def weighted_mse_loss(y_true, y_pred, y_weight):
    return tf.reduce_mean(y_weight * tf.square(y_true-y_pred), 1)