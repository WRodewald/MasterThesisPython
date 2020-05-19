
import tensorflow as tf
import numpy as np
import math

from model import util

# based on the matlab implementation of covarep:
# G. Degottex, J. Kane, T. Drugman, T. Raitio and S. Scherer, 
# "COVAREP - A collaborative voice analysis repository for speech technologies", 
# In Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
# Florence, Italy 2014.

# function calculates LF parameters from 1-parameter LF-Rd
def convert_Rd_to_tx(Rd):
    Rap = (-1.  +  4.8 * Rd) * 0.01
    Rkp = (22.4 + 11.8 * Rd) * 0.01
    RgpInv = (0.44*Rd - 2.*Rap - 4.8*Rap*Rkp) / (1.2*Rkp*Rkp + 0.5*Rkp) 

    tp = 0.5 * RgpInv
    te = tp * (Rkp+1)
    ta = Rap

    return tp, te, ta

# function approximates alpha from Rd with a 2nd/1st order rational polynom
def approximate_alpha(Rd):
        p1 = -0.09818
        p2 =  -0.6536
        p3 =    3.689
        q1 =  0.04759

        return (p1 * Rd * Rd + p2 * Rd + p3) / (Rd + q1)

# function approximates epsilon from Rd with a 2nd/1st order rational polynom
def approximate_epsilon(Rd):
        p1 =  -1.671
        p2 =   2.813
        p3 =   19.71
        q1 = -0.2104

        return (p1 * Rd * Rd + p2 * Rd + p3) / (Rd + q1)

class LFRdLayer(tf.keras.layers.Layer):
    """LF-Rd magnitude response using a 4th order bandpass respponse"""


    def __init__(self, output_size,  **kwargs):
        super(LFRdLayer, self).__init__(**kwargs)
        self.output_size = output_size;

        self.k = tf.constant(tf.range(1,output_size+1, 1.), shape = [1, output_size])


    def build(self, input_shape):
        super(LFRdLayer, self).build(input_shape)

    def call(self, input):
        Rd = input

        # converting Rd to 4-parameter model
        tp, te, ta = convert_Rd_to_tx(Rd)        
        wg = np.pi/tp
        
        # approximations
        a = approximate_alpha(Rd)
        e = approximate_epsilon(Rd)

        # gain parameters
        Ee = tf.complex(1., 0.)
        E0 = tf.complex(-1./(tf.exp(a*te) * tf.sin(wg*te)), 0.)

        wg = tf.complex(wg, 0.) 
        te = tf.complex(te, 0.) 
        ta = tf.complex(ta, 0.)
        a  = tf.complex(a,  0.)
        e  = tf.complex(e,  0.)

        # spectral glottal flow 
        w = tf.complex(0., 2. * np.pi * self.k)

        P1 = E0 * (1. /((a - w)*(a - w) + wg*wg));
        P2 = wg + tf.exp((a - w)*te) *((a - w)*tf.sin(wg * te) - wg * tf.cos(wg * te))
        P3 = Ee*(tf.exp( -w * te)/((e * ta * w) * (e + w)))
        P4 = e*(1 - e * ta)*(1. - tf.exp(- w*(1. - te))) - e * ta * w
        H = P1 * P2 + P3 * P4;


        return H
