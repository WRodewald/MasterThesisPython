#%%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#%reset
%load_ext autoreload
%autoreload 2

import math
import functools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import interpolate

import sounddevice
import soundfile
# matlab export 
import scipy.io as sio

# tensorflow and tensorflow.keras
import tensorflow as tf
from tensorflow import keras

# custom network
from model import SliceLayer
from model import util

# audio management
from tools.framed_audio import FramedAudio
from tools import extract_overtones, magnitude, dataset
from tools import RdOscillator, SerialFilterBank

# enable os.environ line to disable GPU support 
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# allowing growth on GPU 
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


#%% import data
print('import data')

mat_file = 'out.mat'
mat = sio.loadmat(mat_file)

f0 = mat['f']
Rd = mat['Rd']
p0 = mat['p0']
z0 = mat['z0']
g  = mat['g']

fs = 44100

num_samples = f0.shape[0]
num_poles = p0.shape[1]
num_zeros = z0.shape[1]

p0 = tf.reshape(p0, shape=[num_samples, num_poles])
z0 = tf.reshape(z0, shape=[num_samples, num_poles])


# %%

def to_complex(x):
    r = tf.sigmoid(x[:,0::2]) # radius limited to [0..1]
    w = np.pi * tf.sigmoid(x[:,1::2]) # angle limited to [0..pi]
    return tf.complex(r, 0.) * tf.exp(tf.complex(0., w))

inputs = keras.Input(shape=(1,), name='input')
x = inputs
x = tf.keras.layers.Lambda(lambda x: x/100.)(x)
x = tf.keras.layers.Dense(1)(x)
x = tf.keras.layers.Dense(8,   activation='sigmoid')(x)
x = tf.keras.layers.Dense(16,  activation='sigmoid')(x)
x = tf.keras.layers.Dense(32,  activation='sigmoid')(x)
x = tf.keras.layers.Dense(62,  activation='sigmoid')(x)
x = tf.keras.layers.Dense(62,  activation='sigmoid')(x)
x = tf.keras.layers.Dense(62,  activation='sigmoid')(x)
x = tf.keras.layers.Dense(62,  )(x)

x_gain, x_Rd, x_p0, x_z0 = SliceLayer.SliceLayer(slice_lens = [1, 1, 2*num_poles, 2*num_zeros])(x)
x_gain = tf.keras.layers.Lambda(lambda x: 100.*x, name='Gain')(x_gain)
x_Rd   = tf.keras.layers.Lambda(lambda x:   1.*x, name='Rd')(x_Rd)
x_p0   = tf.keras.layers.Lambda(to_complex, name='p0')(x_p0)
x_z0   = tf.keras.layers.Lambda(to_complex, name='z0')(x_z0)

model = tf.keras.Model(inputs=[inputs], outputs=[x_gain, x_Rd, x_p0, x_z0])

gain_weight = 0.1 # x/dB
Rd_weight   = 1.   # x/Rd
w_weight    = 0.01 * fs/(2.*np.pi)  # x/Hz
r_weight    = 0.1 # x/dB  

def mse_loss(y_true, y_pred):
    return tf.square(gain_weight * (y_true - y_pred))


losses = [lambda y_true, y_pred: util.weighted_mse_loss(y_true, y_pred, tf.square(gain_weight)), 
          lambda y_true, y_pred: util.weighted_mse_loss(y_true, y_pred, tf.square(Rd_weight)), 
          lambda y_true, y_pred: util.pole_zero_loss(y_true, y_pred, r_weight, w_weight),
          lambda y_true, y_pred: util.pole_zero_loss(y_true, y_pred, r_weight, w_weight)]

#%%
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10E-5),
    loss=losses)

model.summary()


#%% Training
print('Training Model')

model.fit(x=f0, y=[g, Rd, p0, z0], 
          epochs = 4000, 
          batch_size=20000,
          callbacks=[])

#%% Plot predictions

g_pred, Rd_pred, p0_pred, z0_pred = model.predict(f0, batch_size=20000)



#%% Store Model
print('Store Model')

model_path = 'data/example/prediction_model'
model.save(model_path)

# %%


