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

# matlab export 
import scipy.io as sio

# tensorflow and tensorflow.keras
import tensorflow as tf
from tensorflow import keras

# custom network
from model import LFRdLayer
from model import ZPKToMagLayer
from model import TemporalVarianceLayer
from model import ParameterLayer
from model import BSplineLayer
from model import util

# audio management
from tools.framed_audio import FramedAudio
from tools import extract_overtones, magnitude, dataset

# enable os.environ line to disable GPU support 
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# allowing growth on GPU 
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

#%% Import Dataset
print('Import Dataset')

# get cached analysis result and source sample
num_overtones = 40

wav_file  = dataset.get_sample('scales', 'slow_forte', 'a', 'f6')[0]
json_file = os.path.splitext(wav_file)[0] + '.json'

# import framed audio 
audio = FramedAudio.from_json(json_file)
sample_rate = audio.fs

# run onset / offset detection
onset, offset = magnitude.get_onset_offset(audio)
onset  += 100
offset -= 100
num_samples = offset - onset

#run overtone extraction
overtones = extract_overtones.extract_overtones_from_audio(audio, num_overtones = num_overtones)
overtonesDB = 20. * np.log10(np.abs(overtones[onset:offset, :]))
overtonesDB -= np.max(overtonesDB) 

#extract pitch information
pitch = np.reshape(audio.get_trajectory('pitch')[onset:offset], [num_samples,1])

# store predictor and response
predictor = pitch
response  = overtonesDB

print('Done importing data.')

#%% Build Model
print('Build Model')

#parameters 
num_vt_parameters  = 4 * 10
pz_b_spline_size  = math.ceil(num_samples / 200)
Rd_b_spline_size   = math.ceil(num_samples / 10)
gain_b_spline_size = math.ceil(num_samples / 10)

# clear seesion to prevent clutter from previously compiled models
tf.keras.backend.clear_session()

# dummy input used for ZPK layer to have access to pitch
inputs = keras.Input(shape=(1,), name='input')

# pole/zero (pz) branch
var_pz = TemporalVarianceLayer.TemporalVarianceLayer(num_samples = pz_b_spline_size, weight=10E8) 
x_pz = ParameterLayer.ParameterLayer(pz_b_spline_size, num_vt_parameters, initial_value=0.)(inputs) 
x_pz = var_pz(x_pz)
x_pz = BSplineLayer.BSplineLayer(pz_b_spline_size, num_samples)(x_pz)

x_pz, _, _, _ = ZPKToMagLayer.ZPKToMagLayer(sample_rate, num_overtones, name='PZ')([x_pz, inputs])
x_pz = util.MagToDBLayer()(x_pz)

# Rd branch
var_Rd = TemporalVarianceLayer.TemporalVarianceLayer(num_samples = Rd_b_spline_size, weight=10E8) 
x_Rd = ParameterLayer.ParameterLayer(Rd_b_spline_size, 1, initial_value=0.7)(inputs)
x_Rd = var_Rd(x_Rd)
x_Rd = BSplineLayer.BSplineLayer(Rd_b_spline_size, num_samples)(x_Rd)
x_Rd = tf.keras.layers.Lambda(lambda x: util.lin_scale(tf.sigmoid(x), 0., 1., 0.3, 2.7), name="Rd")(x_Rd)
x_Rd = LFRdLayer.LFRdLayer(num_overtones, name="RdOut")(x_Rd)
x_Rd = util.MagToDBLayer()(x_Rd)

# gain branch
var_gain = TemporalVarianceLayer.TemporalVarianceLayer(num_samples = gain_b_spline_size, weight=10E8) 
x_gain = ParameterLayer.ParameterLayer(gain_b_spline_size, 1, initial_value=0.5)(inputs)
x_gain = var_gain(x_gain)
x_gain = BSplineLayer.BSplineLayer(gain_b_spline_size, num_samples)(x_gain)
x_gain = tf.keras.layers.Lambda(lambda x: util.lin_scale(x, 0., 1., -100, 0), name="gain")(x_gain) # gain scaling
x_gain = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, num_overtones]))(x_gain) # tiling to fit dimensions

# gain layer to "hide" Rd branch ininitially to improve fitting
Rd_gain = util.GainLayer()
x_Rd = Rd_gain(x_Rd)

# model prediction H = H_vt * H_gf * gain or equivalent addition in decibel domain
x = tf.keras.layers.Add()([x_pz, x_gain, x_Rd])

# define network model
model = keras.Model(inputs=inputs, outputs=[x])
  
print('Done building model.')

#%% Compile Model
print('Compile Model')

# variance loss weights for pole/zero, gain and Rd parameters
var_pz.weight  = 10E4
var_gain.weight = 10E2
var_Rd.weight   = 10E2

# set Rd mix increment per epoche
Rd_gain.increment.assign(1./4000.)

# constructing a weight matrix with a -6dB/oct lowpass
fc = 2000.
fk = predictor * np.arange(1, 41) 
weights = tf.cast(tf.abs(1. / (1. + 1j * fk/fc)), dtype='float32')
weights /=tf.reduce_mean(tf.reduce_mean(weights, axis=1))

#compile model with learning rate and loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10E-4),
    loss= lambda y_true, y_pred: util.weighted_mse_loss(y_true, y_pred, weights))

#print model summary
model.summary()

print('Done compiling model.')


#%% train model
print('Train Model')

# log directory for tensorboard
logs = "logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs, histogram_freq = 1)

model.fit(x=predictor, y=response, 
          shuffle=False,
          epochs = 4000, 
          batch_size=40000,
          callbacks=[])


#%% Export .mat file
print('Export .mat file')

# collect all parameters we want to explort
_, z0_out, p0_out, _ = model.get_layer('PZ').output
RdOut_out = model.get_layer('RdOut').output
Rd_out    = model.get_layer('Rd').output
gain_out  = model.get_layer('gain').output
zpk_model = tf.keras.models.Model(inputs=model.input, outputs=[z0_out, p0_out, Rd_out, RdOut_out, gain_out])

[z0_pred, p0_pred, Rd_pred, RdOut_pred, gain_pred] = zpk_model.predict(predictor, batch_size=40000)

# store as .mat file to further analyse in matlab

sio.savemat('out.mat', {
    'f':predictor,          # pitch
    'g':gain_pred,          # gain
    'Rd':Rd_pred,           # Rd parameter
    'RdOut':RdOut_pred,     # Rd decibel magnitude response
    'z0':z0_pred,           # filter zeros
    'p0': p0_pred,          # filter poles
    'pred':predictor,       # pitch (predictor)
    'resp':response})       # response:  



# %%
