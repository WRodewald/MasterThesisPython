
#This script loads the pretrained synthesis-parmeter prediction model and uses it to first predict
# synthesis parameters and afterwards run them through synthesis.

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


#%% Load Model
print('Load Model')

model_path = 'data/example/prediction_model'
model = tf.keras.models.load_model(model_path, compile=False)

# redefining the loss as tensorflow ensists on it

gain_weight = 0.1 # x/dB
Rd_weight   = 1.   # x/Rd
w_weight    = 0.01 * 44100/(2.*np.pi)  # x/Hz
r_weight    = 0.1 # x/dB  

def mse_loss(y_true, y_pred):
    return tf.square(gain_weight * (y_true - y_pred))


losses = [lambda y_true, y_pred: util.weighted_mse_loss(y_true, y_pred, tf.square(gain_weight)), 
          lambda y_true, y_pred: util.weighted_mse_loss(y_true, y_pred, tf.square(Rd_weight)), 
          lambda y_true, y_pred: util.pole_zero_loss(y_true, y_pred, r_weight, w_weight),
          lambda y_true, y_pred: util.pole_zero_loss(y_true, y_pred, r_weight, w_weight)]

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10E-4),
    loss=losses)

#%% Load Data

# load original singer
wav_file_s1  = dataset.get_sample('scales', 'slow_forte', 'a', 'f6')[0]
json_file_s1 = os.path.splitext(wav_file_s1)[0] + '.json'
audio_s1 = FramedAudio.from_json(json_file_s1)

pitch_s1 = audio_s1.get_trajectory('pitch')

# load other singer
wav_file_s2  = dataset.get_sample('scales', 'slow_forte', 'o', 'f6')[0]
json_file_s2 = os.path.splitext(wav_file_s2)[0] + '.json'
audio_s2 = FramedAudio.from_json(json_file_s2)

pitch_s2 = audio_s2.get_trajectory('pitch')

fs = 44100

#%% Synthesis Parameter Prediction
print('Predict Synthesis Parameters')

f0 = pitch_s2
g, Rd, p0, z0 = model.predict(f0, batch_size=40000)

# %% Prepare synthesis parameters (oversample, reshape)
print('Prepare Synthesis Parameters')

num_poles   = p0.shape[1]
num_samples = f0.shape[0]

osc = RdOscillator.RdOscillator(num_instances=64, num_samples=512)
filter = SerialFilterBank.SerialFilterBank(num_poles);

fs = 44100
hop_size = 64
t_frame = hop_size * np.arange(0, num_samples)
t_audio = np.linspace(0, t_frame[-1], num_samples * hop_size);

f0_interpolator = interpolate.interp1d(t_frame, np.squeeze(f0), axis=0)
g_interpolator  = interpolate.interp1d(t_frame, np.squeeze(g),  axis=0)
Rd_interpolator = interpolate.interp1d(t_frame, np.squeeze(Rd), axis=0)
p0_interpolator = interpolate.interp1d(t_frame, np.squeeze(p0), axis=0)
z0_interpolator = interpolate.interp1d(t_frame, np.squeeze(z0), axis=0)


#%% Synthesis
print('Synthesis. This might take a bit...')

# oscillator
audio = osc.tick(Rd_interpolator(t_audio), f0_interpolator(t_audio), fs)

# gain
audio *= util.db2mag(g_interpolator(t_audio)).numpy()

# filter with per sample pole/zero updates
audio = filter.tick_pz(audio, p0_interpolator(t_audio), z0_interpolator(t_audio))

#%% so
soundfile.write('data/example/audio/singer_alt_synthesis.wav', 0.707 * audio / np.max(np.abs(audio)), fs)



# %%
