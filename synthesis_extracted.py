# This scripts loads the extracted sythesis parameters and uses the in synthesis 
# to valdiate the resynthesis performance

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
print('import data')

mat_file = 'out.mat'
mat = sio.loadmat(mat_file)

f0 = mat['f']
Rd = mat['Rd']
p0 = mat['p0']
z0 = mat['z0']
g  = mat['g']

fs = 44100


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
soundfile.write('data/example/audio/extracted_synthesis.wav', 0.707 * audio / np.max(np.abs(audio)), fs)




# %%
