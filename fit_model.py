
#%%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#%reset

%load_ext autoreload
%autoreload 2

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import math
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time

from model import MagToDBLayer, ZPKToMagLayer, NormalizeLayer, LFRdToDBLayer, util, VarianceLayer
from model import SliceLayer
from model import TemporalVarianceLayer
from model import ParameterLayer

from datetime import datetime
from packaging import version

import scipy.io as sio

import tensorflow as tf
from tensorflow import keras

from tools.framed_audio import FramedAudio
from tools import extract_overtones, magnitude, dataset

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# %% 

print('Import Dataset')

if(True):
    dataset_csv = "data/data.csv"
    data = pd.read_csv(dataset_csv, sep=";")

    predictor = data.iloc[:,(0)]
    response  = data.iloc[:,1:41]
    predictor = predictor.to_numpy()
    response  = response.to_numpy()
    
    predictor = predictor[:,None]
        
    predictor = predictor[0:9000,:]
    response  = response[0:9000,:]

    # manual normalization
    predictor_mean = np.mean(predictor)
    predictor_std  = np.std(predictor)

    num_samples = response.shape[0]
    sample_rate = 44100


if(False):
    # get cached analysis result and source sample
    num_overtones = 40
    
    wav_file  = dataset.get_sample('scales', 'slow_forte', 'a', 'f1')[0]
    json_file =  os.path.splitext(wav_file)[0] + '.json'

    audio = FramedAudio.from_json(json_file)

    onset, offset = magnitude.get_onset_offset(audio)
    sample_rate = audio.fs
    num_samples = offset - onset

    overtones = extract_overtones.extract_overtones_from_audio(audio, num_overtones = num_overtones)
    overtones = 20 * np.log10(np.abs(overtones))
    overtones = overtones[onset:offset, :]
    pitch = np.reshape(audio.get_trajectory('pitch')[onset:offset], [num_samples,1])

    predictor = pitch
    response  = overtones

    predictor_mean = np.mean(predictor)
    predictor_std  = np.std(predictor)

num_overtones = 40
num_samples   = predictor.shape[0]

#%% Build Model
print('Build Model')

inputs = keras.Input(shape=(1,), name='input')

parameter_layer = ParameterLayer.ParameterLayer(num_samples, 42)
x = parameter_layer(inputs)


# split in LF-Rd and zpk branch
x_Rd, x_gain, x_zpk = SliceLayer.SliceLayer([[0,1], [1,2], [2,62]])(x)

zpk_var  = TemporalVarianceLayer.TemporalVarianceLayer(num_samples = num_samples, weight=10E6) 
Rd_var   = TemporalVarianceLayer.TemporalVarianceLayer(num_samples = num_samples, weight=10E6) 
gain_var = TemporalVarianceLayer.TemporalVarianceLayer(num_samples = num_samples, weight=10E6) 
x_zpk   = zpk_var(x_zpk)
x_Rd    = Rd_var(x_Rd)
x_gain  = gain_var(x_gain)

# LF-Rd branch
def scale_Rd(input): 
    return util.lin_scale(tf.sigmoid(input), 0., 1., 0.3, 2.7)
    
x_Rd = tf.keras.layers.Lambda(scale_Rd, name="Rd")(x_Rd)
x_Rd = LFRdToDBLayer.LFRdToDBLayer(num_overtones, name="RdOut")(x_Rd)

# gain branch as lamba layer
x_gain = tf.keras.layers.Lambda(lambda x: util.lin_scale(x, 0., 1., -100, 0), name="gain")(x_gain)
x_gain = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, num_overtones]))(x_gain)

# Vocal tract branch
x_zpk, _, _, k = ZPKToMagLayer.ZPKToMagLayer(sample_rate, num_overtones, name='zpk')([x_zpk,inputs])
x_zpk = MagToDBLayer.MagToDBLayer(name='mag2db')(x_zpk)

# sum branch decibel magnitude responses 
x = tf.keras.layers.Add()([x_Rd, x_zpk, x_gain])


model = keras.Model(inputs=inputs, outputs=[x])
  

#%% compile model

gain_var.weight = 10E6
zpk_var.weight  = 10E8
Rd_var.weight   = 10E6

mse_weights = np.reshape(np.linspace(1., 0, num_overtones), [1,num_overtones])
mse_weights = mse_weights / np.average(mse_weights) 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10E-5),
    loss= lambda y_true, y_pred: util.weighted_mse_loss(y_true, y_pred, mse_weights),
    metrics=[])

model.summary()



#%% train network
print('Train Model')

logs = "logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs, histogram_freq = 1)

model.fit(x=predictor, y=response, 
          shuffle=False,
          epochs = 4000, 
          batch_size=40000,
          callbacks=[])


#%%

# create new model, skipping unnecessary prediction 
zpk_output, z0_out, p0_out, k_out = model.get_layer('zpk').output
RdOut_out = model.get_layer('RdOut').output
Rd_out    = model.get_layer('Rd').output
gain_out  = model.get_layer('gain').output
zpk_model = tf.keras.models.Model(inputs=model.input, outputs=[z0_out, p0_out, k_out, Rd_out, RdOut_out, gain_out])

[z0_pred, p0_pred, k_pred, Rd_pred, RdOut_pred, g_pred] = zpk_model.predict(predictor, batch_size=40000)


# store as .mat file to further analyse in matlab

sio.savemat('out.mat', {
    'f':predictor,
    'g':g_pred,
    'Rd':Rd_pred,
    'RdOut':RdOut_pred,
    'z0':z0_pred, 
    'p0': p0_pred, 
    'k':k_pred,
    'pred':predictor,
    #'var':zpk_variance.variance.numpy(),
    'resp':response})


          
#%% predict data and plot
print("Log Results")

predicted = model.predict(predictor, batch_size=40000)

error_test  = np.mean(np.mean(np.abs(predicted - response)))

print(f' Test  Error:      {error_test:.2f}')



# %%


plt.plot(predicted[800,:])
plt.plot(response[800,:])
plt.show()



# %%
