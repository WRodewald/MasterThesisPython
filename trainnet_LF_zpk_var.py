
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


#%% Build Model
print('Build Model')

inputs = keras.Input(shape=(1,), name='input')
x = inputs
x = NormalizeLayer.NormalizeLayer(predictor_mean, predictor_std, name='normalize')(x)
x = tf.keras.layers.Dense(128,  activation='sigmoid')(x)
x = tf.keras.layers.Dense(128,  activation='sigmoid')(x)
x = tf.keras.layers.Dense(128,  activation='sigmoid')(x)
x = tf.keras.layers.Dense(128,  activation='sigmoid')(x)
x = tf.keras.layers.Dense(42)(x)

# split in LF-Rd and zpk branch
def split_layers(input): 
    return tf.expand_dims(input[:,0],1), input[:, 1:]

x_Rd, x_zpk = tf.keras.layers.Lambda(split_layers)(x)
# LF-Rd branch
def scale_Rd(input): 
    return util.lin_scale(tf.sigmoid(input), 0., 1., 0.3, 2.7)

x_Rd = tf.keras.layers.Lambda(scale_Rd, name="Rd")(x_Rd)
x_Rd = LFRdToDBLayer.LFRdToDBLayer(40, name="RdOut")(x_Rd)

# Vocal tract branch
var_layer = VarianceLayer.VarianceLayer(predictor.shape[0])
x_zpk = var_layer(x_zpk)
x_zpk, _, _, _ = ZPKToMagLayer.ZPKToMagLayer(sample_rate, 40, name='zpk')([x_zpk,inputs])
x_zpk = MagToDBLayer.MagToDBLayer(name='mag2db')(x_zpk)

# sum branch decibel magnitude responses 
x = tf.keras.layers.Add()([x_Rd, x_zpk])

model = keras.Model(inputs=inputs, outputs=[x])
  

#%% compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10E-4),
    loss=keras.losses.mean_squared_error,
    metrics=[])

model.summary()


#%% train network
print('Train Model')

logs = "logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs, histogram_freq = 1)

model.fit(x=predictor, y=response, 
          shuffle=False,
          epochs = 2000, 
          batch_size=40000,
          callbacks=[])


#%%

# create new model, skipping unnecessary prediction 
zpk_output, z0_out, p0_out, k_out = model.get_layer('zpk').output
RdOut_out = model.get_layer('RdOut').output
Rd_out = model.get_layer('Rd').output
zpk_model = tf.keras.models.Model(inputs=model.input, outputs=[z0_out, p0_out, k_out, Rd_out, RdOut_out])

[z0_pred, p0_pred, k_pred, Rd_pred, RdOut_pred] = zpk_model.predict(predictor, batch_size=40000)


# store as .mat file to further analyse in matlab

sio.savemat('out.mat', {
    'f':predictor,
    'Rd':Rd_pred,
    'RdOut':RdOut_pred,
    'z0':z0_pred, 
    'p0': p0_pred, 
    'k':k_pred,
    'pred':predictor,
    #'var':var_layer.variance.numpy(),
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
