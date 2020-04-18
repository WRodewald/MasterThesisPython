
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

from model import MagToDBLayer, ZPKToMagLayer, NormalizeLayer 

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

if(False):
    dataset_csv = "data/data.csv"
    data = pd.read_csv(dataset_csv, sep=";")

    predictor = data.iloc[:,(0)]
    response  = data.iloc[:,1:41]
    predictor = predictor.to_numpy()
    response  = response.to_numpy()

    predictor = predictor[:,None]

    # manual normalization
    predictor_mean = np.mean(predictor)
    predictor_std  = np.std(predictor)

    num_samples = response.shape[0]
    sample_rate = 44100


if(True):
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



def split_train_test_eval_indices(num_samples, train_split, val_split, test_split):

    num_train = np.floor(train_split * num_samples).astype(int)
    num_val   = np.floor(val_split   * num_samples).astype(int)
    num_test  = np.floor(test_split  * num_samples).astype(int)

    pool = np.arange(0, num_samples)
    train_indices = pool[np.random.choice(pool.size, num_train)]
    
    pool = np.setdiff1d(pool, train_indices)
    val_indices = pool[np.random.choice(pool.size, num_val)]

    pool = np.setdiff1d(pool, val_indices)
    test_indices = pool[np.random.choice(pool.size, max(pool.size, num_test))]

    return train_indices, val_indices, test_indices


#%% Build Model
print('Build Model')

inputs = keras.Input(shape=(1,), name='input')
x = inputs
x = NormalizeLayer.NormalizeLayer(predictor_mean, predictor_std, name='normalize')(x)
x = tf.keras.layers.Dense(128,  activation='sigmoid')(x)
x = tf.keras.layers.Dense(128,  activation='sigmoid')(x)
x = tf.keras.layers.Dense(128,  activation='sigmoid')(x)
x = tf.keras.layers.Dense(128,  activation='sigmoid')(x)
x = tf.keras.layers.Dense(41)(x)
x, _, _, _ = ZPKToMagLayer.ZPKToMagLayer(sample_rate, 40, name='zpk')([x,inputs])
x = MagToDBLayer.MagToDBLayer(name='mag2db')(x)
model = keras.Model(inputs=inputs, outputs=[x])
  
# compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10E-5),
    loss=keras.losses.mean_squared_error,
    metrics=[])

model.summary()



#%% randomize test /split

# fit / train split
train_split  = 0.7;
test_split   = 0.15;
val_split    = 0.15;

train_idx, val_idx, test_idx = split_train_test_eval_indices(num_samples, train_split, val_split, test_split)



#%% train network
print('Train Model')

logs = "logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs, histogram_freq = 1)

model.fit(x=predictor[train_idx,:], y=response[train_idx,:], epochs = 2000, 
          batch_size=40000,
          validation_data=(predictor[val_idx,:], response[val_idx,:]),
          callbacks=[])


#%%
#model.save('temp/zpk_model') 

sweep = np.linspace(np.min(predictor), np.max(predictor), 1000)
sweep_out = model.predict(sweep)

np.savetxt("out.csv", sweep_out)


# create new model, skipping unnecessary prediction 
zpk_output, z0_out, p0_out, k_out = model.get_layer('zpk').output
zpk_model = tf.keras.models.Model(inputs=model.input, outputs=[z0_out, p0_out, k_out])

[z0, p0, k] = zpk_model.predict(predictor)


# store as .mat file to further analyse in matlab

sio.savemat('out.mat', {
    'f':predictor, 
    'z0':z0, 
    'p0': p0, 
    'k':k,
    'pred':predictor,
    'resp':response})


          
#%% predict data and plot
print("Log Results")

predicted = model.predict(predictor)

error_test  = np.mean(np.mean(np.abs(predicted[train_idx,:] - response[train_idx,:])))
error_val   = np.mean(np.mean(np.abs(predicted[val_idx,:]   - response[val_idx,:])))
error_train = np.mean(np.mean(np.abs(predicted[test_idx,:]  - response[test_idx,:])))

print(f' Test  Error:      {error_test:.2f}')
print(f' Validation Error: {error_val:.2f}')
print(f' Train Error:      {error_train:.2f}')



# %%


plt.plot(predicted[500,:])
plt.plot(response[500,:])
plt.show()

