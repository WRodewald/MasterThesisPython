
#%%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import math
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time


from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

#device_name = tf.test.gpu_device_name()
#if device_name != '/device:GPU:0':
#  raise SystemError('GPU device not found')
#print('Found GPU at: {}'.format(device_name))


# %% 

print('Import Dataset')

dataset = "data/data.csv"

data = pd.read_csv(dataset, sep=";")

predictor = data.iloc[:,(0)]
response  = data.iloc[:,1:41]
predictor = predictor.to_numpy()
response = response.to_numpy()

predictor = predictor[:,None]

# manual normalization
predictor_offset = np.min(predictor);
predictor_scale  = np.max(predictor) - np.min(predictor);
predictor -= predictor_offset
predictor /= predictor_scale

response_offset = np.min(response);
response_scale  = np.max(response) - np.min(response);
response -= response_offset;
response /= response_scale;

num_samples = response.shape[0]

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

from tools import ZPKToMagLayer

inputs = keras.Input(shape=(1,), name='input')
x = tf.keras.layers.Dense(10,  activation='sigmoid', name='dense_1')(inputs)
x = tf.keras.layers.Dense(40,  activation='sigmoid', name='dense_2')(x)
x = tf.keras.layers.Dense(80,  activation='sigmoid', name='dense_3')(x)
x = tf.keras.layers.Dense(160, activation='sigmoid', name='dense_4')(x)
x = tf.keras.layers.Dense(40,  activation='sigmoid', name='dense_5')(x)
x = tf.keras.layers.Dense(41,  activation='sigmoid', name='dense_6')(x)
x = ZPKToMagLayer.ZPKToMagLayer(44100, 40, name='zpk')([x,inputs])
model = keras.Model(inputs=inputs, outputs=x)
  
# compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10E-4),
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

start = time();
model.fit(x=predictor[train_idx,:], y=response[train_idx,:], epochs = 100, 
          batch_size=40024,
          validation_data=(predictor[val_idx,:], response[val_idx,:]),
          callbacks=[])
print(time() - start)
          
#%% predict data and plot
print("Log Results")

predicted = model.predict(predictor)

pred_scaled = predicted * response_scale + response_offset;
resp_scaled = response  * response_scale + response_offset;

error_test  = np.mean(np.mean(np.abs(pred_scaled[train_idx,:] - resp_scaled[train_idx,:])))
error_val   = np.mean(np.mean(np.abs(pred_scaled[val_idx,:]   - resp_scaled[val_idx,:])))
error_train = np.mean(np.mean(np.abs(pred_scaled[test_idx,:]  - resp_scaled[test_idx,:])))

print(f' Test  Error:      {error_test:.2f}')
print(f' Validation Error: {error_val:.2f}')
print(f' Train Error:      {error_train:.2f}')




# %%


plt.plot(pred_scaled[500,:])
plt.plot(resp_scaled[500,:])
plt.show()
