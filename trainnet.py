
#%%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

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
predictor_offset = predictor.mean(axis=0)
predictor -= predictor_offset
predictor_scale = predictor.var(axis=0)
predictor /= predictor_scale

response_offset = response.mean(axis=0);
response -= response_offset;
response_scale = response.var(axis=0);
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
    test_indices = pool[np.random.choice(pool.size, math.max(pool.size, num_test))]

    return train_indices, val_indices, test_indices


#%% Build Model
print('Build Model')

model = tf.keras.Sequential()
inputs = keras.Input(shape=(1,), name='input')
x = tf.keras.layers.Dense(512, activation='relu', name='dense_1')(inputs)
x = tf.keras.layers.Dense(512, activation='relu', name='dense_2')(x)
x = tf.keras.layers.Dense(512, activation='relu', name='dense_3')(x)
x = tf.keras.layers.Dense(512, activation='relu', name='dense_4')(x)
outputs = tf.keras.layers.Dense(40, name='output')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
    
# compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10E-3),
    loss=keras.losses.mean_squared_error,
    metrics=[])

model.summary()

#%% randomize test /split

# fit / train split
train_split  = 0.3;
test_split   = 0.3;
val_split    = 0.4;

train_idx, val_idx, test_idx = split_train_test_eval_indices(num_samples, train_split, val_split, test_split)



#%% train network
print('Train Model')

logs = "logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs, histogram_freq = 1)

model.fit(x=predictor[train_val_indices,:], y=response[train_val_indices,:], epochs = 500, 
          batch_size=400000,
          validation_split=val_split / (1-test_split))
          
#%% predict data and plot
print("Log Results")

predicted = model.predict(predictor)

pred_scaled = predicted * response_scale + response_offset;
resp_scaled = response  * response_scale + response_offset;

error_test  = np.mean(np.mean(np.abs(pred_scaled[test_indices,:] - resp_scaled[test_indices,:])))

error_train = np.mean(np.mean(np.abs(pred_scaled[train_val_indices,:] - resp_scaled[train_val_indices,:])))

print(f' Test  Error: {error_test:.2f}')
print(f' Train Error: {error_train:.2f}')



# %%
