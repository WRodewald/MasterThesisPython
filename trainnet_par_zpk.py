
#%%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#%reset

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import math
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time

from tools import MagToDBLayer, ParZPKToMagLayer, NormalizeLayer 

from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# %% 

print('Import Dataset')

dataset = "data/data.csv"

data = pd.read_csv(dataset, sep=";")

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
x = tf.keras.layers.Dense(128,  activation='relu', name='dense_1')(x)
x = tf.keras.layers.Dense(128,  activation='relu', name='dense_2')(x)
x = tf.keras.layers.Dense(256,  activation='relu', name='dense_3')(x)
x = tf.keras.layers.Dense(50,  name='dense_5')(x)
x = ParZPKToMagLayer.ParZPKToMagLayer(sample_rate, 40, name='zpk')([x,inputs])
x = MagToDBLayer.MagToDBLayer(name='mag2db')(x)
model = keras.Model(inputs=inputs, outputs=x)
  
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

start = time();
model.fit(x=predictor[train_idx,:], y=response[train_idx,:], epochs = 5000, 
          batch_size=40000,
          validation_data=(predictor[val_idx,:], response[val_idx,:]),
          callbacks=[])
print(time() - start)
          
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


# %%
layer_output = model.get_layer('dense_5').output

intermediate_model = tf.keras.models.Model(inputs=model.input,outputs=layer_output)
 
zpk_params = intermediate_model.predict(predictor)

num_instances = zpk_params.shape[0]
num_bins = 2000
f0 = 4.
layer = ParZPKToMagLayer.ParZPKToMagLayer(sample_rate, num_bins)

f = tf.tile(tf.reshape(f0, [1,1]), [num_instances, 1])

# skip some frames

#for i in range(0, num_instances):
#    print(i)
#    response = layer.call([tf.slice(zpk_params, [i,0],[1,-1]), tf.slice(f, [i,0],[1,-1])])
#    response = MagToDBLayer.MagToDBLayer().call(response)


# %%
