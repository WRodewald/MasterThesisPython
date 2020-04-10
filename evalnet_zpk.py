#%%
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from time import time

# %% 

print('Import Dataset')

dataset = "data/data.csv"

data = pd.read_csv(dataset, sep=";")

predictor = data.iloc[0:8000,(0)]
response  = data.iloc[0:8000,1:41]
predictor = predictor.to_numpy()
response  = response.to_numpy()

predictor = predictor[:,None]

num_samples = response.shape[0]
sample_rate = 44100

num_overtones = response.shape[1]

#%%
model= tf.keras.models.load_model('temp/zpk_model')

# Check its architecture
model.summary()


# %%
# predict 

f0 = np.squeeze(predictor)
f0_t = np.arange(0, predictor.size, 1)
f0_t_interpolated = np.arange(0, predictor.size-1, 1/64)

f0_interpolated = interpolate.interp1d(f0_t, f0, axis=0, kind='cubic')(f0_t_interpolated)

start = time()
overtonesDB = model.predict(f0_interpolated)
overtonesDB = overtonesDB - np.max(overtonesDB) - 6
overtonesMag = np.power(10., overtonesDB/20.)
print(time() - start)

# %%


phases = np.zeros([1, num_overtones])
k = np.reshape(np.arange(1, num_overtones+1,), [1, num_overtones])

synth = np.zeros(f0_interpolated.shape)

for i in range(0, f0_interpolated.size):

    phase_inc = f0_interpolated[i]/sample_rate * k
    phases = phases + phase_inc
    phases = phases - (phases >= 1).astype(float)
    mag = overtonesMag[i,:]

    sines = mag * np.sin(2. * np.pi * phases)

    sample = np.sum(sines)

    synth[i] = sample


# %%
