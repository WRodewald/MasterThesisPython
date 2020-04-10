#%%
from tools import audio_io
from tools.framed_audio import FramedAudio
from tools import pitch_optimize
import time, sys

from tools import dataset

from tools import extract_overtones
from tools import pitch_acf

import numpy as np

from matplotlib import pyplot as plt

import tensorflow as tf

import crepe.core

import warnings

from tqdm import tqdm
from scipy import interpolate

#%%
# config
hop_size = 256
block_size = 2048

rerun_analysis = False

config = {
    'centered':True,
    'block-size':block_size,
    'hop-size':hop_size
}

# dataset & audio
vocalset_root = dataset.get_root_path()
src_file = vocalset_root + '/female4/scales/slow_forte/f4_scales_c_slow_forte_o.wav'

json_tmp = 'build/sample.json'

# init audio object, repeat 
audio = FramedAudio.from_json(json_tmp)

print('Found no cached audio or audio with outdated conig. Will re-run analysis')

framed_audio = FramedAudio.from_file(src_file,  config=config)
    
# run crepe
step_size = 5
crepe_time, crepe_pitch, confiidence, _ = crepe.predict(audio.get_raw(), audio.fs, step_size = step_size, center=True)


# interpolate results from crepe
crepe_interpolator = interpolate.interp1d(crepe_time, crepe_pitch)
interp_time = framed_audio.get_time( centered=True)

crepe_pitch_interp = crepe_interpolator(interp_time)

pitch_estimate = crepe_pitch_interp

#%%

N  = framed_audio.get_num_frames()
N0 = 0
fs = framed_audio.fs

# run acf pitch estimate
if pitch_estimate is None: 
    pitch_estimate = pitch_acf.pitch_acf(framed_audio,fs)


num_overtones = 15

block_size = framed_audio.block_size
sample_rate = framed_audio.fs

# output vectors
pitch     = pitch_estimate[N0:(N0+N)]
pitch_inc = np.zeros(N)

pitch_estimate =  tf.reshape(tf.cast(pitch_estimate, dtype='float32'), [N,1,1])

pitch     = tf.reshape(pitch, [N,1,1])
pitch_inc = tf.reshape(pitch_inc, [N,1,1])

pitch     = tf.Variable(tf.cast(pitch, dtype='float32'), name='pitch')
pitch_inc = tf.Variable(tf.cast(pitch_inc, dtype='float32'), name='pitch_inc')

# extract frames prior
frames = np.zeros([N, block_size])
for i in range(0, N):
    frames[i,:] = framed_audio.get_frame(N0+i)

frames = tf.constant(frames, dtype='float32')
frames = tf.reshape(frames, [N, block_size, 1])
window = tf.reshape(tf.signal.hann_window(block_size), [1,block_size,1])

k = tf.reshape(tf.range(1,num_overtones+1, dtype='float32'), [1,1,num_overtones])

t = (tf.range(0, block_size, dtype='float32') - 0.5 * block_size) / sample_rate
t = tf.reshape(t, [1,block_size,1])


#@tf.function
def error_rms():

    phase = pitch * t + 0.5 * pitch_inc * t * t
    div   = tf.exp(tf.complex(0., 2. * np.pi * k * phase))
 
    ot = tf.reduce_sum(tf.complex((frames * window), 0.) / div,1, keepdims=True) / block_size
    
    mag = tf.math.abs(ot)
    ang = tf.math.angle(ot)

    #resynthesis
    resynth = 2. * mag * tf.math.cos(2. * np.pi * k * phase + ang)
    resynth = 2 * tf.reduce_sum(resynth, 2, keepdims=True)

    error = (resynth * window - frames * window)
    error_rms = tf.squeeze(tf.sqrt(tf.reduce_mean(error * error, 1)))

    plt.plot(resynth[500,:,0])
    plt.plot(frames[500,:,0])
    return error_rms


#%%
import tensorflow_probability as tfp
pitch.assign(pitch_estimate[0:N])
pitch_inc.assign(tf.zeros(pitch_inc.shape))

start = time.time()

err = error_rms();
print(np.mean(err))

tfp.math.minimize(error_rms,
                num_steps=200,
                optimizer=tf.optimizers.RMSprop(learning_rate=1))
err = error_rms();
print(np.mean(err))

tfp.math.minimize(error_rms,
                num_steps=200,
                optimizer=tf.optimizers.RMSprop(learning_rate=0.5))
err = error_rms();
print(np.mean(err))                

tfp.math.minimize(error_rms,
                num_steps=400,
                optimizer=tf.optimizers.RMSprop(learning_rate=0.25))
err = error_rms();
print(np.mean(err))

tfp.math.minimize(error_rms,
                num_steps=400,
                optimizer=tf.optimizers.RMSprop(learning_rate=0.25))

                
tfp.math.minimize(error_rms,
                num_steps=400,
                optimizer=tf.optimizers.RMSprop(learning_rate=0.125))
                  
err = error_rms();
print(np.mean(err))

stop = time.time()         
print(stop-start)

#%% optimization

pitch.assign(pitch_estimate[0:N])
pitch_inc.assign(tf.zeros(pitch_inc.shape))

start = time.time()
opt = tf.keras.optimizers.RMSprop(learning_rate=1) # RMSprop with learning=1
for i in range(200):
    pitch0 = pitch.numpy();
    
    opt.minimize(error_rms, var_list=[pitch, pitch_inc])
err = error_rms();

offset = pitch.numpy() - pitch0;
print('--')
print(np.mean(np.abs(offset)))
print(np.mean(err))

opt = tf.keras.optimizers.RMSprop(learning_rate=0.5) # RMSprop with learning=1
for i in range(200):
    pitch0 = pitch.numpy();

    opt.minimize(error_rms, var_list=[pitch, pitch_inc])
err = error_rms();

offset = pitch.numpy() - pitch0;
print('--')
print(np.mean(np.abs(offset)))
print(np.mean(err))


opt = tf.keras.optimizers.RMSprop(learning_rate=0.25) # RMSprop with learning=1
for i in range(200):
    pitch0 = pitch.numpy();

    opt.minimize(error_rms, var_list=[pitch, pitch_inc])
err = error_rms();

offset = pitch.numpy() - pitch0;
print('--')
print(np.mean(np.abs(offset)))
print(np.mean(err))


opt = tf.keras.optimizers.RMSprop(learning_rate=0.25) # RMSprop with learning=1
for i in range(200):
    pitch0 = pitch.numpy();
    
    opt.minimize(error_rms, var_list=[pitch, pitch_inc])
err = error_rms();

offset = pitch.numpy() - pitch0;
print('--')
print(np.mean(np.abs(offset)))
print(np.mean(err))
stop = time.time()

print(stop-start)

# %%
