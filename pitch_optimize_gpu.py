#%%
from tools import audio_io
from tools.framed_audio import FramedAudio
from tools import pitch_optimize
import time, sys

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
hop_size = 64
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

pitch_estimate_crepe = crepe_pitch_interp


N  = framed_audio.get_num_frames()
N0 = 0
fs = framed_audio.fs
pitch_estimate = pitch_estimate_crepe
# run acf pitch estimate
if pitch_estimate is None: 
    pitch_estimate = pitch_acf.pitch_acf(framed_audio,fs)


num_overtones = 15

block_size = framed_audio.block_size
sample_rate = framed_audio.fs

#%% 

from tools import pitch_optimize
import importlib
importlib.reload(pitch_optimize)

options = {'iterations': 400, 
           'batch-size': 4096, 
            'pitch-rate': 0.1, 
            'pitch-inc-rate': 4000,
            'rate-decay':0.1}

pitch, pitch_inc = pitch_optimize.pitch_optimize_gpu(framed_audio, pitch_estimate = pitch_estimate, options=options)
 

plt.plot(pitch_inc)
plt.show()