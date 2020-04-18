
#%%
#%load_ext autoreload
#%autoreload 2
from tools import dataset
from tools.framed_audio import FramedAudio
from tools import crepe_estimate
from tools import magnitude

from tools import pitch_optimize

from matplotlib import pyplot as plt

import numpy as np

from tqdm import tqdm

#%%


config = {
        'centered':True,
        'block-size':2048,
        'hop-size':64
    }

# dataset & audio
files = dataset.get_sample('scales', 'slow_forte', '*', '*')

# run bulk prediction
for i in range(len(files)):
    
    print(f"File {i+1} of {len(files)}")

    audio = FramedAudio.from_file(files[i],  config=config)
    pitch_crepe = crepe_estimate.predict(audio)
    onset, offset = magnitude.get_onset_offset(audio)

    
    options = {'iterations': 400, 
               'batch-size': 4096, 
               'pitch-rate': 0.1, 
               'pitch-inc-rate': 4000,
               'rate-decay':0.1}

    pitch, pitch_inc = pitch_optimize.pitch_optimize_gpu(audio, pitch_estimate = pitch_crepe, options=options)
 

    audio.store_trajectory('pitch', pitch)
    audio.store_trajectory('pitch-inc', pitch_inc)
    audio.store_trajectory('crepe', pitch_crepe)

    audio.save_json()




# %%
