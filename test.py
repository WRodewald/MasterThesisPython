
from tools import audio_io
from tools import framed_audio
from tools import pitch_optimize

from tools import dataset

import numpy as np

from matplotlib import pyplot as plt

import crepe.core

import warnings

from scipy import interpolate

if __name__ == '__main__':

    # config
    hop_size = 256
    block_size = 2048

    # dataset & audio
    vocalset_root = dataset.get_root_path()
    src_file = vocalset_root + '/female4/scales/slow_forte/f4_scales_c_slow_forte_o.ogg'

    data, fs = audio_io.read(src_file)   
    audio = framed_audio.FramedAudio(data, block_size, hop_size, centered=True)
    
    # run crepe
    step_size = 5
    crepe_time, crepe_pitch, confiidence, _ = crepe.predict(data, fs, step_size = step_size, center=True)

    # interpolate results from crepe
    crepe_interpolator = interpolate.interp1d(crepe_time, crepe_pitch)
    time = audio.get_time(fs, centered=True)

    crepe_pitch_interp = crepe_interpolator(time)

     
    # run min-search pitch estimation
    pitch, pitch_inc = pitch_optimize.pitch_optimize(audio,fs, 
                                                    pitch_estimate = crepe_pitch_interp,
                                                    options = {
                                                    'skip_f0_estimation':True,  
                                                    'method':None},
                                                    num_threads=15)


    plt.plot(crepe_time, crepe_pitch, time, pitch)
    plt.show()

