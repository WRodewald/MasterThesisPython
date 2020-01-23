
from tools import audio_io
from tools import framed_audio
from tools import pitch_optimize

from tools import dataset

import numpy as np

from matplotlib import pyplot as plt

import crepe.core

import warnings


if __name__ == '__main__':

    
    vocalset_root = dataset.get_root_path()

    src_file = vocalset_root + '/female4/scales/slow_forte/f4_scales_c_slow_forte_o.ogg'


    data, fs = audio_io.read(src_file)

    hop_size = 256
   
    audio = framed_audio.FramedAudio(data, 2048, hop_size, centered=True)

    step_size = 10
    crepe_time, crepe_pitch, confiidence, _ = crepe.predict(data, fs, step_size = step_size)

    pitch, pitch_inc = pitch_optimize.pitch_optize(audio,fs, verbose=True, num_threads=15)

    time = np.arange(audio.get_num_frames()) / (fs/hop_size)

    plt.plot(crepe_time, crepe_pitch, time, pitch)
    plt.show()

