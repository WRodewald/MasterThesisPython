import numpy as np
from scipy import stats
from scipy import signal

from matplotlib import pyplot as plt

def get_rms(framed_audio):

    rms = np.zeros(framed_audio.get_num_frames())

    for i in range(framed_audio.get_num_frames()):
        rms[i] = np.math.sqrt(np.sum(np.power(framed_audio.get_frame(i),2.)))

    return rms


def get_onset_offset(framed_audio, t_sigma = 3, move_avg_len = 11):

    # returns the signal onset and offset as frame indices
    # expects the onset and offset to be close to the start and end of the audio respectivley
    # t_sigma is used to weight high 

    rms = 20. * np.log10(get_rms(framed_audio))
    rms = rms - np.max(rms)

    
    # calculate rms change
    rms_diff = np.append([0.], np.diff(rms))

    # centered moving average  
    window = np.ones(move_avg_len)/move_avg_len
    rms_diff = signal.convolve(rms_diff, window, mode='same')

    t = framed_audio.get_time()
    
    # find maximum rms accend and minimum rms descent with weighted gaussian
    onset  = np.argmax(rms_diff * stats.norm.pdf(t,t[+1], t_sigma))
    offset = np.argmin(rms_diff * stats.norm.pdf(t,t[-1], t_sigma))

    return onset, offset

