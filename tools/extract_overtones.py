import numpy as np
from tools.framed_audio import FramedAudio
import matplotlib.pyplot as plt

# this function extracts some overtones
def extract_ovetones(frame, pitch, pitch_inc, fs, num_overtones = 10, resynthesize = True):
    k = np.reshape(np.arange(1,num_overtones+1), (-1,1)).T

    N = frame.size
    
    t = (np.arange(N) - 0.5 * N + 0.5)/fs

    phase = np.reshape(pitch * t + 0.5 * pitch_inc * t * t, (-1,1))
    
    divider = np.exp(1j * 2 * np.pi * k * phase)

    overtones = np.sum(np.reshape(frame,(-1,1)) / divider, axis=0)
    overtones /= 0.5 * frame.size

    if(resynthesize):
            
        resynth = 2. * np.abs(overtones.T) * np.cos(2. * np.pi * k * phase + np.angle(overtones.T))
        resynth = np.real(np.sum(resynth,1)) / 2

        return overtones, resynth
    else:
        return overtones






