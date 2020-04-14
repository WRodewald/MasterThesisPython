#%%
from tools.framed_audio import FramedAudio
from tools import magnitude
import crepe

import numpy as np

from scipy import interpolate

from matplotlib import pyplot as plt

def predict(framed_audio, perform_octave_correction = True):
    source_file = framed_audio.src_file

    step_size = 5
    crepe_time, crepe_pitch, confidence, _ = crepe.predict(framed_audio.get_raw(), framed_audio.fs, step_size = step_size, center=True)

    if(perform_octave_correction):
        # calculate time onset and offset
        onset, offset = magnitude.get_onset_offset(framed_audio)
        t = framed_audio.get_time()
        t_onset = t[onset]
        t_offset = t[offset]
        
        # calculate associated indices with a margin of 1% of audio length 
        margin = int(np.ceil(0.01 * crepe_time.size))
        onset_crepe  = np.argmin(np.abs(t-t_onset))  + margin
        offset_crepe = np.argmin(np.abs(t-t_offset)) - margin

        # get bin of highest confidence within onset : offset as a starting point
        idx_highest_confidence = onset_crepe + np.argmax(confidence[onset:offset])

        # run octave error correction
        crepe_pitch = fix_octave_errors(crepe_pitch, idx_highest_confidence)


    # interpolate results from crepe
    crepe_interpolator = interpolate.interp1d(crepe_time, crepe_pitch)
    time = framed_audio.get_time( centered=True)

    crepe_pitch_interp = crepe_interpolator(time)


    return crepe_pitch_interp


def fix_octave_errors(f0, base_idx, oct_range = [-2,+2], max_cents = 100):

    # this function takes an f0 prediction vector and tries to correct octave errors
    # pitch jump from one freq to 2x, 3x, 4x or 1/2x 1/3x, 1/4x of the current pitch within 
    # max_cents cents are assumed to be octave jumps and are corrected
        
    def correctStep(f1, f2, oct_range, max_cents):

        diff = f2 / f1;

        if(diff > 1):     
            diff_oct = np.round(diff)
            step = diff_oct - 1      
        else:
            diff_oct = 1. / np.round(1/diff)
            step = (1 / diff_oct) - 1       
        

        diff_cent = 100 * 12 * np.log2(f2/ (f1*diff_oct));

        within_range = (step >= oct_range[0] ) and (step <= oct_range[1]) and (step is not 0)

        if(within_range and (np.abs(diff_cent) < max_cents)):
            f2 = f2 / diff_oct
    
        return f2

    for i in range (base_idx, 1, -1):
            f0[i-1] = correctStep(f0[i], f0[i-1], oct_range, max_cents)     


    for i in range (base_idx, len(f0)-1):
            f0[i+1] = correctStep(f0[i], f0[i+1], oct_range, max_cents)     

    return f0
