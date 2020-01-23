import numpy as np
from matplotlib import pyplot as plt
# estimate the pitch for one frame
def pitch_acf_frame(frame, fs, f_max = 2000):
    pitch_max_idx = int(np.round(fs/f_max))

    block_size = frame.size

    coeff = np.correlate(frame, frame, "full")
    coeff = coeff[block_size:(2*block_size+1)]

    # dynamic limit for the offset
    rising_coeffs = np.where(np.diff(coeff) > 0)
    dyn_offset = pitch_max_idx
    if((rising_coeffs[0].size > 0) and (rising_coeffs[0][0] > dyn_offset)):
        dyn_offset = rising_coeffs[0][0]

    # highest xcorr bin within our pitch limit
    pitchIdx = dyn_offset + coeff[dyn_offset:].argmax()

    pitch = fs / (1+pitchIdx)

    return pitch
    
# estmate the pitch for multiple frames
def pitch_acf(framed_audio, fs, f_max = 2000):

    # num frames
    N = int(framed_audio.get_num_frames())

    # pitch vector
    pitch = np.zeros(N)
    
    # frame wise analysis
    for i in range(N):
        frame = framed_audio.get_frame(i)
        pitch[i] = pitch_acf_frame(frame, fs, f_max)
        
    return pitch
