#%%
import numpy as np
from tools import pitch_acf 
from . import extract_overtones
from . import framed_audio
from scipy.optimize import minimize

from tqdm import tqdm

import multiprocessing

from matplotlib import pyplot as plt



# for a given frame, we use the pitch to resynthesize the frame and return the rms error between frame and synth
def get_error_for_pitch_prediction(frame, pitch_est, pitch_inc_est, fs, num_overtones = 10, plot = False):
    
    win = np.hanning(frame.size)
    frame = frame * win
    
    _, synth = extract_overtones.extract_ovetones(frame, pitch_est, pitch_inc_est, fs, num_overtones)
    synth = 2 * synth * win

    err_rms = np.sqrt(np.mean((synth-frame)**2) / frame.size)
    return err_rms

# function returns the pitch and pitch increment for one frame
# method:   'Nelder-Mead' None
def pitch_optimize_frame(frame, fs, pitch_est = 0, options = {}):

    # fall back to acf for first estimate
    if(pitch_est == 0):
        pitch_est = pitch_acf.pitch_acf_frame(frame,fs)
    
    pitch = pitch_est
    pitch_inc = 0

    # extract options
    method = options.get('method', 'Nelder-Mead')
    skip_f0_estimation = options.get('skip_f0_estimation', False)
    num_overtones = options.get('num_overtones', 10)

    # first round of min search
    if(not skip_f0_estimation):
        fun = lambda args: get_error_for_pitch_prediction(frame, args[0], 0, fs, num_overtones)
        res = minimize(fun, [pitch], method=method)
        pitch = res.x[0]
    
    # second round of min search with increment
    fun = lambda args: get_error_for_pitch_prediction(frame, args[0], args[1], fs, num_overtones)
    res = minimize(fun, [pitch, 0], method=method)
    pitch = res.x[0]
    pitch_inc = res.x[1]


    if(False):
        win = np.hanning(frame.size)
        frame = frame * win
        
        _, synth = extract_overtones.extract_ovetones(frame, pitch, pitch_inc, fs, num_overtones)
        synth = 2 * synth * win

        t = np.arange(frame.size)
        plt.plot(t, frame, t, synth)
        plt.show()

    return pitch, pitch_inc


def pitch_optimize_frame_wrap(args):
    return pitch_optimize_frame(*args)

# function returns the pitch and pitch increment for multiple frames
def pitch_optimize(framed_audio, pitch_estimate = None, options={}, num_threads = 1):

    N = framed_audio.get_num_frames()
    fs = framed_audio.fs

    # run acf pitch estimate
    if pitch_estimate is None: 
        pitch_estimate = pitch_acf.pitch_acf(framed_audio,fs)

    assert pitch_estimate.size == N, 'pitch_optimize: Invalid pitch estimate'

    # output vectors
    pitch     = np.zeros(N)
    pitch_inc = np.zeros(N)
    
    
    print("Running optimized pitch extraction, block size=%d, hop size=%d"% (framed_audio.block_size, framed_audio.hop_size))
    
    if(num_threads == 1):

        # analysis per frame
        for i in tqdm(range(N)):
            frame = framed_audio.get_frame(i)

            frame_pitch, frame_pitch_inc = pitch_optimize_frame(frame, fs, pitch_estimate[i], options)

            pitch[i] = frame_pitch
            pitch_inc[i] = frame_pitch_inc

    else:
        inputs = []

        for i in range(N):
            inputs.append((framed_audio.get_frame(i), fs, pitch_estimate[i], options))
        
        with multiprocessing.Pool(processes=num_threads) as pool:            
            results = list(tqdm(pool.imap(pitch_optimize_frame_wrap, inputs ), total=N ))


        for i in range(N):
            pitch[i] = results[i][0]
            pitch_inc[i] = results[i][1]

    print("Done")



    return pitch, pitch_inc
