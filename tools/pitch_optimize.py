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
def pitch_optimize_frame(frame, fs, pitch_est = 0, num_overtones = 10):
    # fall back to acf for first estimate
    if(pitch_est == 0):
        pitch_est = pitch_acf.pitch_acf_frame(frame,fs)
    
    pitch = pitch_est
    pitch_inc = 0

    # first round of min search
    fun = lambda args: get_error_for_pitch_prediction(frame, args[0], 0, fs, num_overtones)
    #res = minimize(fun, [pitch_est], method='Nelder-Mead')
    res = minimize(fun, [pitch_est])
    pitch = res.x[0]
    
    # second round of min search with increment
    fun = lambda args: get_error_for_pitch_prediction(frame, args[0], args[1], fs, num_overtones)
    #res = minimize(fun, [pitch, 0], method='Nelder-Mead')
    res = minimize(fun, [pitch, 0])
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
def pitch_optize(framed_audio, fs, num_overtones = 10, verbose = False, num_threads = 1):

    N = framed_audio.get_num_frames()

    # run acf pitch estimate
    pitch_est = pitch_acf.pitch_acf(framed_audio,fs)

    # output vectors
    pitch     = np.zeros(N)
    pitch_inc = np.zeros(N)
    
    
    print("Running optimized pitch extraction, block size=%d, hop size=%d"% (framed_audio.block_size, framed_audio.hop_size))
    
    if(num_threads == 0):

        # analysis per frame
        for i in tqdm(range(N)):
            frame = framed_audio.get_frame(i)

            frame_pitch, frame_pitch_inc = pitch_optimize_frame(frame, fs, pitch_est[i], num_overtones)

            pitch[i] = frame_pitch
            pitch_inc[i] = frame_pitch_inc

    else:
        inputs = []

        for i in range(N):
            inputs.append((framed_audio.get_frame(i), fs, pitch_est[i], num_overtones))
        
        with multiprocessing.Pool(processes=num_threads) as pool:            
            results = list(tqdm(pool.imap(pitch_optimize_frame_wrap, inputs ), total=N ))


        for i in range(N):
            pitch[i] = results[i][0]
            pitch_inc[i] = results[i][1]

    print("Done")



    return pitch, pitch_inc
