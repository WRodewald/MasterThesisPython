#%%
import numpy as np
from tools import pitch_acf 
from . import extract_overtones
from . import framed_audio
from scipy.optimize import minimize

import tensorflow as tf

import time
import math

from tqdm import tqdm, trange

import multiprocessing

from matplotlib import pyplot as plt



# for a given frame, we use the pitch to resynthesize the frame and return the rms error between frame and synth
def get_error_for_pitch_prediction(frame, pitch_est, pitch_inc_est, fs, num_overtones = 15, plot = False):
    
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
    num_overtones = options.get('num_overtones', 15)

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
    

def pitch_optimize_gpu(framed_audio, pitch_estimate = None, options={}):

    N = framed_audio.get_num_frames()
    fs = framed_audio.fs
    block_size = framed_audio.block_size

    # run acf pitch estimate
    if pitch_estimate is None: 
        pitch_estimate = pitch_acf.pitch_acf(framed_audio,fs)

    assert pitch_estimate.size == N, 'pitch_optimize: Invalid pitch estimate'

    # output vectors
    pitch     = np.zeros(N)
    pitch_inc = np.zeros(N)

    # optimization parameters
    batch_size = options.get('batch-size', 1024)
    
    # output vector
    pitch = np.zeros(N)
    pitch_inc = np.zeros(N)

    start = time.time()
    num_batches = math.ceil(N / batch_size)
    for b in range(num_batches):
        frame0 = b * batch_size
        frame_range = np.arange(frame0, min(frame0+batch_size, N))

        # collect frames
        batch_frames = np.zeros([frame_range.size, block_size])
        for f in frame_range:
            batch_frames[f-frame0,:] = framed_audio.get_frame(f)

        batch_pitch = pitch_estimate[frame_range]

        print(f"\n\rRunning batch {b+1} of {num_batches}")

        # forward to batch processing
        batch_pitch, batch_pitch_inc, pitch_grad, pitch_inc_grad = pitch_optimize_batch_gpu(batch_frames, batch_pitch , fs, options)

        # store results
        pitch[frame_range] = batch_pitch
        pitch_inc[frame_range] = batch_pitch_inc
    
    stop = time.time()

    print(f"\n\rBatch processing took {stop-start} seconds")
    return pitch, pitch_inc


def pitch_optimize_batch_gpu(frames, pitch_estimate, fs, options):

    N = frames.shape[0]
    block_size = frames.shape[1]

    # learning rate    
    iterations     = options.get('iterations', 400)
    num_overtones  = options.get('num_overtones', 15)
    pitch_rate     = options.get('pitch-rate', 0.05)
    pitch_inc_rate = options.get('pitch-inc-rate', 4000)
    rate_decay     = options.get('rate-decay', 0.1)
    
    # calculate rate-decay per iteration
    rate_decay = np.exp(np.log(rate_decay)/iterations)

    # prepare vectors
    pitch     = tf.cast(tf.reshape(pitch_estimate, [N,1,1]), dtype='float32')
    pitch_inc = tf.cast(np.zeros([N,1,1]), dtype='float32')

    frames = tf.cast(tf.reshape(frames, [N, block_size, 1]), dtype='float32')
    
    # hanning window over time
    window = tf.reshape(tf.signal.hann_window(block_size), [1,block_size,1])

    # factor for overtones where k[0] = 1 = 1st harmonic
    k = tf.reshape(tf.range(1,num_overtones+1, dtype='float32'), [1,1,num_overtones])

    # time vector, zero-centered for each frame
    t = (tf.range(0, block_size, dtype='float32') - 0.5 * block_size + 0.5) / fs
    t = tf.reshape(t, [1,block_size,1])

    # cost function, calculates cost in rms_dB of difference signal, and gradients
    @tf.function
    def error_rms(pitch_in, pitch_inc_in):
        
        # calculate phase over time 
        phase = pitch_in * t + 0.5 * pitch_inc_in * t * t

        # claculate divider for complex fourier analysis
        div   = tf.exp(tf.complex(0., 2. * np.pi * k * phase))

        # complex overtone magnitude and phase 
        ot = tf.reduce_sum(tf.complex((frames * window), 0.) / div,1, keepdims=True) / block_size
        
        # resynthesis with purely harmonic model
        resynth = 2. * tf.math.abs(ot) * tf.math.cos(2. * np.pi * k * phase + tf.math.angle(ot))
        resynth = 2. * tf.reduce_sum(resynth, 2, keepdims=True)

        # error is dB(rms(org-synth)) 
        difference = (resynth * window - frames * window)
        error_rms = tf.squeeze(tf.sqrt(tf.reduce_mean(difference * difference, 1)))

        error_db = 20. * tf.math.log(error_rms) / tf.math.log(10.)

        return error_db, tf.gradients(error_db, pitch_in)[0], tf.gradients(error_db, pitch_inc_in)[0]

    avg_pitch_grad = 0
    avg_pitch_inc_grad = 0

    # iterative gradient decent    
    with trange(iterations) as t_range:
        for it in t_range:

            # calculate current position
            cost, pitch_grad, pitch_inc_grad = error_rms(pitch, pitch_inc)

            # descent
            pitch     -= pitch_rate * pitch_grad
            pitch_inc -= pitch_inc_rate * pitch_inc_grad

            # update rate
            pitch_rate *= rate_decay
            pitch_inc_rate *= rate_decay


            # store some results
            avg_pitch_grad = np.mean(np.abs(pitch_grad))
            avg_pitch_inc_grad = np.mean(np.abs(pitch_inc_grad))

            t_range.set_postfix({'gradients' : [avg_pitch_grad, avg_pitch_inc_grad]})
        

    return pitch[:,0,0].numpy(), pitch_inc[:,0,0].numpy(), avg_pitch_grad, avg_pitch_inc_grad

