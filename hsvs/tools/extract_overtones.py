import numpy as np
from .framed_audio import FramedAudio
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing

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

def synthesize_overtones(overtones, pitch, pitch_inc, fs, N):
    
    num_overtones = len(overtones)

    overtones = np.reshape(overtones, [num_overtones, 1])

    k = np.reshape(np.arange(1,num_overtones+1), (-1,1)).T
    
    t = (np.arange(N) - 0.5 * N + 0.5)/fs

    phase = np.reshape(pitch * t + 0.5 * pitch_inc * t * t, (-1,1))

    resynth = 2. * np.abs(overtones.T) * np.cos(2. * np.pi * k * phase + np.angle(overtones.T))
    resynth = np.real(np.sum(resynth,1)) / 2
    return resynth

def extract_ovetones_wrap(args):
    return extract_ovetones(*args)

def extract_overtones_from_audio(audio, pitch = None, pitch_inc = None, num_overtones = 40, num_threads=1):
    
    if(pitch == None): 
        pitch = audio.get_trajectory('pitch')

    if(pitch_inc == None): 
        pitch_inc = audio.get_trajectory('pitch-inc')

    window = np.hanning(audio.block_size)
    overtones = np.zeros([audio.get_num_frames(), num_overtones], dtype='complex128')

    N = audio.get_num_frames()

    print('Extracting overtones ...')
    if(num_threads == 1):        
        # analysis per frame
        for i in tqdm(range(audio.get_num_frames())):
            frame = audio.get_frame(i) * window
            overtones[i,:]  = extract_ovetones(frame, pitch[i], pitch_inc[i], audio.fs, num_overtones, False)

    else:
        # multi threaded

        # collect inputs
        inputs = []
        for i in range(N):
            frame = audio.get_frame(i) * window
            inputs.append((frame,  pitch[i], pitch_inc[i], audio.fs, num_overtones, False))
        
        # schedule workers and run
        with multiprocessing.Pool(processes=num_threads) as pool:            
            results = list(tqdm(pool.imap(extract_ovetones_wrap, inputs, chunksize=1), total=N ))

        # collect outputs
        for i in range(N):
            overtones[i,:] = results[i][:]

    return overtones





