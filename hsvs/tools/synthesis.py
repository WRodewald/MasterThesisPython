
from . import SerialFilterBank
from . import RdOscillator

import numpy as np
from scipy import interpolate

from hsvs.model import util

def run(pitch, Rd, gain, poles, zeros, fs, hop_size, verbose = True):

    num_poles   = poles.shape[1]
    num_samples = pitch.shape[0]

    # create oscillator and filter bank
    osc = RdOscillator.RdOscillator(num_instances=64, num_samples=512)
    filter = SerialFilterBank.SerialFilterBank(num_poles);

    # create time vectors for parameter rate and audio rate 
    t_frame = hop_size * np.arange(0, num_samples)
    t_audio = np.linspace(0, t_frame[-1], num_samples * hop_size);

    # linear interpolators for parameters
    f0_interpolator = interpolate.interp1d(t_frame, np.squeeze(pitch), axis=0)
    g_interpolator  = interpolate.interp1d(t_frame, np.squeeze(gain),  axis=0)
    Rd_interpolator = interpolate.interp1d(t_frame, np.squeeze(Rd),    axis=0)
    p0_interpolator = interpolate.interp1d(t_frame, np.squeeze(poles), axis=0)
    z0_interpolator = interpolate.interp1d(t_frame, np.squeeze(zeros), axis=0)

    # process oscillator
    print('processing source')
    audio = osc.tick(Rd_interpolator(t_audio), f0_interpolator(t_audio), fs)

    # gain
    print('processing gain')
    audio *= util.db2mag(g_interpolator(t_audio)).numpy()

    # filter with per sample pole/zero updates    
    print('processing filter')
    audio = filter.tick_pz(audio, p0_interpolator(t_audio), z0_interpolator(t_audio), verbose=verbose)

    return audio

