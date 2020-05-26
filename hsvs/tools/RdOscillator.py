
import numpy as np
import json
import os

from . import lf_rd
import scipy.signal as signal

import numba

@numba.jit(nopython=True)
def tick_jit_buffer(out, Rd, f0, fs, phase, num_instances, num_samples, table):

    for i in range(len(Rd)):
        out[i], phase = tick_jit(Rd[i], f0[i], fs, phase, num_instances, num_samples, table)

    return out, phase


@numba.jit(nopython=True)
def tick_jit(Rd, f0, fs, phase, num_instances, num_samples, table):
    
        phase += f0/fs
        phase -= np.floor(phase)
        
        Rd_pos = num_instances * (Rd - 0.3) / (2.7-0.3)
        t_pos  = num_samples * phase

        # Rd indices and fraction
        Rd_x0 = int(np.floor(Rd_pos))
        Rd_x0 = min(Rd_x0, num_instances-1)
        Rd_x1 = min(Rd_x0 + 1, num_instances-1)
        Rd_frac = Rd_pos-Rd_x0

        # phase / time indices and fraction
        t_x0 = int(np.floor(t_pos)) % num_samples
        t_x1 = t_x0 + 1
        t_x1 -= (t_x1 == num_samples) * t_x1 # wrap 
        t_frac = t_pos - t_x0
            
        # 2x2 values
        x00 = table[Rd_x0, t_x0]
        x01 = table[Rd_x0, t_x1]
        x10 = table[Rd_x1, t_x0]
        x11 = table[Rd_x1, t_x1]

        # 2d linear interpolation
        y0 = x00 +  t_frac * (x01 - x00)
        y1 = x10 +  t_frac * (x11 - x10)
        y  = y0  + Rd_frac * (y1  -  y0)

        return y, phase

# simple LF-Rd wavetable oscillator based on Fant 1985, Fant 1995

class RdOscillator:


    def construct_table(self, num_instances, num_samples):
        self.num_instances = num_instances
        self.num_samples   = num_samples

        self.table = np.zeros([num_instances, num_samples])

        Rds = np.linspace(0.3, 2.7, num_instances)

        # build up wavetable
        for i in range(self.num_instances):            
            t = np.linspace(0, 1, self.num_samples+1)[0:-1]
            y = lf_rd.calculate_waveform(Rds[i], t, align_te=True)

            self.table[i,:] = y

    def reset(self):        
        self.phase = 0

    def tick(self, Rd, f0, fs):

        if(np.isscalar(Rd)):
            out, phase = tick_jit(Rd, f0, fs, self.phase, self.num_instances, self.num_samples, self.table)
            self.phase = phase
            return out
        else:
            out = np.zeros(Rd.shape)
            out, phase = tick_jit_buffer(out, Rd, f0, fs, self.phase, self.num_instances, self.num_samples, self.table)
            return out

    def __init__(self, num_instances = 128, num_samples = 2048):
    
        self.reset()
        self.construct_table(num_instances, num_samples)
