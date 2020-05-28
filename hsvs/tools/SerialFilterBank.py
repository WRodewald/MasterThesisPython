
import numpy as np
from . import BiQuad

import numba
from tqdm import tqdm

BiQuad_type = numba.deferred_type()
BiQuad_type.define(BiQuad.BiQuad.class_type.instance_type)

spec = [
    ('filters', BiQuad_type[:]),
]

# simple LF-Rd wavetable oscillator based on Fant 1985, Fant 1995
class SerialFilterBank:


    def tick(self, x, verbose=False):
        
        if np.isscalar(x):
            y = x
            for filter in self.filters:
                y = filter.tick(y)
            return y
        else:
            
            y = x
            for i in tqdm(range(len(y)), disable=(not verbose)):       
                for filter in self.filters:
                    y[i] = filter.tick(y[i])
            return y


    #processes signal with per-sample pole zero updates
    def tick_pz(self, x, poles, zeros, verbose = False):


        if np.isscalar(x):
            self.set_pz_conjugates(poles, zeros)
            y = x
            for filter in self.filters:
                y = filter.tick(y)
            return y
        else:
            
            y = x
            for i in tqdm(range(len(y)), disable=(not verbose)):                
                self.set_pz_conjugates(poles[i,:], zeros[i,:])
                for filter in self.filters:                    
                    y[i] = filter.tick(y[i])
            return y



    # updates coefficients with array of poles zeros which are extended
    # to include conjuate pairs
    def set_pz_conjugates(self, poles, zeros):

        for i in range(len(poles)):
            p0 = poles[i]
            z0 = zeros[i]

            re_p0 = np.real(p0)
            im_p0 = np.imag(p0)
            re_z0 = np.real(z0)
            im_z0 = np.imag(z0)

            #self.filters[i].a1 = -np.real(np.conj(p0) + p0)
            #self.filters[i].a2 =  np.real(np.conj(p0) * p0)
            
            self.filters[i].b0 = 1
            #self.filters[i].b1 = -np.real(np.conj(z0) + z0)
            #self.filters[i].b2 =  np.real(np.conj(z0) * z0)
            
            self.filters[i].a1 = -(re_p0 + re_p0)
            self.filters[i].a2 = re_p0 * re_p0 + im_p0 * im_p0
            
            self.filters[i].b1 = -(re_z0 + re_z0)
            self.filters[i].b2 = re_z0 * re_z0 + im_z0 * im_z0



    def __init__(self, num_sos):
    
        self.filters = []
        for i in range(num_sos):
            self.filters.append(BiQuad.BiQuad())


        