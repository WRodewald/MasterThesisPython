
import numpy as np
from tools import BiQuad

# simple LF-Rd wavetable oscillator based on Fant 1985, Fant 1995
class SerialFilterBank:


    def tick(self, x):

        y = x
        for filter in self.filters:
            y = filter.tick(y)

        return y

    # updates coefficients with array of poles zeros which are extended
    # to include conjuate pairs
    def set_pz_conjugates(self, poles, zeros):

        for i in range(len(poles)):
            p0 = poles[i]
            z0 = zeros[i]

            self.filters[i].a1 = -np.real(np.conj(p0) + p0)
            self.filters[i].a2 =  np.real(np.conj(p0) * p0)
            
            self.filters[i].b0 = 1
            self.filters[i].b1 = -np.real(np.conj(z0) + z0)
            self.filters[i].b2 =  np.real(np.conj(z0) * z0)


    def __init__(self, num_sos):
    
        self.filters = []
        for i in range(num_sos):
            self.filters.append(BiQuad.BiQuad())


        