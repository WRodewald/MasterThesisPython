
import numpy as np
import numba

spec = [
    ('b0', numba.float32),
    ('b1', numba.float32),
    ('b2', numba.float32),
    ('a1', numba.float32),
    ('a2', numba.float32),
    ('z0', numba.float32),
    ('z1', numba.float32),
]

# simple LF-Rd wavetable oscillator based on Fant 1985, Fant 1995
@numba.jitclass(spec=spec)
class BiQuad:


    def tick(self, x):

        #TDF2
        y       = self.b0 * x               + self.z0
        self.z0 = self.b1 * x - self.a1 * y + self.z1
        self.z1 = self.b2 * x - self.a2 * y

        return y

    def __init__(self, num_instances = 128, num_samples = 2048):
    
        self.b0 = 1.
        self.b1 = 0.
        self.b2 = 0.
        self.a1 = 0.
        self.a2 = 0.

        self.z0 = 0.
        self.z1 = 0.
