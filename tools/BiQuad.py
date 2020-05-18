
import numpy as np

# simple LF-Rd wavetable oscillator based on Fant 1985, Fant 1995
class BiQuad:


    def tick(self, x):

        #TDF2
        y       = self.b0 * x               + self.z0
        self.z0 = self.b1 * x - self.a1 * y + self.z1
        self.z1 = self.b2 * x - self.a2 * y

        return y

    def __init__(self, num_instances = 128, num_samples = 2048):
    
        self.b0 = 1
        self.b1 = 0.
        self.b2 = 0.
        self.a1 = 0.
        self.a2 = 0.

        self.z0 = 0.
        self.z1 = 0
