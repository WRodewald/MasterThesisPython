
import numpy as np

import unittest

import tools
from tools.pitch_acf import pitch_acf
from tools.framed_audio import FramedAudio

class Test_pitch_acf(unittest.TestCase):

    def test_pitch_acf(self):
        block_size = 4096
        fs = 44100
        f0Sweep = np.linspace(200, 800, 8 * block_size)
        phase = np.add.accumulate(2. * np.pi * f0Sweep / fs)
        x  = np.sin(phase)
        audio = FramedAudio(x, fs, block_size, block_size)

        pitch = pitch_acf(audio,fs)
        pitch_reference = f0Sweep[0:-1:block_size]
        
        self.assertTrue(np.max(np.abs(pitch-pitch_reference)) < 50)



if __name__ == '__main__':   
    unittest.main()

