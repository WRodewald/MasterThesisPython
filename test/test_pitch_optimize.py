
import numpy as np

import unittest

import tools
from tools.framed_audio import FramedAudio
from tools import pitch_optimize

import matplotlib.pyplot as plt

class Test_pitch_optimize(unittest.TestCase):

    def test_pitch_optimize(self):

        block_size = 2048
        fs = 44100
        f0_sweep = np.linspace(100, 200, block_size)
        phase = np.add.accumulate((2. * np.pi * f0_sweep) / fs)
        frame = np.cos(phase)

        pitch, pitch_inc = pitch_optimize.pitch_optimize_frame(frame, fs)
        
        expected_pitch = np.mean(f0_sweep)
        expected_pitch_inc = 44100 * (np.max(f0_sweep) - np.min(f0_sweep))/block_size

        self.assertAlmostEqual(pitch,expected_pitch, delta=0.1)
        self.assertAlmostEqual(pitch_inc,expected_pitch_inc, delta=10)



if __name__ == '__main__':   
    unittest.main()

