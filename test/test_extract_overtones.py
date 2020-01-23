
import numpy as np

import unittest

import tools
from tools.extract_overtones import extract_ovetones
from tools.framed_audio import FramedAudio

import matplotlib.pyplot as plt

class Test_extract_overtones(unittest.TestCase):

    def test_extract_overtones(self):

        block_size = 800
        fs = 20000
        phase_offset = 0.5*np.pi
        f0_sweep = np.linspace(100, 100, block_size)
        phase = np.add.accumulate((2. * np.pi * f0_sweep) / fs)
        frame = np.cos(phase + phase_offset)

        frame_win = frame * np.hanning(frame.size)

        overtones, resynth = extract_ovetones(frame_win, 100, 0, fs)

        # since we applied a hanning, we expect our resynthsized signal to
        # have half the magnitude 

        #plt.plot(abs(overtones)) 
        #plt.show()

        #plt.plot(2*resynth)

        # check resynthesized signal
        self.assertAlmostEqual(np.max(np.abs(2*resynth-frame)), 0.00, delta=0.01)

        # check overtones
        self.assertAlmostEqual(np.abs(overtones[0]), 0.5, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(overtones[1:])), 0, delta=0.1)

        # check phase of first harmonic
        self.assertAlmostEqual(np.angle(overtones[0]), phase_offset, delta=0.01)
        
    
    def test_extract_overtones_no_resynth(self):

        block_size = 800
        fs = 20000
        phase_offset = 0.5*np.pi
        f0_sweep = np.linspace(100, 100, block_size)
        phase = np.add.accumulate((2. * np.pi * f0_sweep) / fs)
        frame = np.cos(phase + phase_offset)

        frame_win = frame * np.hanning(frame.size)

        overtones = extract_ovetones(frame_win, 100, 0, fs, resynthesize=False)

        # since we applied a hanning, we expect our resynthsized signal to
        # have half the magnitude 

        # check overtones
        self.assertAlmostEqual(np.abs(overtones[0]), 0.5, delta=1)
        self.assertAlmostEqual(np.max(np.abs(overtones[1:])), 0, delta=0.1)

        # check phase of first harmonic
        self.assertAlmostEqual(np.angle(overtones[0]), phase_offset, delta=0.01)
        


if __name__ == '__main__':   
    unittest.main()

