
import numpy as np

import unittest

import tools
from tools.framed_audio import FramedAudio


class Test_FramedAudio(unittest.TestCase):


    def test_centered(self):    
        audio = FramedAudio(np.arange(512), 256,128, centered=True)

        # frames expected as follows
        # 1: [-128, 128[
        # 2: [0,    256[
        # 3: [128,  384[
        # 4: [256,  512[

        self.assertEqual(audio.get_num_frames(), 4)
        self.assertEqual(audio.get_frame(0).size, 256)
        self.assertEqual(audio.get_frame(1).size, 256)
        self.assertEqual(audio.get_frame(2).size, 256)
        self.assertEqual(audio.get_frame(3).size, 256)

        self.assertEqual(audio.get_frame(0)[-1], 127)
        self.assertEqual(audio.get_frame(1)[-1], 255)
        self.assertEqual(audio.get_frame(2)[-1], 383)
        self.assertEqual(audio.get_frame(3)[-1], 511)

        
        audio = FramedAudio(np.arange(512), 256, 128, centered=False)

        # frames expected as follows
        # 1: [-128, 128[
        # 2: [0,    256[
        # 3: [128,  384[
        # 4: [256,  512[

        self.assertEqual(audio.get_num_frames(), 3)
        self.assertEqual(audio.get_frame(0).size, 256)
        self.assertEqual(audio.get_frame(1).size, 256)
        self.assertEqual(audio.get_frame(2).size, 256)

        self.assertEqual(audio.get_frame(0)[-1], 255)
        self.assertEqual(audio.get_frame(1)[-1], 383)
        self.assertEqual(audio.get_frame(2)[-1], 511)

        
        audio = FramedAudio(np.arange(513), 256, 128, centered=True)
        self.assertEqual(audio.get_num_frames(), 4)


    def test_get_time(self):   
        audio = FramedAudio(np.arange(512), 256,256)

        expected = np.array([0., 1.])
        result   = audio.get_time(256, False)
        self.assertTrue(np.array_equal(expected, result))
                
        expected = np.array([0.5, 1.5])
        result   = audio.get_time(256, True)
        self.assertTrue(np.array_equal(expected, result))

        
        audio = FramedAudio(np.arange(512), 256, 256, centered=True)
        
        expected = np.array([-0.5, 0.5])
        result   = audio.get_time(256, False)
        self.assertTrue(np.array_equal(expected, result))
                
        expected = np.array([0., 1.])
        result   = audio.get_time(256, True)
        self.assertTrue(np.array_equal(expected, result))


    def test_get_num_frames(self):                
        audio = FramedAudio(np.zeros((1,128)),256, 128)   
        self.assertEqual(audio.get_num_frames(), 1)
        
        audio = FramedAudio(np.zeros((1,256)),256, 128)   
        self.assertEqual(audio.get_num_frames(), 1)
        
        audio = FramedAudio(np.zeros((1,384)),256, 128)   
        self.assertEqual(audio.get_num_frames(), 2)
        
        audio = FramedAudio(np.zeros((1,385)),256, 128)   
        self.assertEqual(audio.get_num_frames(), 2)
                
        audio = FramedAudio(np.zeros((0,0)),256, 128)   
        self.assertEqual(audio.get_num_frames(), 1)

        # with centering

    def test_get_frame(self):      
        
        vector = np.arange(1, 7)
        audio = FramedAudio(vector, 4, 2)   

        frame = audio.get_frame(0)    
        expected =  np.array([1,2,3,4])
        self.assertTrue(np.array_equal(frame, expected))
        
        frame = audio.get_frame(1)    
        expected =  np.array([3,4,5,6])
        self.assertTrue(np.array_equal(frame, expected))
        

    def test_get_raw(self):      
        
        vector = np.arange(1, 7)
        audio = FramedAudio(vector, 4, 2)  
        raw = audio.get_raw()    

        self.assertTrue(np.array_equal(raw, vector))


if __name__ == '__main__':   
    unittest.main()

