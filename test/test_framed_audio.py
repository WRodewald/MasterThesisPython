
import numpy as np

import unittest

import tools
from tools.framed_audio import FramedAudio
from tools import audio_io

class Test_FramedAudio(unittest.TestCase):

    def test_from_file(self):   
        src = 'test/audio_sample.ogg'

        audio = FramedAudio.from_file(src, 2048, 1024, True)

        array, _ = audio_io.read(src)

        self.assertTrue(np.array_equal(array, audio.array))

    def test_centered(self):    
        audio = FramedAudio(np.arange(512), 44100, 256,128, centered=True)

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

        
        audio = FramedAudio(np.arange(512), 44100, 256, 128, centered=False)

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

        
        audio = FramedAudio(np.arange(513), 44100, 256, 128, centered=True)
        self.assertEqual(audio.get_num_frames(), 4)


    def test_get_time(self):   
        audio = FramedAudio(np.arange(512), 44100, 256,256)

        expected = np.array([0., 1.])
        result   = audio.get_time(256, False)
        self.assertTrue(np.array_equal(expected, result))
                
        expected = np.array([0.5, 1.5])
        result   = audio.get_time(256, True)
        self.assertTrue(np.array_equal(expected, result))

        
        audio = FramedAudio(np.arange(512), 44100, 256, 256, centered=True)
        
        expected = np.array([-0.5, 0.5])
        result   = audio.get_time(256, False)
        self.assertTrue(np.array_equal(expected, result))
                
        expected = np.array([0., 1.])
        result   = audio.get_time(256, True)
        self.assertTrue(np.array_equal(expected, result))


    def test_get_num_frames(self):                
        audio = FramedAudio(np.zeros((1,128)), 44100, 256, 128)   
        self.assertEqual(audio.get_num_frames(), 1)
        
        audio = FramedAudio(np.zeros((1,256)), 44100, 256, 128)   
        self.assertEqual(audio.get_num_frames(), 1)
        
        audio = FramedAudio(np.zeros((1,384)), 44100, 256, 128)   
        self.assertEqual(audio.get_num_frames(), 2)
        
        audio = FramedAudio(np.zeros((1,385)), 44100, 256, 128)   
        self.assertEqual(audio.get_num_frames(), 2)
                
        audio = FramedAudio(np.zeros((0,0)),44100, 256, 128)   
        self.assertEqual(audio.get_num_frames(), 1)

        # with centering

    def test_get_frame(self):      
        
        vector = np.arange(1, 7)
        audio = FramedAudio(vector, 44100, 4, 2)   

        frame = audio.get_frame(0)    
        expected =  np.array([1,2,3,4])
        self.assertTrue(np.array_equal(frame, expected))
        
        frame = audio.get_frame(1)    
        expected =  np.array([3,4,5,6])
        self.assertTrue(np.array_equal(frame, expected))
        

    def test_get_raw(self):      
        
        vector = np.arange(1, 7)
        audio = FramedAudio(vector, 44100, 4, 2)  
        raw = audio.get_raw()    

        self.assertTrue(np.array_equal(raw, vector))


    def test_trajectories(self):
        
        vector = np.arange(128)
        audio = FramedAudio(vector, 44100, 4, 2)
        
        traj = np.arange(audio.get_num_frames())
        audio.store_trajectory('traj', traj)

        traj2 = audio.get_trajectory('traj')

        self.assertTrue(np.array_equal(traj, traj2))

        self.assertTrue( audio.get_trajectory('no-key') is None)

    def test_json(self):
        vector = np.arange(128)
        audio = FramedAudio(vector, 44100, 4, 2)
        
        traj = np.arange(audio.get_num_frames())
        audio.store_trajectory('traj', traj)

        audio.save_json('temp.json')
        
        # create another instance with same vector but different 
        audio2 = FramedAudio.from_json('temp.json')

        self.assertTrue( np.array_equal(audio.get_trajectory('traj'), audio2.get_trajectory('traj')))
        self.assertEqual( audio.get_config(), audio2.get_config())
        # todo impl test with from_file 

        src = 'test/audio_sample.ogg'

        audio = FramedAudio.from_file(src, 2048, 1024)

        audio.save_json('temp.json') 

        audio2 = FramedAudio.from_json('temp.json')
        self.assertEqual( audio.get_config(), audio2.get_config())
        self.assertEqual( audio.src_file, audio2.src_file)
        self.assertTrue( np.array_equal(audio.array, audio2.array))


    def test_set_get_config(self):
        vector = np.arange(128)
        audio = FramedAudio(vector, 44100, 4, 2)
        
        cfg = audio.get_config();
        self.assertEqual(cfg['fs'], 44100)
        self.assertEqual(cfg['block-size'], 4)
        self.assertEqual(cfg['hop-size'], 2)
        
        vector = np.arange(128)
        audio2 = FramedAudio(vector, 123, 2, 4)

        audio2.set_config(cfg)
        self.assertEqual(audio2.get_config(), cfg)





    

if __name__ == '__main__':   
    unittest.main()

