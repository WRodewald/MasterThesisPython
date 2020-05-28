
import numpy as np
import json
import os

from . import audio_io


# TODO implement feature to write to a framed audio in addition to reading frames from it?

# wrapper for an audio piece, with support for
#   frames, with hop-size, block-size, 
class FramedAudio:

    # configuration for frame separation
    hop_size=512
    block_size = 1024
    centered = False
    fs = 44100

    # raw audio buffer
    array = np.zeros((1,1))

    # path to a wav file in case the object gets constructed with a file
    src_file = None

    # trajectories, dict of np.arrays' with size == get_num_frames()
    trajectories = {}

    def __init__(self, audio, fs=44100, block_size=2048, hop_size=512, centered=False, config=None):

        self.array = audio
        self.fs = fs
        self.block_size = block_size
        self.hop_size = hop_size      
        self.centered = centered
        self.trajectories = {}

        if(not config is None):
            self.set_config(config)
            
    @staticmethod
    def from_file(src,block_size=2048, hop_size=512, centered=False, config=None):
        
        array, fs = audio_io.read(src)
        audio = FramedAudio(array, fs, block_size, hop_size, centered)
        audio.src_file = src
        
        if(not config is None):
            audio.set_config(config)

        return audio


    # returns the origina audio signal
    def get_raw(self): 
        return self.array
    
    
    # returns the number of frames necessary to fully cover the audio
    def get_num_frames(self):
        offset = int(-0.5 * self.block_size) if self.centered else 0
        return  1 + max(0,int(np.floor(self.array.size - self.block_size - offset)/self.hop_size))


    # returns number of samples
    def get_num_samples(self):
        return self.array.size

    # returns a time vector for each frame
    # with centered=True, get_time return the time at the center of each frame.
    def get_time(self, centered=True):
        offset = int(-0.5 * self.block_size) if self.centered else 0
        samples = self.hop_size * np.arange(float(self.get_num_frames())) + offset
        if(centered): samples += 0.5 * self.block_size

        return samples / self.fs

    # returns the frame <idx>
    def get_frame(self, idx):

        assert(idx < self.get_num_frames())
        offset = int(-0.5 * self.block_size) if self.centered else 0

        first_idx = self.hop_size * idx + offset
        last_idx  = np.min((first_idx + self.block_size, self.array.size))

        pre_pad = 0
        if(first_idx < 0):
            pre_pad = 0 - first_idx
            first_idx = 0
                    
        frame = self.array[first_idx:last_idx]
        return np.pad(frame, (pre_pad, self.block_size-frame.size-pre_pad))

    # sets an attribute to be exported / cached as json
    def store_trajectory(self, attr_key, attr_val):

        assert len(attr_val) == self.get_num_frames(), 'a trajectory must have the same number of samples as frames'
        self.trajectories[attr_key] = attr_val
        
    def get_trajectory(self, attr_key):
        if(attr_key in self.trajectories):
            return self.trajectories[attr_key]

        return None


    def get_config(self):

        cfg = {}
        cfg['hop-size']   = self.hop_size
        cfg['block-size'] = self.block_size
        cfg['centered']   = self.centered
        cfg['fs']         = self.fs

        return cfg

    # sets the config with the given dict
    # throws an Exception when this 
    def set_config(self, cfg):      
        if(self.get_config() == cfg): 
            return

        self.hop_size   = cfg['hop-size']
        self.block_size = cfg['block-size']
        self.centered   = cfg['centered']
        if('fs' in cfg): self.fs = cfg['fs']

        if(len(self.trajectories) > 0):
            print('FUCK')
            #raise Exception('Changing config with stored trajectories may result in unexpected behaviour')

    # checks if all config attribuetes in cfg are 
    def matches_cfg(self, config):

        my_cfg = self.get_config()
        
        all_match = True
        for key in config:
            all_match &= (key in my_cfg) and (my_cfg[key] == config[key])

        return all_match

    def save_json(self, json_file=None):

        data = {}

        # config
        data['config'] = self.get_config()

        # wav/ogg source file
        if(not self.src_file is None):
            data['src'] = self.src_file

        # trajectories (pitch, etc)
        data['data'] = {}
        for key in self.trajectories:
            data['data'][key] = self.trajectories[key].tolist()

        if(json_file == None):
            json_file = os.path.splitext(self.src_file)[0] + '.json'

        # create folder
        json_dir = os.path.dirname(json_file)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        # write
        with open(json_file, 'w+') as outfile:
            json.dump(data, outfile)

    @staticmethod
    def from_json(json_file, wav_replacement = None):

        try:
            with open(json_file) as json_dict:       

                data = json.load(json_dict)
                
                # config
                cfg = data['config']

                # attributes
                trajectories = {}
                for key in data['data']:
                    trajectories[key] = np.asarray(data['data'][key])


                # src file  
                src = wav_replacement
                if('src' in data):
                    src = data['src']

                # create object
                if(not src is None):
                    obj = FramedAudio.from_file(src, config=cfg)
                    obj.trajectories = trajectories
                    return obj

                else:
                    dummy_src = np.zeros(cfg['block-size'])
                    obj = FramedAudio(dummy_src, config=cfg)
                    obj.trajectories = trajectories
                    return obj
        except:
            return None