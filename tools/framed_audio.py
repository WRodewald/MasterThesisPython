
import numpy as np
import json

from . import audio_io

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

    # generic attributes
    attr = {}



    def __init__(self, audio, fs, block_size, hop_size, centered=False):

        self.array = audio
        self.fs = fs
        self.block_size = block_size
        self.hop_size = hop_size      
        self.centered = centered
        self.attr = {}
            
    @staticmethod
    def from_file(src, block_size, hop_size, centered=False):
        
        array, fs = audio_io.read(src)
        audio = FramedAudio(array, fs, block_size, hop_size, centered)
        audio.src_file = src
        return audio


    # returns the origina audio signal
    def get_raw(self): 
        return self.array
    
    
    # returns the number of frames necessary to fully cover the audio
    def get_num_frames(self):
        offset = int(-0.5 * self.block_size) if self.centered else 0
        return  1 + max(0,int(np.floor(self.array.size - self.block_size - offset)/self.hop_size))


    # returns a time vector for each frame
    # with centered=True, get_time return the time at the center of each frame.
    def get_time(self, fs, centered=True):
        offset = int(-0.5 * self.block_size) if self.centered else 0
        samples = self.hop_size * np.arange(float(self.get_num_frames())) + offset
        if(centered): samples += 0.5 * self.block_size

        return samples / fs

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
        assert attr_val.size == self.get_num_frames(), 'a trajectory must have the same number of samples as frames'
        self.attr[attr_key] = attr_val
        
    def get_trajectory(self, attr_key):
        if(attr_key in self.attr):
            return self.attr[attr_key]

        return None


    def get_config(self):
        print('get_config')

        cfg = {}
        cfg['hop-size']   = self.hop_size
        cfg['block-size'] = self.block_size
        cfg['centered']   = self.centered
        cfg['fs']         = self.fs

        return cfg

    # sets the config with the given dict
    # throws an Exception when this 
    def set_config(self, cfg):      
        print('set_config')

        if(self.get_config() == cfg): 
            return

        self.hop_size   = cfg['hop-size']
        self.block_size = cfg['block-size']
        self.centered   = cfg['centered']
        self.fs         = cfg['fs']

        print(len(self.attr))
        if(len(self.attr) > 0):
            print('FUCK')
            #raise Exception('Changing config with stored trajectories may result in unexpected behaviour')



    def save_json(self, json_file):

        data = {}

        # config
        data['config'] = self.get_config()

        # wav/ogg source file
        if(not self.src_file is None):
            data['src'] = self.src_file

        # trajectories (pitch, etc)
        data['data'] = {}
        for key in self.attr:
            data['data'][key] = self.attr[key].tolist()

        # write
        with open(json_file, 'w+') as outfile:
            json.dump(data, outfile)

    @staticmethod
    def from_json(json_file):

        with open(json_file) as json_dict:       

            data = json.load(json_dict)
            
            # config
            cfg = data['config']

            # attributes
            attr = {}
            for key in data['data']:
                attr[key] = np.asarray(data['data'][key])

            # src file  
            src = None
            if('src' in data):
                src = data['src']

            # create object
            if(not src is None):
                obj = FramedAudio.from_file(src, cfg['block-size'], cfg['hop-size'], cfg['centered'])
                obj.attr = attr
                return obj

            else:
                dummy_src = np.zeros(cfg['block-size'])
                obj = FramedAudio(dummy_src, cfg['fs'], cfg['block-size'], cfg['hop-size'], cfg['centered'])
                obj.attr = attr
                return obj
                

        raise Exception('Could not open json file: ' + json_file)