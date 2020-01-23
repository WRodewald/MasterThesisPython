
import numpy as np

# provides a 
class FramedAudio:

    hop_size=512
    block_size = 1024

    offset = 0

    array = np.zeros((1,1))


    def __init__(self, audio, block_size, hop_size, centered=False):
        self.array = audio
        self.block_size = block_size
        self.hop_size = hop_size      

        # if centered, we shift the hops so that the center of the first hop alligns with the t=0 mark
        if(centered):
            self.offset = int(-0.5 * self.block_size)



    def get_raw(self): 
        return self.array

    def get_num_frames(self):
        return  1 + max(0,int(np.floor(self.array.size - self.block_size - self.offset)/self.hop_size))


    def get_frame(self, idx):

        assert(idx < self.get_num_frames())

        first_idx = self.hop_size * idx + self.offset
        last_idx  = np.min((first_idx + self.block_size, self.array.size))

        pre_pad = 0
        if(first_idx < 0):
            pre_pad = 0 - first_idx
            first_idx = 0
        
            
        frame = self.array[first_idx:last_idx]
        return np.pad(frame, (pre_pad, self.block_size-frame.size-pre_pad))