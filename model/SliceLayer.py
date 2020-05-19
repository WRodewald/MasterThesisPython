
import tensorflow as tf


class SliceLayer(tf.keras.layers.Layer):
    """Utility Slice layer to slice 2-dimensional input tensors """

    # slice_indices: Nx2 dimensional vector
    def __init__(self, slice_indices=None, slice_lens=None,   **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.slice_indices = slice_indices
        
        if not(slice_lens is None):
            slice_indices = []
            cur_idx = 0
            for slice_len in slice_lens:
                slice_indices.append([cur_idx, cur_idx + slice_len])
                cur_idx += slice_len
                
            self.slice_indices = slice_indices



    def build(self, input_shape):
        super(SliceLayer, self).build(input_shape)

    def call(self, input):
        
        slices = []
        for i in range(len(self.slice_indices)):
            start = self.slice_indices[i][0]
            stop  = self.slice_indices[i][1]
            slices.append(input[:,start:stop])

        return tuple(slices)
        

