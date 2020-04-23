
import tensorflow as tf


class SliceLayer(tf.keras.layers.Layer):
    """Utility Slice layer to slice 2-dimensional input tensors """

    # slice_indices: Nx2 dimensional vector
    def __init__(self, slice_indices,  **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
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
        

