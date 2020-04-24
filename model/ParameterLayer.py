
import tensorflow as tf

class ParameterLayer(tf.keras.layers.Layer):

    def __init__(self, num_samples, num_parameters, is_static = True, **kwargs):
        super(ParameterLayer, self).__init__(**kwargs)

        # store number of samples in dataset to allocate the trainable matrix later on
        self.num_samples = num_samples
        self.num_parameters = num_parameters
        self.is_static = is_static

    def build(self, input_shape):
        super(ParameterLayer, self).build(input_shape)
        
        
        if(self.is_static):
            shape = tf.TensorShape([1, self.num_parameters])
        else:
            shape = tf.TensorShape([self.num_samples, self.num_parameters])

        self.parameters = tf.Variable(initial_value=0.5 * tf.ones((shape)))

    def call(self, input):
        
        if(self.is_static):
            return tf.tile(self.parameters, [self.num_samples, 1])
        else:
            return self.parameters




