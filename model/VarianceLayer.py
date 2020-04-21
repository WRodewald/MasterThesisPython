
import tensorflow as tf

class VarianceLayer(tf.keras.layers.Layer):

    def __init__(self, num_samples, **kwargs):
        super(VarianceLayer, self).__init__(**kwargs)

        # store number of samples in dataset to allocate the trainable matrix later on
        self.num_samples = num_samples

    def build(self, input_shape):
        super(VarianceLayer, self).build(input_shape)
        
        # allocate trainable variables for N samples / K parameters
        shape = tf.TensorShape([self.num_samples, input_shape[1:][0]])
        self.variance = tf.Variable(initial_value=tf.zeros((shape)))

    def call(self, input):
        
        self.add_loss(100000 * tf.square(self.variance)) # add weighted loss from variance
        return input + self.variance




