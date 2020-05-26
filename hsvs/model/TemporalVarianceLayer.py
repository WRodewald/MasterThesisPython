
import tensorflow as tf

class TemporalVarianceLayer(tf.keras.layers.Layer):

    def __init__(self, num_samples, weight, **kwargs):
        super(TemporalVarianceLayer, self).__init__(**kwargs)

        # store number of samples in dataset to allocate the trainable matrix later on
        self.kernel = tf.reshape([-0.25, 0.5, -0.25], shape=[3,1,1])
        self.num_samples = num_samples;
        self.weight = weight

    def build(self, input_shape):
        super(TemporalVarianceLayer, self).build(input_shape)
        
        # allocate trainable variables for N samples / K parameters
        shape = tf.TensorShape([self.num_samples, input_shape[1:][0]])
        self.variance = tf.Variable(initial_value=tf.zeros((shape)))

    def call(self, input):
        
        # tranpose and expand first dimension
        tranposed = tf.expand_dims(tf.transpose(self.variance), 2)
        
        # apply convolution over batch dimension (now 2nd dimension)
        filtered = tf.nn.convolution(tranposed, self.kernel, padding='SAME')
        filtered = tf.nn.convolution(tranposed, self.kernel, padding='SAME')
        filtered = tf.nn.convolution(tranposed, self.kernel, padding='SAME')
        filtered = tf.nn.convolution(tranposed, self.kernel, padding='SAME')
        
        # remove channel dim and transpose back
        filtered = tf.transpose(tf.squeeze(filtered, axis=2))

        loss = self.weight * tf.reduce_mean(tf.square(filtered))
        # apply weight as 
        self.add_loss(loss)
        self.add_metric(loss, aggregation='mean', name='var_loss')
        return input + self.variance




