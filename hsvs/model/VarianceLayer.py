
import tensorflow as tf

class VarianceLayer(tf.keras.layers.Layer):

    def __init__(self, num_samples, weight = 1, **kwargs):
        super(VarianceLayer, self).__init__(**kwargs)

        # store number of samples in dataset to allocate the trainable matrix later on
        self.num_samples = num_samples
        self.weight = weight

    def build(self, input_shape):
        super(VarianceLayer, self).build(input_shape)
        
        # allocate trainable variables for N samples / K parameters
        shape = tf.TensorShape([self.num_samples, input_shape[1:][0]])
        self.variance = tf.Variable(initial_value=tf.zeros((shape)))

    def call(self, input):
        loss = self.weight * tf.reduce_mean(tf.square(self.variance))
        self.add_loss(loss) # add weighted loss from variance
        self.add_metric(loss, aggregation='mean', name='var_loss')
        return input + self.variance




