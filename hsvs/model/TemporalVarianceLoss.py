
import tensorflow as tf

class TemporalVarianceLoss(tf.keras.layers.Layer):

    def __init__(self, weight, **kwargs):
        super(TemporalVarianceLoss, self).__init__(**kwargs)

        self.kernel = tf.reshape([-0.25, 0.5, -0.25], shape=[3,1,1])
        self.weight = weight

    def build(self, input_shape):
        super(TemporalVarianceLoss, self).build(input_shape)
        
    def call(self, input):
        
        # tranpose and expand first dimension
        tranposed = tf.expand_dims(tf.transpose(input), 2)
        
        # apply convolution over batch dimension (now 2nd dimension)
        filtered = tf.nn.convolution(tranposed, self.kernel, padding='SAME')
        
        # remove channel dim and transpose back
        filtered = tf.transpose(tf.squeeze(filtered, axis=2))
        loss = self.weight * tf.reduce_mean(tf.square(filtered))

        # apply weight as 
        self.add_loss(loss)
        self.add_metric(loss, aggregation='mean', name='var_loss')
        return input




