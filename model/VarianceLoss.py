
import tensorflow as tf

class VarianceLoss(tf.keras.layers.Layer):

    def __init__(self, weight, **kwargs):
        super(VarianceLoss, self).__init__(**kwargs)

        self.weight = weight

    def build(self, input_shape):
        super(VarianceLoss, self).build(input_shape)
        
    def call(self, input):
        
        difference = input - tf.reduce_mean(input, axis=0, keepdims=True)
        loss = self.weight * tf.reduce_mean(tf.square(difference))

        # apply weight as 
        self.add_loss(loss)
        self.add_metric(loss, aggregation='mean', name='var_loss')
        return input




