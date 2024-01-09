import sys
import tensorflow.keras as keras
import tensorflow as tf

@keras.saving.register_keras_serializable()
class FeedForward(keras.layers.Layer):
    '''
    subclassed implementation, with default values, of feed-forward (dense) layer
    This is a little extraneous, given the existence of the Dense class. 
    But adding it anyway for educational purposes 
    '''
    
    
    def __init__(self, input_dim=1, output_dim=32, activation='sigmoid', output_fromLog=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # for numerical stability, we may wish to not apply logits at output
        # probably won't be necessary given larger context of our model - in our calculations of gradients semi-analytically
        # but including for customizability
        self.logit_output = output_fromLog
        if activation == 'sigmoid':
            self.activation = tf.nn.sigmoid
        else:
            raise ValueError('Error: Activation function passed not supported; please try again.')
    
    def build(self, input_shape):
        '''
        This is the required method signature for build in Keras subclass API: ensures weights are initialized per our 'initializer' specifications
        random normal is fine for weights; zero is fine for biases (constants)
        '''
        self.w = self.add_weight(
            shape=(self.input_dim, self.output_dim), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(self.output_dim,), initializer="zeros", trainable=True)

    def call(self, inputs):
        if not self.logit_output:
            return self.activation(tf.matmul(inputs, self.w) + self.b)
        else:
            return tf.matmul(inputs, self.w) + self.b
        
@keras.saving.register_keras_serializable() 
class MLP(keras.Model):
    '''
    Multi-layer perceptron (ANN) model constructor class using Keras subclassing implementation
    '''
    def __init__(self, num_embed_layer=2, phys_dimension=1, embed_dim=32):
        super().__init__()
        
        if num_embed_layer < 2:
            raise ValueError('Error: Please construct model with at least 2 deep layers.')
        
        self.dense_layers = []
        
        for l in range(num_embed_layer):
            if l == 0:
                # expand from differential equation's physical dimension to embedded dimension
                lay = FeedForward(input_dim=phys_dimension, output_dim=embed_dim)
                self.dense_layers.append(lay)
            elif l == num_embed_layer - 1:
                # collapse from embedded dimension to physical dimension (of differential equation's physical dimension)
                lay = FeedForward(input_dim=embed_dim, output_dim=phys_dimension)
                self.dense_layers.append(lay)
            else:
                # continue progressing through model graph in embedded dimension
                lay = FeedForward(input_dim=embed_dim, output_dim=embed_dim)
                self.dense_layers.append(lay)
        
    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        # returns transformed data after full forward-pass
        return x