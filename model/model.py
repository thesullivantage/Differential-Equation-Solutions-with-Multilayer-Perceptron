import sys
import tensorflow.keras as keras
import tensorflow as tf

class FeedForward(keras.layers.Layer):
    '''
    subclassed implementation, with default values, of feed-forward (dense) layer
    This is a little extraneous, given the existence of the Dense class. 
    But adding it anyway for educational purposes 
    '''
    def __init__(self, embed_dim=32, input_dim=1, activation='sigmoid'):
        '''
        
        '''
        if activation == 'sigmoid':
            self.activation = tf.nn.sigmoid
        else:
            print('Error: Activation function passed not supported; please try again.')
            sys.exit(1)
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim, embed_dim), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(embed_dim,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)

class MLP(keras.layers.Layer):
    def __init__(self, num_embed_layer=2):
        


# class Decoder(layers.Layer):
#     """Converts z, the encoded digit vector, back into a readable digit."""

#     def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
#         self.dense_output = layers.Dense(original_dim, activation="sigmoid")

#     def call(self, inputs):
#         x = self.dense_proj(inputs)
#         return self.dense_output(x)


# class ModelSubClassing(keras.Model):
#     def __init__(self, num_classes):
#         super().__init__()
#         # define all layers in init
#         # Layer of Block 1
#         self.conv1 = keras.layers.Conv2D(
#                           32, 3, strides=2, activation="relu"
#                      )
#         self.max1  = keras.layers.MaxPooling2D(3)
#         self.bn1   = keras.layers.BatchNormalization()

#         # Layer of Block 2
#         self.conv2 = keras.layers.Conv2D(64, 3, activation="relu")
#         self.bn2   = keras.layers.BatchNormalization()
#         self.drop  = keras.layers.Dropout(0.3)

#         # GAP, followed by Classifier
#         self.gap   = keras.layers.GlobalAveragePooling2D()
#         self.dense = keras.layers.Dense(num_classes)


#     def call(self, input_tensor, training=False):
#         # forward pass: block 1 
#         x = self.conv1(input_tensor)
#         x = self.max1(x)
#         x = self.bn1(x)

#         # forward pass: block 2 
#         x = self.conv2(x)
#         x = self.bn2(x)

#         # droput followed by gap and classifier
#         x = self.drop(x)
#         x = self.gap(x)
#         return self.dense(x)