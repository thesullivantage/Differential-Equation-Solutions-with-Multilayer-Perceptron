
import tensorflow.keras as keras
import tensorflow as tf
from . import loss

# class TrainUtils(keras.layers):
    
    
class Train(object):
    '''
    Training class; executed in jupyter notebook/training script, for now
    '''
    def __init__(self, opt='SGD'):
        loss = loss.LossODE()
        pass
    # def train_step(self):
    #     with tf.GradientTape() as tape:
    #         loss = custom_loss()
    #     trainable_variables=list(weights.values())+list(biases.values())
    #     gradients = tape.gradient(loss, trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, trainable_variables))