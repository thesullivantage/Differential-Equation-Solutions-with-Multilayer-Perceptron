
import tensorflow.keras as keras
import tensorflow as tf

class TrainUtils(keras.layers):
    
    
class Train():
    @classmethod
    def train_step():
        with tf.GradientTape() as tape:
            loss = custom_loss()
        trainable_variables=list(weights.values())+list(biases.values())
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))