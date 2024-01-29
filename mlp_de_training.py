import sys
import os
from importlib import reload

sys.path.insert(0, 'Differential-Equation-Solutions-with-Multilayer-Perceptron')

# to reload modules as we make changes
reload(loss)
reload(model)

import model_pk
from model_pk import loss, model

import sys
import tensorflow.keras as keras
import tensorflow as tf

### FOR SGD:
# optimizer = keras.optimizers.SGD(learning_rate=1e-3)

### FOR BATCH (not minibatch)
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

mlp = model.MLP(num_embed_layer=2, phys_dimension=1,embed_dim=32)
loss_fn = loss.LossODE(f0=0, dell=1e-4)

# Make simple ODE domain for f'(x) = 2x: STANDARDIZED OR NORMALIZED
phys_dimension = 1
# Range of independent variable values on unit norm interval
start = 0  # Start value
limit = 1  # End value (exclusive)
delta = 0.01  # Step size

# Create a TensorFlow array in the style of np.arange
inputs = tf.reshape(tf.range(start, limit, delta, dtype=tf.float32), (-1, phys_dimension))

# Training loop
num_epochs = 25  # Define the number of epochs

### Training using SGD optimization (MSE), without replacement
### n (dim. inputs) gradient updates
for epoch in range(1, num_epochs+1):
    ### increment squared loss over entire epoch (dataset)
    epoch_total_loss = 0
    for input_example in inputs:
        # seems easiest to resize individual data points
        input_example_reshaped = tf.reshape(input_example, (1,1))
        with tf.GradientTape() as tape:
            # Compute the loss for the current input example (squared error)
            loss_curr = loss_fn.compute_loss_element(mlp, input_example_reshaped)
            epoch_total_loss += loss_curr
        # Compute gradients and update model parameters
        gradients = tape.gradient(loss_curr, mlp.trainable_variables)
        optimizer.apply_gradients(zip(gradients, mlp.trainable_variables))
    epoch_avg_loss = epoch_total_loss / tf.cast(inputs.shape[0], tf.float32)
    print(f'Epoch {epoch} Average Loss: ', epoch_avg_loss[0][0])
    
### Training using batch optimization, MSE, Adam
### single gradient update (for now) per batch/epoch
inputs = tf.reshape(tf.range(start, limit, delta, dtype=tf.float32), (-1, phys_dimension))

for epoch in range(1, num_epochs+1):
    ### increment squared loss over entire epoch (dataset)
    
    
    
    for input_example in inputs:
        # seems easiest to resize individual data points
        input_example_reshaped = tf.reshape(input_example, (1,1))
        with tf.GradientTape() as tape:
            # Compute the loss for the current input example (squared error)
            loss_curr = loss_fn.compute_loss_element(mlp, input_example_reshaped)
            epoch_total_loss += loss_curr
        # Compute gradients and update model parameters
        gradients = tape.gradient(loss_curr, mlp.trainable_variables)
        optimizer.apply_gradients(zip(gradients, mlp.trainable_variables))
    epoch_avg_loss = epoch_total_loss / tf.cast(inputs.shape[0], tf.float32)
    print(f'Epoch {epoch} Average Loss: ', epoch_avg_loss[0][0])
    