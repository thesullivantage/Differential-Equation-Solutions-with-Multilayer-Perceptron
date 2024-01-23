from model_pk import *
import tensorflow as tf
import tensorflow.keras as keras

'''
DEV NOTES:
- Need ensure vectorized input
- pass analytical function object to loss objects below
'''
### Try Adam Optimizer
# optimizer = tf.keras.optimizers.Adam()  # You can choose and configure your optimizer
### Try SGD
optimizer = keras.optimizers.SGD(learning_rate=1e-3)

### MARK DEV: refactor here
loss_fn = loss.LossODE()
phys_dimension = 1
mlp = model.MLP(num_embed_layer=2, phys_dimension=phys_dimension,embed_dim=32)

# Range of independent variable values
start = 0  # Start value
limit = 100  # End value (exclusive)
delta = 0.1  # Step size

# Create a TensorFlow array in the style of np.arange
inputs = tf.reshape(tf.range(start, limit, delta, dtype=tf.float32), (-1, 1))

# Training loop

### Trained using SGD optimization (MSE), without replacement
# Training loop

num_epochs = 20  # Define the number of epochs

### Trained using SGD optimization (MSE), without replacement
### n (dim. inputs) gradient updates
for epoch in range(1, num_epochs+1):
    ### increment squared loss over entire epoch (dataset)
    epoch_total_loss = 0
    with tf.GradientTape() as tape:

        for input_example in inputs:
        # input_example_reshaped = tf.reshape(input_example, (1, phys_dimension))input_example_reshaped = tf.reshape(input_example, (1,))
            input_example_reshaped = tf.reshape(input_example, (1,1))
            # Compute the loss for the current input example (squared error)
            loss_curr = loss_fn.compute_loss_element(mlp, input_example_reshaped)
            epoch_total_loss += loss_curr
        # Compute gradients and update model parameters
        mse_epoch = tf.math.sqrt(tf.reduce_mean(epoch_total_loss))
        
    gradients = tape.gradient(mse_epoch, mlp.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mlp.trainable_variables))
    epoch_avg_loss = epoch_total_loss / tf.cast(inputs.shape[0], tf.float32)
    print(f'Epoch {epoch} Average Loss: ', mse_epoch)

### OLD
# for epoch in range(num_epochs):
    
#     for input_example in inputs:
        
#         with tf.GradientTape() as tape:
            
#             # Compute the loss
#             loss_curr = loss.custom_de_loss(inputs, mlp)

#         # Compute gradients and update model weights
#         gradients = tape.gradient(loss_curr, mlp.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, mlp.trainable_variables))

#         # Print loss every epoch (or as needed)
#         print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")