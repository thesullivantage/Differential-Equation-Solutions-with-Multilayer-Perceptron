from model import *
import tensorflow.keras as keras
import tensorflow as tf


'''
DEV NOTES:
- Need ensure vectorized input
- pass analytical function object to loss objects below
'''
### Try Adam Optimizer
# optimizer = tf.keras.optimizers.Adam()  # You can choose and configure your optimizer
### Try SGD
optimizer = keras.optimizers.SGD(learning_rate=1e-3)

loss = loss.LossODE()
mlp = model.MLP(num_embed_layer=2, phys_dimension=1,embed_dim=32)

# Range of independent variable values
start = 0  # Start value
limit = 100  # End value (exclusive)
delta = 0.1  # Step size

# Create a TensorFlow array in the style of np.arange
inputs = tf.range(start, limit, delta)


# Iterate over the batches of a dataset.

# train_dataset needs to be made
# loss function needs vectorizability

# Training loop
num_epochs = 25  # Define the number of epochs
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        
        # Compute the loss
        loss_curr = model.custom_loss(inputs)

    # Compute gradients and update model weights
    gradients = tape.gradient(loss, mlp.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mlp.trainable_variables))

    # Print loss every epoch (or as needed)
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")
### OLD CODE BELOW:

