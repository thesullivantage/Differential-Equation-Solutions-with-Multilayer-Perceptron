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
mlp = model.MLP(num_embed_layer=2, phys_dimension=1,embed_dim=32)

# Range of independent variable values
start = 0  # Start value
limit = 100  # End value (exclusive)
delta = 0.1  # Step size

# Create a TensorFlow array in the style of np.arange
inputs = tf.reshape(tf.range(start, limit, delta, dtype=tf.float32), (-1, 1))

# Training loop
num_epochs = 25  # Define the number of epochs
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        
        # Compute the loss
        loss_curr = loss.custom_de_loss(inputs, mlp)

    # Compute gradients and update model weights
    gradients = tape.gradient(loss, mlp.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mlp.trainable_variables))

    # Print loss every epoch (or as needed)
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

