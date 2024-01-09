import tensorflow as tf
import numpy as np

'''
For this example, we will create an MLP Neural Net with 2 hidden layers, sigmoid activation functions, and a gradient descent optimizer algorithm. Other topologies may also be used, this is just an example. We encourage you to try different approaches to this method.
'''

f0 = 1
inf_s = np.sqrt(np.finfo(np.float32).eps)
learning_rate = 0.01
training_steps = 5000
batch_size = 100
display_step = 500
# Network Parameters
n_input = 1     # input layer number of neurons
n_hidden_1 = 32 # 1st layer number of neurons
n_hidden_2 = 32 # 2nd layer number of neurons
n_output = 1    # output layer number of neurons

### MODEL (INIT): WEIGHTS AND BIASES CAN BE INTIALIZED IN LAYERS, THEMSELVES 
weights = {
'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
'out': tf.Variable(tf.random.normal([n_hidden_2, n_output]))
}
biases = {
'b1': tf.Variable(tf.random.normal([n_hidden_1])),
'b2': tf.Variable(tf.random.normal([n_hidden_2])),
'out': tf.Variable(tf.random.normal([n_output]))
}

### LOOP: WE WANT THIS IN A TRAINING LOOP CONFIG
# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

### MODEL (CALL): WE WANT THIS IN MODEL CODE
# Create model
def multilayer_perceptron(x):
  x = np.array([[[x]]],  dtype='float32')
  layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
  layer_1 = tf.nn.sigmoid(layer_1)
  layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
  layer_2 = tf.nn.sigmoid(layer_2)
  output = tf.matmul(layer_2, weights['out']) + biases['out']
  return output

### MODEL (OBJECTIVE): IN THIS CASE, WANT THIS IN MODEL CODE SINCE WE CALCULATE GRADIENTS (AND SOLUTION) SEMI-ANALYTICALLY
def ode_basic_loss():
  summation = []
  for x in np.linspace(-1,1,10):
    dNN = (g(x+inf_s)-g(x))/inf_s
    summation.append((dNN - f(x))**2)
  return tf.sqrt(tf.reduce_mean(tf.abs(summation)))