

import tensorflow as tf
import numpy as np

'''
Notes: Vectorize Everything (Numpy)
'''

class LossODE(object):
    
    def __init__(self, f0 = 0):
        self.dell = np.sqrt(np.finfo(np.float32).eps)
        self.initial = f0
    ### get different functions 
    ### MARK DEV: issue with 
    
    def approx_eval(self, subclassed_model, input):
        '''
        MARK DEV: DIMENSION FINE
        x*NN(x) + f_0
        '''
        return subclassed_model(input) + self.initial

    def ode_analy(self, input, funcKey = 'place_h'):
        '''
        Evaluate analytical function based on funcKey
        MARK DEV: Probably a better way to do this: cls.dict_of_functs
        '''
        if funcKey == 'place_h':
            return 2*input
    
    def custom_de_loss(self, inputs, model):
        # Vectorized computation of dNN ###
        dNN = (self.approx_eval(model, inputs + self.dell) - self.approx_eval(model, inputs)) / self.dell

        # Vectorized computation of the loss
        loss = tf.square(dNN - self.ode_analy(inputs))

        # Mean of the loss
        return tf.sqrt(tf.reduce_mean(loss))
    
    # # Custom loss function to approximate the derivatives
    # def custom_loss_old(self, inputs, model):
    #     summation = []
    #     for x in np.linspace(-1,1,10):
    #         dNN = (self.approx_eval(model, x+self.dell)-self.approx_eval(model, x))/self.dell
    #         summation.append((dNN - self.ode_analy(x))**2)
    #     return tf.sqrt(tf.reduce_mean(tf.abs(summation)))

    def custom_de_loss(self, inputs, model):
        def compute_loss_element(x):
            '''
            Function to be mapped to each element of tensor of inputs to get loss element.
            '''
            dNN = (self.approx_eval(model, x + self.dell) - self.approx_eval(model, x)) / self.dell
            return tf.square(dNN - self.ode_analy(x))

        # Apply the modular (map) function to each element of the input tensor
        loss_elements = tf.map_fn(compute_loss_element, inputs, dtype=tf.float32)

        # Compute the mean of the loss elements
        return tf.sqrt(tf.reduce_mean(tf.abs(loss_elements)))