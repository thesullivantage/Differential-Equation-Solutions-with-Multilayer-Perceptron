

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

'''
Notes: Vectorize Everything (Numpy)
'''

class LossODE(object):
    
    def __init__(self):
        self.dell = np.sqrt(np.finfo(np.float32).eps)
        self.loss_summation = []
    ### get different functions 
    ### MARK DEV: issue with 
    
    def reset_loss(self):
        '''
        reset loss array
        '''
        self.loss_summation = []
    
    def approx_eval(self, subclassed_model, inputs, initial):
        '''
        x*NN(x) + f_0
        '''
        return subclassed_model(inputs) + initial

    def ode_analy(self, inputs, funcKey = 'place_h'):
        '''
        Evaluate analytical function based on funcKey
        MARK DEV: Probably a better way to do this: cls.dict_of_functs
        '''
        if funcKey == 'place_h':
            return 2*(inputs)
    
    def custom_loss(self, inputs):
        '''
        MARK DEV: need this for 2d funcs via vectorized (tensorized) TF logic
        '''
        for x in inputs:
        # for x in np.linspace(-1,1,10):
            dNN = (self.approx_eval(x+self.dell)-self.approx_eval(x))/self.dell
            self.loss_summation.append((dNN - self.ode_analy(x))**2)
        return tf.sqrt(tf.reduce_mean(tf.abs(self.loss_summation)))
        