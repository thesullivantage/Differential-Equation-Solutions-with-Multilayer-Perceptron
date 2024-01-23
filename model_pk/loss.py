

import tensorflow as tf
import numpy as np

'''
Notes: Vectorize Everything (Numpy)
'''

class LossODE(object):
    
    def __init__(self, f0 = 0):
        '''
        MARK DEV (PDE): need multi-dim. BC in addition to IC
        '''
        self.dell = np.sqrt(np.finfo(np.float32).eps)
        self.initial = f0
    ### get different functions 
    ### MARK DEV: issue with 
    
    def approx_eval(self, model, input):
        '''
        MARK DEV: DIMENSION FINE
        x*NN(x) + f_0
        '''
        return model(input) + self.initial

    def ode_analy(self, input, funcKey = 'place_h'):
        '''
        Evaluate analytical function based on funcKey
        MARK DEV: Probably a better way to do this: cls.dict_of_functs
        '''
        if funcKey == 'place_h':
            return 2*input + self.initial
        
    def compute_loss_element(self, model, x):
        '''
        ### MARK DEV: PDE: multi-var tuple instead of x; pass dimensionality too
        Function to be mapped to each element of tensor of inputs to get loss element.
        '''
        ### with batch size of 1 (which is thus far mandatory) returns only the value
        # val = tf.squeeze(x, axis=0) 
        # print('VAL HERE: ', val)
        ### derivative (via linearization): del_val approach 0
        NN_deriv = (self.approx_eval(model, x + self.dell) - self.approx_eval(model, x)) / self.dell
        loss_sub = NN_deriv - self.ode_analy(x)
        print('Differential', loss_sub)
        return tf.math.square(NN_deriv - self.ode_analy(x))