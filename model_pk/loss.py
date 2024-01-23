

import tensorflow as tf
import numpy as np

'''
Notes: Vectorize Everything (Numpy)
'''

class LossODE(object):
    
    def __init__(self, f0 = 0, dell = None):
        '''
        MARK DEV (PDE): need multi-dim. BC in addition to IC
        '''
        if dell is None:
            self.dell = np.sqrt(np.finfo(np.float32).eps)
        else: 
            self.dell = dell
        self.initial = f0
    ### get different functions 
    ### MARK DEV: issue with 
    
    def approx_eval(self, model, input):
        '''
        MARK DEV: DIMENSION FINE
        x*NN(x) + f_0
        '''
        return input*model(input) + self.initial

    def ode_analy(self, input, funcKey = 'first_ord_sample'):
        '''
        Evaluate analytical function based on funcKey
        MARK DEV: Probably a better way to do this: cls.dict_of_functs
        '''
        if funcKey == 'first_ord_sample':
            return 2*input
        
    def compute_loss_element(self, model, x):
        '''
        ### MARK DEV: PDE: multi-var tuple instead of x; pass dimensionality too
        Function to be mapped to each element of tensor of inputs to get loss element.
        '''
        ### with batch size of 1 (which is thus far mandatory) returns only the value
        NN_deriv = (self.approx_eval(model, x + self.dell) - self.approx_eval(model, x)) / self.dell
        return tf.math.square(NN_deriv - self.ode_analy(x))
