# Development Notes on MLP-DE Model - For Client

## Model Notes
- Model written in Python via the Keras subclassing API.
- Adjusted the dimension of inputs to: `(batch_size, number_of_features)`, since this is what Keras input layers expect.
- Epsilon (difference in forward difference numerical calculation) is passed as a parameter. 
    - TODO: implement central difference

## Loss Notes
- Linear formulation of neural network trail solution of DE, including initial condition, makes sense. 
- The loss (objective) function relying mainly upon the numerical calculation of function derivative increases the complexity of this challenge.

## Training Notes - Preliminary
- Training needs to take place on the unit interval (0, 1) for stability of gradient descent.
- Up to this point, I have been implementing SGD: runninng gradient descent for every sample (batch size: 1).
- Training is currently not converging. __ Squared error values (using SGD; not computing mean) values appear consistent with the x-values themselves.__.
- Some thoughts on that:
    - Changing epsilon from the default value to a larger number did not help convergence.
      - Default value: from the [numpy docs](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html): it is currently the 64-bit "difference between 1.0 and the next smallest representable float larger than 1.0."
  - For modifying optimization strategy - could try:
    - Adam optimization using a batch strategy (gradient descent run after one full pass through data - per epoch).
    - Modified SGD using minibatches of data __to get MSE instead of individual squared error values__.
    - NOTE: for modifying optimization strategy (in addition to changing optimizer), keep the following in the same level of indentation:
        - "with tape" loop: listens for used model parameters in forward-pass
        - `gradients = tape.gradient(loss_curr, mlp.trainable_variables)`
        - `optimizer.apply_gradients(zip(gradients, mlp.trainable_variables))`
        - __Therefore, changing optimization strategy requires changing loss functionality.__  
