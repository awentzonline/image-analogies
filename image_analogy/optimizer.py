import numpy as np
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b


class Optimizer(object):
    '''The Optimizer optimizes models.'''
    def optimize(self, x, model):
        evaluator = ModelEvaluator(model)
        data_bounds = np.repeat(  # from VGG - there's probaby a nicer way to express this...
            [(-103.939, 255. - 103.939, -116.779, 255.0 - 116.779, -123.68, 255 - 123.68)],
            np.product(x.shape) // 3, axis=0
        ).reshape((np.product(x.shape), 2))
        x, min_val, info = fmin_l_bfgs_b(
            evaluator.loss, x.flatten(),
            fprime=evaluator.grads, maxfun=20, maxiter=20,
            factr=1e7,
            m=4,
            bounds=data_bounds,
            iprint=0)
        return x, min_val, info


class ModelEvaluator(object):
    '''The ModelEvaluator class makes it possible to compute loss and gradients
    in one pass while retrieving them via two separate functions, "loss" and "grads".
    This is done because scipy.optimize requires separate functions for loss and
    gradients, but computing them separately would be inefficient.
    '''
    def __init__(self, model):
        self.loss_value = None
        self.grads_values = None
        self.model = model

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.model.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
