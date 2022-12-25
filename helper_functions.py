# -*- coding: utf-8 -*-
"""helper_functions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CJdEimeIltqTRTUH0kGohEUbEe-g2lVT
"""

#Jasvith Raj Basani
#
#
#History
# 16/12/2022 - Created this File
# 24/12/2022 - Optimization support

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from functools import partial
!pip install optax
import optax

class Cost_Functions:

  def __init__(self):
    r"""
    Class of cost functions
    For now, includes:
    1) Frobenius norm for matrix distance
    2) Fidelity between states
    """

  def frob_norm(self, mat_1, mat_2):
    r"""
    Frobenius norm between matrices
    """
    assert jnp.shape(mat_1) == jnp.shape(mat_2)
    N = jnp.size(mat_1, 0)
    return jnp.linalg.norm(mat_1 - mat_2)/jnp.sqrt(N)

  def fidelity_pulses(self, pulse_1, pulse_2, dt):
    fid = jnp.trapz(jnp.conj(pulse_1) * pulse_2) * dt
    return fid

  def fidelity_states(self, state_1, state_2):
    r"""
    I dont even know what a state looks like as a data structure
    """
    raise NotImplementedError()
  
  def check_unitary(self, mat):
    N = jnp.size(mat, 0)
    if (jnp.allclose(jnp.matmul(mat, jnp.transpose(jnp.conj(mat))), jnp.eye(N)) and jnp.linalg.det(mat) == 1):
      print ("Unitary")
    else:
      print ("Not Unitary")
    return None

  def mean_squared_error(self, vec_1, vec_2):
    assert jnp.shape(vec_1) == jnp.shape(vec_2)
    return jnp.mean(jnp.abs(vec_1 - vec_2)**2)

  def categorical_cross_entropy(self, vec_1, vec_2):
    assert jnp.shape(vec_1) == jnp.shape(vec_2)
    return -jnp.mean(vec_1 * jax.nn.log_softmax(vec_2, axis = -1))

class Pulse_Maker:

  def __init__(self):
    r"""
    Class of pre-defined pulses
    For now, includes: gaussian, exponential, lorentzian
    """

  def gaussian(self, k, mu, sigma):
    r"""
    Returns normalized pulse of the form: 
    $(\frac{2}{\pi})^{0.25} * 1/sigma^{0.5} * e^{-(x - mu)^{2}/(sigma^{2})}$
    """
    #return (2/jnp.pi) ** 0.25 * 1/sigma**0.5 * jnp.exp(-(k - mu)**2/(sigma**2))
    y = jnp.exp(-(k - mu)**2/(sigma**2))
    return y/(jnp.sqrt(jnp.trapz(y ** 2)))

  def exponential(self, t, kappa):
    r"""
    Returns normalized pulse of the form: 
    $\sqrt(2*kappa) * e^{-\kappa t}$
    """
    #return ((2 * kappa) ** 0.5) * jnp.exp(-kappa * t)
    y = jnp.exp(-kappa * t)
    return y/(jnp.sqrt(jnp.trapz(y ** 2)))

  def lorentzian(self, k, mu, kappa):
    r"""
    Returns normalized lorentzian pulse
    """
    y = (kappa)/((k - mu)**2 + (kappa)**2)
    return (y)/(jnp.sqrt(jnp.trapz(y ** 2)))

  def hyperbolic_secant(self, k):
    raise NotImplementedError()

class Optimizer:

  def __init__(self):
    r"""
    Class of functions to aid gradient-based optimization
    Includes jax-based gradient calculation and update function based optimizers provided by the optax library
    """

  @partial(jit, static_argnums = (0, 1, ))
  def calc_gradient(self, func, param_num, *args):

    r"""
    Calculates and returns the value of a function and its gradient
    Uses jax's value_and_grad function, subject to updates in method
    NOTE: has_aux is True, indicating function shoud provide loss and auxilary outputs
    param func: python function to execute and calulate gradients of (typically the loss function)
    param param_num: syntax - '(num_0, num_1 ....)': index of parameters to optimize, in braces. Order to be followed in *args
    param *args: arguments that go into func
    """
    (loss_val, _), grads = value_and_grad(func, param_num, has_aux = True)(*args)
    return loss_val, _, grads


  def run_optimization(self, func, optimizer_type, lr, num_epochs, params_to_diff, args, save_history = False):

    r"""
    Runs optimization and returns the optimized parameters
    param func: python function to execute and calulate gradients of (typically the loss function)
    param optimizer_type: Type of optimizer to be used. Presently, only Adam, Adamax, SGD, RMSprop, Adagrad are supported. More optimizers and documentation can be found here: https://optax.readthedocs.io/en/latest/api.html
    param lr: learning rate alpha (no decay support for now)
    param num_epochs: Number of epochs to run the optimization
    param params_to_diff: syntax - '(num_0, num_1 ....)': index of parameters to optimize, in braces. Order to be followed in args
    param save_history: Bool, to save model parameters and history
    param args: syntax - '[param_1, param_2, ....]': list of parameters that go into func, order to be maintained for params_to_diff
    """
    if optimizer_type == 'adam':
      opt = optax.adam(learning_rate = lr)
    elif optimizer_type == 'sgd':
      opt = optax.sgd(learning_rate = lr)
    elif optimizer_type == 'rmsprop':
      opt = optax.rmsprop(learning_rate = lr)
    elif optimizer_type == 'adamax':
      opt = optax.adamax(learning_rate = lr)
    elif optimizer_type == 'adagrad':
      opt = optax.adagrad(learning_rate = lr)

    params_to_optimize = []
    loss_data = []
    auxilary_data = []
    opt_state_hist = []
    args_hist = []

    for i in params_to_diff:
      params_to_optimize.append(args[i])
    params_to_optimize = tuple(params_to_optimize)

    opt_state = opt.init(params_to_optimize)

    @partial(jit, static_argnums = (0, 1, ))
    def calc_grad(func, params_to_diff, args):
      (loss_val, _), grads = value_and_grad(func, params_to_diff, has_aux = True)(*args)
      return loss_val, _, grads

    @partial(jit, static_argnums = (0, 1, ))
    def update(func, params_to_diff, params_to_optimize, opt_state, args):
      loss_val, _, grads = calc_grad(func, params_to_diff, args)
      updates, opt_state = opt.update(grads, opt_state)
      params_to_optimize = optax.apply_updates(params_to_optimize, updates)
      for param in params_to_optimize:
        for j in params_to_diff:
          args[j] = param
      return loss_val, _, opt_state, args
      
    loss_val, _, opt_state_new, args_new = update(func, params_to_diff, params_to_optimize, opt_state, args)

    if save_history == True:
      for epoch in range(num_epochs):
        loss_val, _, opt_state_new, args_new = update(func, params_to_diff, params_to_optimize, opt_state_new, args_new)
        loss_data.append(loss_val)
        auxilary_data.append(_)
        '''Uncomment the below 2 lines to save model history as well'''
        #opt_state_hist.append(opt_state_new)
        #args_hist.append(args_new)
      return loss_data, auxilary_data, args_new
    elif save_history == False:
      for epoch in range(num_epochs):
        loss_val, _, opt_state_new, args_new = update(func, params_to_diff, params_to_optimize, opt_state_new, args_new)
        print (f'Epoch Number: {epoch}, Loss Value: {np.array(loss_val):.8f}, Auxilary Value: {np.array(_)}')
      return args_new


optim_pulse = %time optimizer.run_optimization(test_loss, 'adam', 0.25, 500, (0,), [pulse_1, pulse_2])