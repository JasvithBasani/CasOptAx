#Jasvith Raj Basani
#
#
#History
# 16/12/2022 - Created this File


import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from functools import partial
import optax
import matplotlib.pyplot as plt

class Cost_Functions:

    @partial(jit, static_argnums = (0, ))
    def frob_norm(self, mat_1, mat_2):
        r"""
        Frobenius Norm between matrices

        :param mat_1:
        :param mat_2:
        :return:
        """
        assert jnp.shape(mat_1) == jnp.shape(mat_2)
        N = jnp.size(mat_1, 0)
        return jnp.linalg.norm(mat_1 - mat_2) / jnp.sqrt(N)

    @partial(jit, static_argnums=(0,))
    def fidelity_pulses(self, pulse_1, pulse_2, dt):
        r"""
        Fidelity between pulses

        :param pulse_1:
        :param pulse_2:
        :param dt:
        :return:
        """
        fid = jnp.trapz(jnp.conj(pulse_1) * pulse_2) * dt
        return jnp.abs(fid)

    @partial(jit, static_argnums=(0,))
    def fidelity_states(self, state_1, state_2, dk):
        r"""
        Fidelity between 2 states, each with a two-photon wavefunction

        :param state_1:
        :param state_2:
        :param dk:
        :return:
        """
        fid_vals = jax.vmap(lambda i: (jnp.trapz(jnp.trapz(state_1[i] * jnp.conj(state_2[i]))) * dk ** 2))(jnp.arange(0, len(state_1)))
        return fid_vals, jnp.sum(jnp.abs(fid_vals))

    def check_unitary(self, mat):
        r"""
        Check if matrix is unitary

        :param mat:
        :return:
        """
        N = jnp.size(mat, 0)
        if (jnp.allclose(jnp.abs(jnp.matmul(mat, jnp.transpose(jnp.conj(mat)))), jnp.eye(N), atol=1e-5)) and (
        jnp.isclose(jnp.abs(jnp.linalg.det(mat)), 1, atol=1e-5)):
            print("Unitary")
        else:
            print("Not Unitary")
        return None

    def mean_squared_error(self, vec_1, vec_2):
        assert jnp.shape(vec_1) == jnp.shape(vec_2)
        return jnp.mean(jnp.abs(vec_1 - vec_2) ** 2)

    def categorical_cross_entropy(self, vec_1, vec_2):
        assert jnp.shape(vec_1) == jnp.shape(vec_2)
        return -jnp.mean(vec_1 * jax.nn.log_softmax(vec_2, axis=-1))


class Optimizer:

    @partial(jit, static_argnums=(0, 1,))
    def calc_gradient(self, func, param_num, *args):
        r"""
        Calculates and returns the value of a function and its gradient
        Uses jax's value_and_grad function, subject to updates in method
        NOTE: has_aux is True, indicating function shoud provide loss and auxilary outputs



        :param func: python function to execute and calulate gradients of (typically the loss function)
        :param param_num: syntax - '(num_0, num_1 ....)': index of parameters to optimize, in braces. Order to be followed in *args
        :param args: arguments that go into func
        :return:
        """
        (loss_val, _), grads = value_and_grad(func, param_num, has_aux=True)(*args)
        return loss_val, _, grads

    def run_optimization(self, func, optimizer_type, lr, num_epochs, params_to_diff, args, jit_compile=True, plot_loss=[False, 'linear'], save_history=False):
        raise NotImplementedError()