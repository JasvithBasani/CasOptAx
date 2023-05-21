#Jasvith Raj Basani
#
#
#History
# 19/11/2022 - Created this File
# 16/12/2022 - Working single photon transmission function
# 02/04/2023 - Working two photon transmission function

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import warnings

class TLE:

  def __init__(self,
               k = jnp.linspace(-6, 6, 70)):
    r"""
    Class of differentiable scattering matrix elements, to compute output wavefunctions.
    These coefficients can be found here: https://doi.org/10.1364/PRJ.1.000110
    """
    self.k = k
    self.dk = self.k[2] - self.k[1]
    warnings.warn('Ensure that all frequency arrays are consistant, default k = jnp.arange(-6, 6, 70)')


  @partial(jit, static_argnums = (0, ))
  def S_mat_t(self, k, p, Omega = 0.0, Gamma = 1.0, gamma = 0.0):
    r"""
    Single photon transmission coefficient

    :param k: input frequency
    :param p: output frequency
    :param Omega: detuning of emitter frequency from k
    :param Gamma: coupling rate of emitter to waveguide mode
    :param gamma: loss rate into environment modes
    :return:
    """

    return ((k - Omega - 1j * 0.5 * (Gamma - gamma))/(k - Omega + 1j * 0.5 * (Gamma + gamma)))

  @partial(jit, static_argnums = (0, ))
  def psi_out_t(self, k, p, psi_in, Omega = 0.0, Gamma = 1.0, gamma = 0.0):
    r"""
    Calculates single-photon transmitted wavefunction

    :param k: input frequency
    :param p: output frequency
    :param psi_in: input wavefunction being scattered off TLE
    :param Omega: detuning of emitter frequency from k
    :param Gamma: coupling rate of emitter to waveguide mode
    :param gamma: loss rate into environment modes
    :return:
    """

    return self.S_mat_t(k, p, Omega, Gamma, gamma) * psi_in

  @partial(jit, static_argnums = (0, ))
  def S_mat_r(self, k, p, Omega = 0.0, Gamma = 1.0, gamma = 0.0):
    r"""
    Atomic excitation amplitude

    :param k: input frequency
    :param p: output frequency
    :param Omega: detuning of emitter frequency from k
    :param Gamma: coupling rate of emitter to waveguide mode
    :param gamma: loss rate into environment modes
    :return:
    """

    return ((Gamma ** 0.5)/(k - Omega + 1j * 0.5 * (gamma + Gamma)))

  #@partial(jit, static_argnums=(0,))
  def S_mat_tt_numpy(self, k_1, k_2, p_1, p_2, Omega=0.0, Gamma=1.0, gamma=0.0):
    r"""
    Two-photon transmission coefficient - numpy version (will be deprecated soon).
    Note that this returns (N, N, N, N) size tensor.
    This function is extremely slow right now - code optimization via jax.vmap to be done in S_mat_tt function.
    Ideally, this function should be called only once per unique emitter.

    :param k_1: 1D input frequency distribution of the first photon
    :param k_2: 1D input frequency distribtion of the second photon
    :param p_1: 1D output frequency distribution of the first photon
    :param p_2: 1D output frequency distribution of the second photon
    :param Omega: detuning of the emitter frequency from {k_1}
    :param Gamma: coupling rate of emitter to waveguide modes
    :param gamma: loss rate into environment modes
    :return: The full scattering matrix, linear component and nonlinear component
    """
    warnings.warn('This version of the S_mat_tt function will be deprecated soon')
    k_1, k_2, p_1, p_2 = np.array(k_1), np.array(k_2), np.array(p_1), np.array(p_2)

    # t_k_1 = np.array(self.S_mat_t(k_1, k_1, Omega, Gamma, gamma))
    # s_k_1 = np.array(self.S_mat_r(k_1, k_1, Omega, Gamma, gamma))
    # t_k_2 = np.array(self.S_mat_t(k_2, k_2, Omega, Gamma, gamma))
    # s_k_2 = np.array(self.S_mat_r(k_2, k_2, Omega, Gamma, gamma))
    # t_p_1 = np.array(self.S_mat_t(p_1, p_1, Omega, Gamma, gamma))
    # s_p_1 = np.array(self.S_mat_r(p_1, p_1, Omega, Gamma, gamma))

    S_mat = np.zeros((len(k_1), len(k_2), len(p_1), len(p_2))) + 0j
    linear_mem = np.zeros((len(k_1), len(k_2), len(p_1), len(p_2))) + 0j
    nonlinear_mem = np.zeros((len(k_1), len(k_2), len(p_1), len(p_2))) + 0j
    for nk_1 in range(len(k_1)):
      t_k_1_val = ((k_1 - Omega - 1j * 0.5 * (Gamma - gamma)) / (k_1 - Omega + 1j * 0.5 * (Gamma + gamma)))[nk_1]
      # t_k_1_val = ((k_1 - Omega - 1j * (Gamma - gamma))/(k_1 - Omega + 1j * (Gamma + gamma)))[nk_1]
      s_k_1_val = ((Gamma ** 0.5) / (k_1 - Omega + 1j * 0.5 * (Gamma + gamma)))[nk_1]
      # s_k_1_val = (((2 * Gamma) ** 0.5)/(k_1 - Omega + 1j * (Gamma + gamma)))[nk_1]
      for nk_2 in range(len(k_2)):
        t_k_2_val = ((k_2 - Omega - 1j * 0.5 * (Gamma - gamma)) / (k_2 - Omega + 1j * 0.5 * (Gamma + gamma)))[nk_2]
        # t_k_2_val = ((k_2 - Omega - 1j * (Gamma - gamma))/(k_2 - Omega + 1j * (Gamma + gamma)))[nk_2]
        s_k_2_val = ((Gamma ** 0.5) / (k_2 - Omega + 1j * 0.5 * (Gamma + gamma)))[nk_2]
        # s_k_2_val = (((2 * Gamma) ** 0.5)/(k_2 - Omega + 1j * (Gamma + gamma)))[nk_2]
        for np_1 in range(len(p_1)):
          t_p_1_val = ((p_1 - Omega - 1j * 0.5 * (Gamma - gamma)) / (p_1 - Omega + 1j * 0.5 * (Gamma + gamma)))[np_1]
          # t_p_1_val = ((p_1 - Omega - 1j * (Gamma - gamma))/(p_1 - Omega + 1j * (Gamma + gamma)))[np_1]
          s_p_1_val = ((Gamma ** 0.5) / (p_1 - Omega + 1j * 0.5 * (Gamma + gamma)))[np_1]
          # s_p_1_val = (((2 * Gamma) ** 0.5)/(p_1 - Omega + 1j * (Gamma + gamma)))[np_1]
          p_2_val = k_1[nk_1] + k_2[nk_2] - p_1[np_1]
          # Nonlinear Term
          t_p_2_val = ((p_2_val - Omega - 1j * 0.5 * (Gamma - gamma)) / (p_2_val - Omega + 1j * 0.5 * (Gamma + gamma)))
          # t_p_2_val = ((p_2_val - Omega - 1j * (Gamma - gamma))/(p_2_val - Omega + 1j * (Gamma + gamma)))
          s_p_2_val = ((Gamma ** 0.5) / (p_2_val - Omega + 1j * 0.5 * (Gamma + gamma)))
          # s_p_2_val = (((2 * Gamma) ** 0.5)/(p_2_val - Omega + 1j * (Gamma + gamma)))
          np_2 = np.where(np.abs((k_1 + k_2 - p_1) - p_2_val) < self.dk * 1e-1)
          nonlinear_mem[nk_1, nk_2, np_1, np_2] = 1j * (Gamma ** 0.5) /2 /jnp.pi * s_k_1_val * s_k_2_val * (s_p_1_val + s_p_2_val) / self.dk
          S_mat[nk_1, nk_2, np_1, np_2] = 1j * (Gamma ** 0.5) / 2 / jnp.pi * s_k_1_val * s_k_2_val * (s_p_1_val + s_p_2_val) / self.dk
          # S_mat[nk_1, nk_2, np_1, np_2] = 1j * ((2 * Gamma) ** 0.5) /2 /jnp.pi * s_k_1_val * s_k_2_val * (s_p_1_val + s_p_2_val) / dk
          # Linear Terms
          if np_1 == nk_1:
            linear_mem[nk_1, nk_2, nk_1, nk_2] += t_k_1_val * t_k_2_val /2 /self.dk /self.dk
            S_mat[nk_1, nk_2, nk_1, nk_2] += t_k_1_val * t_k_2_val / 2 / self.dk / self.dk
          if np_1 == nk_2:
            linear_mem[nk_1, nk_2, nk_2, nk_1] += t_k_1_val * t_k_2_val /2 /self.dk /self.dk
            S_mat[nk_1, nk_2, nk_2, nk_1] += t_k_1_val * t_k_2_val / 2 / self.dk / self.dk

    return np.array(S_mat), np.array(linear_mem), np.array(nonlinear_mem)

  def S_mat_tt(self, k_1, k_2, p_1, p_2, Omega = 0.0, Gamma = 1.0, gamma = 0.0):
    r"""
    Two-photon transmission coefficient - jax version.
    Note that this returns (N, N, N, N) size tensor.

    :param k_1: 1D input frequency distribution of the first photon
    :param k_2: 1D input frequency distribtion of the second photon
    :param p_1: 1D output frequency distribution of the first photon
    :param p_2: 1D output frequency distribution of the second photon
    :param Omega: detuning of the emitter frequency from {k_1}
    :param Gamma: coupling rate of emitter to waveguide modes
    :param gamma: loss rate into environment modes
    :return: The full scattering matrix, linear component and nonlinear component
    """

    nk_1, nk_2, np_1 = jnp.arange(0, len(k_1)), jnp.arange(0, len(k_2)), jnp.arange(0, len(p_1))

    @jit
    def t_s(k, Omega, Gamma, gamma, idx):
      t = ((k - Omega - 1j * 0.5 * (Gamma - gamma)) / (k - Omega + 1j * 0.5 * (Gamma + gamma)))[idx]
      s = ((Gamma ** 0.5) / (k - Omega + 1j * 0.5 * (Gamma + gamma)))[idx]
      return t, s

    t_k_1, s_k_1 = jax.vmap(lambda idx: t_s(k_1, Omega, Gamma, gamma, idx))(nk_1)
    t_k_2, s_k_2 = jax.vmap(lambda idx: t_s(k_2, Omega, Gamma, gamma, idx))(nk_2)
    t_p_1, s_p_1 = jax.vmap(lambda idx: t_s(p_1, Omega, Gamma, gamma, idx))(np_1)

    p_2_val = jax.vmap(jax.vmap(jax.vmap(lambda k_1, k_2, p_1: k_1 + k_2 - p_1, in_axes = (None, None, 0)), in_axes = (None, 0, None)), in_axes = (0, None, None))(k_1, k_2, p_1)
    p_2_val = p_2_val.reshape(len(k_1) * len(k_2) * len(p_1))

    #Can't jit this because of the bools - is there a better way to do this?
    def linear_scattering():
      linear_mem = jnp.zeros((len(k_1), len(k_2), len(p_1), len(p_2))) + 0j
      idx_array = jax.vmap(jax.vmap(jax.vmap(lambda nk_1, nk_2, np_1: jnp.array([nk_1, nk_2, np_1]), in_axes = (None, None, 0)), in_axes = (None, 0, None)), in_axes = (0, None, None))(nk_1, nk_2, np_1)
      idx_array = jnp.transpose(jnp.array(idx_array).reshape(len(nk_1) * len(nk_2) * len(np_1), 3))
      idx_array_1 = jnp.transpose(idx_array)[idx_array[0] == idx_array[2]]
      idx_array_2 = jnp.transpose(idx_array)[idx_array[1] == idx_array[2]]

      linear_data = jax.vmap(jax.vmap(jax.vmap(lambda tk_1, tk_2, tp_1: jnp.array([tk_1, tk_2, tp_1]), in_axes = (None, None, 0)), in_axes = (None, 0, None)), in_axes = (0, None, None))(t_k_1, t_k_2, t_p_1)
      linear_data = jnp.transpose(jnp.array(linear_data).reshape(len(nk_1) * len(nk_2) * len(np_1), 3))

      linear_data_1 = jnp.transpose(linear_data)[idx_array[0] == idx_array[2]]
      linear_data_1 = jnp.transpose(linear_data_1)
      linear_data_1 = linear_data_1[0] * linear_data_1[1] / 2 / self.dk ** 2
      linear_data_2 = jnp.transpose(linear_data)[idx_array[1] == idx_array[2]]
      linear_data_2 = jnp.transpose(linear_data_2)
      linear_data_2 = linear_data_2[0] * linear_data_2[1] / 2 / self.dk ** 2

      idx_array_1 = jnp.transpose(idx_array_1)
      idx_array_2 = jnp.transpose(idx_array_2)
      linear_idx_1 = jnp.vstack((idx_array_1[0], idx_array_1[1], idx_array_1[0], idx_array_1[1]))
      linear_idx_2 = jnp.vstack((idx_array_2[0], idx_array_2[1], idx_array_2[1], idx_array_2[0]))

      linear_mem_1 = linear_mem.at[tuple(linear_idx_1)].set(linear_data_1)
      linear_mem_2 = linear_mem.at[tuple(linear_idx_2)].set(linear_data_2)
      return linear_mem_1 + linear_mem_2

    def bound_state(Omega, Gamma, gamma):
      nonlinear_mem = jnp.zeros((len(k_1), len(k_2), len(p_1), len(p_2))) + 0j
      idx_array = jnp.array(jnp.meshgrid(nk_1, nk_2, np_1))
      idx_array = idx_array.reshape(3, len(nk_1) * len(nk_2) * len(np_1))

      p_2_val = jax.vmap(jax.vmap(jax.vmap(lambda k_1, k_2, p_1: k_1 + k_2 - p_1, in_axes = (None, None, 0)), in_axes = (None, 0, None)), in_axes = (0, None, None))(k_1, k_2, p_1)
      p_2_val = p_2_val.reshape(len(nk_1) * len(nk_2) * len(np_1))

      #This line is the bottleneck - think of a way to make this faster
      np_2 = jax.vmap(lambda idx: jnp.where(jnp.abs(k_1 + k_2 - p_1 - p_2_val[idx]) < dk * 0.1, size = 1, fill_value = jnp.nan))(jnp.arange(0, len(p_2_val)))
      np_2 = jnp.array(np_2[0]).reshape(len(p_2_val))
      tf_bools = jnp.array(~jnp.isnan(np_2))
      np_2_data = np_2[tf_bools]
      tf_bools = jnp.concatenate((tf_bools, tf_bools, tf_bools, tf_bools)).reshape(4, len(nk_1) * len(nk_2) * len(np_1))

      idx_array = jnp.concatenate((idx_array[0], idx_array[1], idx_array[2], np_2)).reshape(4, len(nk_1) * len(nk_2) * len(np_1))
      idx_array = jnp.array(idx_array[tf_bools].reshape(4, len(np_2_data)), dtype=jnp.int16)

      def s(k, Omega, Gamma, gamma):
        s = ((Gamma ** 0.5) / (k - Omega + 1j * 0.5 * (Gamma + gamma)))
        return s

      sk_1 = s(k_1, Omega, Gamma, gamma)
      sk_2 = s(k_2, Omega, Gamma, gamma)
      sp_1 = s(p_1, Omega, Gamma, gamma)
      sp_2 = s(p_2_val, Omega, Gamma, gamma)

      s_array = jnp.array(jnp.meshgrid(sk_1, sk_2, sp_1))
      s_array = s_array.reshape(3, len(sk_1) * len(sk_2) * len(sp_1))
      s_array = jnp.concatenate((s_array[0], s_array[1], s_array[2], sp_2)).reshape(4, len(sk_1) * len(sk_2) * len(sp_1))
      s_array = s_array[tf_bools].reshape(4, len(np_2_data))

      def make_nl_data(s_array, Gamma):
        nl_data = 1j * (Gamma ** 0.5) / (2 * jnp.pi) * s_array[0] * s_array[1] * (s_array[2] + s_array[3]) / self.dk
        return nl_data

      bound_state_data = make_nl_data(s_array, Gamma)
      nonlinear_mem = nonlinear_mem.at[tuple(idx_array)].set(bound_state_data)
      return nonlinear_mem

    linear_scatter_data = linear_scattering()
    nonlinear_scatter_data = bound_state(Omega, Gamma, gamma)
    S_matrix = linear_scatter_data + nonlinear_scatter_data
    return S_matrix, linear_scatter_data, nonlinear_scatter_data

  @partial(jit, static_argnums = (0, 3, ))
  def psi_out_tt(self, psi_in, S_matrix, domain_check = False):
    r"""
    Transmitted two-photon output wavefunction, via tensordot/einsum of S_matrix with input wavefunction

    :param psi_in: Input two-photon wavefunction, function of {k_1, k_2}
    :param S_matrix: Scattering matrix of size (N, N, N, N), calculated from function S_mat_tt
    :param domain_check: If true: prints the normalization constant of the output wavefunction - should be as close to 1 as possible
            Note: This does not work if function is jitted
    :return:
    """

    #psi_out = jnp.tensordot(S_matrix, psi_in, axes=([2, 3], [0, 1])) * self.dk * self.dk
    psi_out = jnp.einsum('ijkl, kl -> ij', S_matrix, psi_in) * self.dk**2

    def check(psi_out):
      r"""
      2D integral of the absolute value of the output wavefunction.
      Normalization constant shoud be approximately 1.
      If not, increase the range of {k_1, k_2} or {p_1, p_2} or reduce dt.
      """
      return jnp.trapz(jnp.trapz(jnp.abs(psi_out))) * self.dt**2

    if domain_check == True:
      print (f'Normalization constant of output wavefunction is: {check(psi_out)}')

    return psi_out
