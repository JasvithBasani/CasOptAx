# History
# 15/11/2022 - Created this File
# 14/12/2022 - Added Clements Support
# 15/9/2023 - Matrix permanent calculation

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import block_diag
import scipy.stats
from functools import partial, reduce

class Linear_Optics:

  def __init__(self,
               N_modes: jnp.int16 = None,
               ):
    
    r"""
    Class of differentiable linear optics tools.
    In particular, returns matrices implemented by photonic integrated meshes
    
    Args:
      N_modes: Number of spatial modes input to the network.
    """
    self.N_modes = N_modes

  def haar_mat(self, N) -> jnp.ndarray:
    r"""

    :param N: Number of spatial modes input to the network.
    :return: Haar random unitary matrix
    """
    return scipy.stats.unitary_group.rvs(N) + 0j


  @partial(jit, static_argnums = (0, ))
  def MZI(self, theta, phi, alpha = 0 + 0j, beta = 0 + 0j) -> jnp.ndarray:
    r"""
    Single MZI transfer function, to return a 2x2 unitary transformation

    :param theta: MZI phase shift $\theta$
    :param phi: MZI phase shift $\phi$
    :param alpha: Directional coupler error $\alpha$
    :param beta: Directional coupler error $\beta$
    :return: Matrix given by: eq (5) in https://doi.org/10.1364/OPTICA.424052
    """
    assert self.N_modes == 2

    t_00 = jnp.exp(1j * phi) * (jnp.cos(alpha - beta) * jnp.sin(theta/2) + 1j * jnp.sin(alpha + beta) * jnp.cos(theta/2))
    t_01 = (jnp.cos(alpha + beta) * jnp.cos(theta/2) + 1j * jnp.sin(alpha - beta) * jnp.sin(theta/2))
    t_10 = jnp.exp(1j * phi) * (jnp.cos(alpha + beta) * jnp.cos(theta/2) - 1j * jnp.sin(alpha - beta) * jnp.sin(theta/2))
    t_11 = -(jnp.cos(alpha - beta) * jnp.sin(theta/2) - 1j * jnp.sin(alpha + beta) * jnp.cos(theta/2))
    T = 1j * jnp.exp(1j * theta/2) * jnp.array([[t_00, t_01],
                                                [t_10, t_11]])
    return T

  def get_MZI_phases(self, U):
    raise NotImplementedError()

  def set_MZI(self, U, alpha, beta):
    raise NotImplementedError()

  def scf_matrix(self, theta, phi, D, alpha, beta):
    raise NotImplementedError()

  @partial(jit, static_argnums = (0, ))
  def clements_matrix(self, theta, phi, D, alpha, beta) -> jnp.ndarray:
    r"""
    Differentiable mesh network based on the Clements (rectangular) decomposition,
    to return NxN unitary transformation.
    MZIs are indexed from top to bottom and left to right

    :param theta: 1D array of phase shift values
    :param phi: 1D array of phase shift values
    :param D: 1D array for output phase screen
    :param alpha: 1D array for directional coupler errors
    :param beta: 1D array for directional coupler errors
    :return: Unitary matrix configured to the Clements mesh, given the phases
    """

    assert (len(theta)) == int(self.N_modes * (self.N_modes - 1)/2)
    assert (len(phi)) == int(self.N_modes * (self.N_modes - 1)/2)
    assert (len(alpha)) == int(self.N_modes * (self.N_modes - 1)/2)
    assert (len(beta)) == int(self.N_modes * (self.N_modes - 1)/2)
    assert (len(D)) == int(self.N_modes)

    def even_clements(N_modes, theta, phi, D, alpha, beta):
      r"""
      Clements matrix for even N_modes
      """
      col_matrices = []
      idx = 0
      for i in range(N_modes):
        if i%2 == 0:
          t = theta[idx : idx + N_modes//2]
          p = phi[idx : idx + N_modes//2]
          a = alpha[idx : idx + N_modes//2]
          b = beta[idx : idx + N_modes//2]
          idx = idx + N_modes//2

          a_d = jnp.diag(jnp.dstack((jnp.cos(jnp.pi/4 + a), jnp.cos(jnp.pi/4 + a))).reshape(N_modes))
          a_01 = jnp.roll(jnp.diag(jnp.dstack((1j * jnp.sin(np.pi/4 + a), jnp.zeros(N_modes//2))).reshape(N_modes)), 1, axis = 1)
          a_10 = jnp.roll(jnp.diag(jnp.dstack((jnp.zeros(N_modes//2), 1j * jnp.sin(jnp.pi/4 + a))).reshape(N_modes)), -1, axis = 1)
          H_a = a_d + a_01 + a_10

          b_d = jnp.diag(jnp.dstack((jnp.cos(jnp.pi/4 + b), jnp.cos(jnp.pi/4 + b))).reshape(N_modes))
          b_01 = jnp.roll(jnp.diag(jnp.dstack((1j * jnp.sin(jnp.pi/4 + b), np.zeros(N_modes//2))).reshape(N_modes)), 1, axis = 1)
          b_10 = jnp.roll(jnp.diag(jnp.dstack((jnp.zeros(N_modes//2), 1j * jnp.sin(jnp.pi/4 + b))).reshape(N_modes)), -1, axis = 1)
          H_b = b_d + b_01 + b_10

          Theta = jnp.dstack((jnp.exp(1j * t), jnp.ones(N_modes//2))).reshape(N_modes)
          Phi = jnp.dstack((jnp.exp(1j * p), jnp.ones(N_modes//2))).reshape(N_modes)

        else:
          t = theta[idx : idx + N_modes//2 - 1]
          p = phi[idx : idx + N_modes//2 - 1]
          a = alpha[idx : idx + N_modes//2 - 1]
          b = beta[idx : idx + N_modes//2 - 1]
          idx = idx + (N_modes//2 - 1)

          a_d = jnp.diag(jnp.dstack((jnp.cos(jnp.pi/4 + a), jnp.cos(jnp.pi/4 + a))).reshape(N_modes - 2))
          a_01 = jnp.roll(jnp.diag(jnp.dstack((1j * jnp.sin(np.pi/4 + a), jnp.zeros(N_modes//2 - 1))).reshape(N_modes - 2)), 1, axis = 1)
          a_10 = jnp.roll(jnp.diag(jnp.dstack((jnp.zeros(N_modes//2 - 1), 1j * jnp.sin(jnp.pi/4 + a))).reshape(N_modes - 2)), -1, axis = 1)
          H_a = a_d + a_01 + a_10

          b_d = jnp.diag(jnp.dstack((jnp.cos(jnp.pi/4 + b), jnp.cos(jnp.pi/4 + b))).reshape(N_modes - 2))
          b_01 = jnp.roll(jnp.diag(jnp.dstack((1j * jnp.sin(jnp.pi/4 + b), np.zeros(N_modes//2 - 1))).reshape(N_modes - 2)), 1, axis = 1)
          b_10 = jnp.roll(jnp.diag(jnp.dstack((jnp.zeros(N_modes//2 - 1), 1j * jnp.sin(jnp.pi/4 + b))).reshape(N_modes - 2)), -1, axis = 1)
          H_b = b_d + b_01 + b_10

          H_a = block_diag(jnp.array([1]), H_a, jnp.array([1]))
          H_b = block_diag(jnp.array([1]), H_b, jnp.array([1]))

          Theta = jnp.hstack((jnp.array(1, dtype = 'complex64'), jnp.dstack((jnp.exp(1j * t), jnp.ones(N_modes//2 - 1))).reshape(N_modes - 2), jnp.array(1, dtype = 'complex64')))
          Phi = jnp.hstack((jnp.array(1, dtype = 'complex64'), jnp.dstack((jnp.exp(1j * p), jnp.ones(N_modes//2 - 1))).reshape(N_modes - 2), jnp.array(1, dtype = 'complex64')))


        M = H_b @ jnp.diag(Theta) @ H_a @ jnp.diag(Phi)
        col_matrices.append(M)
      return jnp.diag(jnp.exp(1j * D)) @ reduce(jnp.matmul, col_matrices[::-1])
    
    def odd_clements(N_modes, theta, phi, D, alpha, beta):
      r"""
      Clements matrix for odd N_modes
      """
      col_matrices = []
      idx = 0
      for i in range(N_modes):
        if i%2 == 0:
          t = theta[idx : idx + N_modes//2]
          p = phi[idx : idx + N_modes//2]
          a = alpha[idx : idx + N_modes//2]
          b = beta[idx : idx + N_modes//2]
          idx = idx + N_modes//2

          a_d = jnp.diag(jnp.dstack((jnp.cos(jnp.pi/4 + a), jnp.cos(jnp.pi/4 + a))).reshape(N_modes - 1))
          a_01 = jnp.roll(jnp.diag(jnp.dstack((1j * jnp.sin(np.pi/4 + a), jnp.zeros(N_modes//2))).reshape(N_modes - 1)), 1, axis = 1)
          a_10 = jnp.roll(jnp.diag(jnp.dstack((jnp.zeros(N_modes//2), 1j * jnp.sin(jnp.pi/4 + a))).reshape(N_modes - 1)), -1, axis = 1)
          H_a = a_d + a_01 + a_10

          b_d = jnp.diag(jnp.dstack((jnp.cos(jnp.pi/4 + b), jnp.cos(jnp.pi/4 + b))).reshape(N_modes - 1))
          b_01 = jnp.roll(jnp.diag(jnp.dstack((1j * jnp.sin(jnp.pi/4 + b), np.zeros(N_modes//2))).reshape(N_modes - 1)), 1, axis = 1)
          b_10 = jnp.roll(jnp.diag(jnp.dstack((jnp.zeros(N_modes//2), 1j * jnp.sin(jnp.pi/4 + b))).reshape(N_modes - 1)), -1, axis = 1)
          H_b = b_d + b_01 + b_10

          H_a = block_diag(H_a, jnp.array([1]))
          H_b = block_diag(H_b, jnp.array([1]))

          Theta = jnp.hstack((jnp.dstack((jnp.exp(1j * t), jnp.ones(N_modes//2))).reshape(N_modes - 1), jnp.array(1, dtype = 'complex64')))
          Phi = jnp.hstack((jnp.dstack((jnp.exp(1j * p), jnp.ones(N_modes//2))).reshape(N_modes - 1), jnp.array(1, dtype = 'complex64')))
        
        else:
          t = theta[idx : idx + N_modes//2]
          p = phi[idx : idx + N_modes//2]
          a = alpha[idx : idx + N_modes//2]
          b = beta[idx : idx + N_modes//2]
          idx = idx + N_modes//2

          a_d = jnp.diag(jnp.dstack((jnp.cos(jnp.pi/4 + a), jnp.cos(jnp.pi/4 + a))).reshape(N_modes - 1))
          a_01 = jnp.roll(jnp.diag(jnp.dstack((1j * jnp.sin(np.pi/4 + a), jnp.zeros(N_modes//2))).reshape(N_modes - 1)), 1, axis = 1)
          a_10 = jnp.roll(jnp.diag(jnp.dstack((jnp.zeros(N_modes//2), 1j * jnp.sin(jnp.pi/4 + a))).reshape(N_modes - 1)), -1, axis = 1)
          H_a = a_d + a_01 + a_10

          b_d = jnp.diag(jnp.dstack((jnp.cos(jnp.pi/4 + b), jnp.cos(jnp.pi/4 + b))).reshape(N_modes - 1))
          b_01 = jnp.roll(jnp.diag(jnp.dstack((1j * jnp.sin(jnp.pi/4 + b), np.zeros(N_modes//2))).reshape(N_modes - 1)), 1, axis = 1)
          b_10 = jnp.roll(jnp.diag(jnp.dstack((jnp.zeros(N_modes//2), 1j * jnp.sin(jnp.pi/4 + b))).reshape(N_modes - 1)), -1, axis = 1)
          H_b = b_d + b_01 + b_10

          H_a = block_diag(jnp.array([1]), H_a)
          H_b = block_diag(jnp.array([1]), H_b)

          Theta = jnp.hstack((jnp.array(1, dtype = 'complex64'), jnp.dstack((jnp.exp(1j * t), jnp.ones(N_modes//2))).reshape(N_modes - 1)))
          Phi = jnp.hstack((jnp.array(1, dtype = 'complex64'), jnp.dstack((jnp.exp(1j * p), jnp.ones(N_modes//2))).reshape(N_modes - 1)))

        M = H_b @ jnp.diag(Theta) @ H_a @ jnp.diag(Phi)
        col_matrices.append(M)
      return jnp.diag(jnp.exp(1j * D)) @ reduce(jnp.matmul, col_matrices[::-1])

    if self.N_modes %2 == 0:
      return even_clements(self.N_modes, theta, phi, D, alpha, beta)
    else:
      return odd_clements(self.N_modes, theta, phi, D, alpha, beta)

  def get_clements_phases(self, U, inverse = False):
    r"""
    Decompose mesh network into phases based on the Clements (rectangular) decomposition.

    :param U: 2D matrix to configure the Clements mesh
    :param inverse:
    :return: Returns MZI phases $\theta, \phi$ and phase screen $\D$
    """

    t = np.zeros(self.N_modes * (self.N_modes - 1)//2)
    p = np.zeros(self.N_modes * (self.N_modes - 1)//2)
    M = U
    T_inv_list = []
    T_list = []

    def get_two_mode_unitary(i, j, theta, phi, inverse = False):
      U_temp = np.eye(self.N_modes, dtype = 'complex128')
      m = min(i, j) - 1
      H = 1/np.sqrt(2) * np.array([[1, 1j], [1j, 1]])
      Theta = np.array([[np.exp(1j * theta), 0], [0, 1]])
      Phi = np.array([[np.exp(1j * phi), 0], [0, 1]])
      M = H @ Theta @ H @ Phi
      U_temp[m : m + 2, m : m + 2] = M
      if inverse:
        U_temp = np.conj(U_temp).T
      return U_temp

    def null_matrix_element(mode1, mode2, M_row, M_col, M, inverse = False):
        if inverse:
            if M[M_row - 1, M_col] == 0:
                thetar = 0; phir = 0
            elif M[M_row-1, M_col-1]==0:
                thetar = np.pi; phir = 0
            else:
                r = -M[M_row - 1, M_col] / M[M_row - 1, M_col - 1]
                thetar = 2*np.arctan(np.abs(r))
                phir = -np.angle(r)
        else:
            if M[M_row - 2, M_col - 1]==0:
                thetar = 0; phir = 0
            elif M[M_row - 1, M_col - 1]==0:
                thetar = np.pi; phir = 0
            else:
                r = M[M_row - 2, M_col - 1] / M[M_row - 1, M_col - 1]
                thetar = 2*np.arctan(np.abs(r))
                phir = -np.angle(r)

        U = get_two_mode_unitary(mode1, mode2, thetar, phir, inverse=inverse)
        if inverse: M = M @ U
        else:   M = U @ M

        return M, phir, thetar

    for i in range(1, self.N_modes):
      if i%2 == 1:
        for j in range(i):
          m = i - j
          U, phi, theta = null_matrix_element(m, m + 1, self.N_modes - j, i - j, U, inverse = True)
          T_list.append((m - 1, m, theta, phi))
      else:
        for j in range(1, i + 1):
          m = self.N_modes + j - i - 1
          U, phi, theta = null_matrix_element(m, m + 1, self.N_modes + j - i, j, U)
          T_inv_list.append((m - 1, m, theta, phi))

    D = np.angle(np.diag(U))

    T_inv_list.reverse()
    for T_inv_matrix in T_inv_list:
      m, n, theta, phi = T_inv_matrix
      phi_temp = D[m] - D[n]
      D[m] = D[n] - phi - np.pi - theta
      D[n] = D[n] + np.pi - theta
      T_list.append((m, n, theta, phi_temp))

    idx = {k:0 for k in range(self.N_modes - 1)}

    theta_array = np.zeros((self.N_modes//2, self.N_modes))
    phi_array = np.zeros((self.N_modes//2, self.N_modes))
    for MZ in T_list:
      i, j, theta, phi = MZ
      col = 2 * idx[i] + i%2
      idx[i] = idx[i] + 1
      row = int(i/2)
      theta_array[row, col] = theta
      phi_array[row, col] = phi

    idx = 0
    for i in range(self.N_modes):
      for j in range(self.N_modes//2 - i%2):
        t[idx] = theta_array[j, i]
        p[idx] = phi_array[j, i]
        idx = idx + 1
    D = D % (2 * np.pi)

    return t, p, D

  def set_clements(self, U, alpha, beta) -> jnp.ndarray:
    r"""
    Sets mesh to a given matrix, with beam-splitter errors that can be added.

    :param U: 2D unitary clements matrix
    :param alpha: 1D array of directional coupler errors
    :param beta: 1D array of directional coupler errors
    :return: Returns NxN matrix transform
    """

    theta, phi, D = self.get_clements_phases(U)
    U_out = self.clements_matrix(theta, phi, D, alpha, beta)
    return U_out

  def local_EC(self, theta, phi, D, alpha, beta):
    raise NotImplementedError()

  @partial(jit, static_argnums = (0, ))
  def calc_perm(self, U):
    r"""
    Calculates the permanent of the square matrix U using the Ryser formula

    :param U: NxN square matrix to calculate the permanent of
    :return: permanent of U
    """
    n_dim_x, n_dim_y = jnp.shape(U)
    sign = jnp.tile(jnp.array([-1, 1]), 2 ** (n_dim_x - 1))

    def generate_grey(idx):
      bin_idx = (idx + 1) % (2 ** n_dim_x)
      new_grey = bin_idx ^ (bin_idx // 2)
      return new_grey

    new_grey = jax.vmap(lambda idx: generate_grey(idx))(jnp.arange(0, 2 ** n_dim_x))
    new_grey_rolled = jnp.roll(new_grey, 1)
    direction = jnp.where(new_grey < new_grey_rolled, +1.0, -1.0)

    grey_diff = np.array([2 ** np.binary_repr(i + 1)[::-1].index('1') for i in range(2 ** (n_dim_x - 1))] * 2)
    grey_diff_index = jnp.array(np.log2(grey_diff), dtype=jnp.int16)
    new_vector = U[grey_diff_index]

    reduced = jnp.prod(jnp.cumsum(jnp.einsum('ij, i -> ij', new_vector, direction), axis=0), axis=1)
    perm = jnp.sum(reduced * sign)

    return perm