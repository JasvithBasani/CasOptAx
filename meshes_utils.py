# History
# NOTE: This file was created to add support based on the MESHES (https://github.com/QPG-MIT/meshes.git) package to CasOptAx.
# 22/10/2024 - Created this File
# 22/10/2024 - Added Pruned SCF Mesh/Green Machine Support

import numpy as np
from numpy.linalg import svd
from scipy.linalg import cossin

import jax
import jax.numpy as jnp 
from jax import jit

from functools import partial
import itertools
from scipy.stats import unitary_group
import warnings

# !git clone https://github.com/QPG-MIT/meshes.git
warnings.warn("Please make sure the meshes package is imported. To import the meshes package, git clone https://github.com/QPG-MIT/meshes.git")
import meshes as ms
from meshes.mesh import StructuredMeshNetwork
from meshes.crossing import Crossing, MZICrossing

class Prunable_SCF(StructuredMeshNetwork):

  def __init__(self, 
               N: int = None,
               p_phase: np.ndarray = 0.0,
               p_splitter: np.ndarray = 0.0,
               p_crossing: np.ndarray = None,
               phi_out: np.ndarray = None,
               M: np.ndarray = None,
               stride_vals: np.ndarray = None,
               X: Crossing = MZICrossing(),
               order: bool = True
               ):
    r"""
    Compiling unitary based on the pruned Sine-Cosine Fractal decomposition. Alternatively implemented by the generalized Green Machine.
    For more information about the SCF Mesh: 10.1515/nanoph-2022-0525
    For more information about the Green Machine: arXiv.2310.05889

    :param N: Mesh size. Not needed if matrix M specified.
    :param p_phase: Phase shifts, dim = (N(N - 1)/2 * X.n_phase + N)
    :param p_splitter: Splitter imperfections, dim = (N(N - 1)/2, X.n_splitter)
    :param p_crossing: Crossing parameters, dim = (N(N - 1)/2, X.n_phase)
    :param phi_out: Output phases, dim=(N)
    :param M: Target matrix.
    :param X: Crossing type.
    :param order: Boolean value to permute singular values in case of instability in decomposition.
    """
    
    #Ensure that mesh size is a power of 2
    assert N == 2**int(np.log2(N))
    if stride_vals is not None:
      assert np.max(stride_vals) <= N//2
      
    if stride_vals is None:
      stride_vals = self.make_stride_array(N, alpha = 2.0)
    else:
      stride_vals = np.array(stride_vals)

    lens = [N//2] * len(stride_vals)
    shifts = [0] * len(stride_vals)

    def jump_layers(old_order, new_order):
      order = {key: i for i, key in enumerate(new_order)}
      output = sorted(old_order, key = lambda d: order[old_order[d]])
      return output

    def generate_perms(N, stride_vals):
      perm = [None] * len(stride_vals + 1)
      perm_vals = [np.arange(N)]
      for idx, s in enumerate(stride_vals):
        old_order = perm_vals[0]
        perm_vals = [(np.outer(1, x) + np.outer(np.arange(0, N, 2*s), 1)).flatten() for x in [np.outer(np.arange(s),1)+np.array([[0,s]]), np.outer(1,np.arange(0,2*s,2))+np.array([[0],[1]])]]
        perm[idx:idx + 2] = perm_vals
        new_order = perm_vals[0]
        perm[idx] = np.array(jump_layers(old_order, new_order))
      return perm

    permutations = np.array(generate_perms(N, stride_vals))

    super(Prunable_SCF, self).__init__(N, lens, shifts, p_phase = p_phase, p_splitter = p_splitter, p_crossing = p_crossing, phi_out = phi_out, perm = permutations, X = X, phi_pos = 'out')

    def config_truncated(U, D_ij):
      r"""
      Perform recursive SVD on U

      :param U: Unitary matrix to decompose
      :param D_ij: Array of crossing amplitudes
      """
      N = len(U)
      if (N > 2):
        (U11, U12, U21, U22) = (U[:N//2, :N//2], U[:N//2, N//2:], U[N//2:, :N//2], U[N//2:, N//2:])
        (V, D, W) = cossin([U11, U12, U21, U22])
        if order:
          p = np.arange(N//2)
        else:
          np.random.seed(0)
          p = np.argsort(np.random.randn(N//2))
        V1 = V[:N//2, p]; V2 = V[N//2:, p + N//2]
        W1 = W[p, :N//2]; W2 = W[p + N//2, N//2:]
        D11 = D[p, p]; D12 = D[p, p + N//2]
        D21 = D[p + N//2, p]; D22 = D[p + N//2, p + N//2]
        D_ij[0, 0, N//2 - 1, :] = D11
        D_ij[0, 1, N//2 - 1, :] = D12
        D_ij[1, 0, N//2 - 1, :] = D21
        D_ij[1, 1, N//2 - 1, :] = D22
        '''The lines below can result in numerical instability in decomposing very sparse matrices. Noticed in particular for the DFT matrix.'''
        # (V1, D11, W1) = np.linalg.svd(U11)
        # (V2, D22, W2) = np.linalg.svd(U22)
        # D_ij[0, 0, N//2 - 1, :] = D11
        # D_ij[0, 1, N//2 - 1, :] = np.diag(V1.T.conj().dot(U12).dot(W2.T.conj()))
        # D_ij[1, 1, N//2 - 1, :] = D22
        # D_ij[1, 0, N//2 - 1, :] = np.diag(V2.T.conj().dot(U21).dot(W1.T.conj()))

        config_truncated(W1, D_ij[:, :, :N//2, :N//4])
        config_truncated(W2, D_ij[:, :, :N//2, N//4:])
        config_truncated(V1, D_ij[:, :, N//2:, :N//4])
        config_truncated(V2, D_ij[:, :, N//2:, N//4:])

      else:
        D_ij[:, :, 0, 0] = U

    D_ij = np.zeros([2, 2, N - 1, N//2]) + 0j
    config_truncated(M, D_ij)

    # Convert the crossing amplitudes D_ij into phase shifts
    p_crossing = self.p_crossing.reshape([len(stride_vals), N//2, 2])
    phi_out = self.phi_out
    p_0 = [np.arange(N)]
    for idx in range(len(stride_vals)):
      s = stride_vals[idx]
      old_order = p_0[0]
      (p_1, p_2) = [(np.outer(1, x) + np.outer(np.arange(0, N, 2 * s), 1)).flatten() for x in [np.outer(np.arange(s), 1) + np.array([[0, s]]), np.outer(1, np.arange(0, 2 * s, 2)) + np.array([[0], [1]])]]
      p_1 = jump_layers(old_order, p_1)
      phi_out[:] = phi_out[p_1]
      D_ij[:, 0, idx, :] *= np.exp(1j * phi_out[::2])
      D_ij[:, 1, idx, :] *= np.exp(1j * phi_out[1::2])
      p_crossing[idx] = np.array(self.X.Tsolve((D_ij[0, 0, idx], D_ij[0, 1, idx]), 'T1:')[:1])[0].T
      # phi_out[:] = np.angle(D_ij[:, :, idx]/self.X.T(p_crossing[idx]))[:, 0, :].T.flatten()[p_2]
      phi_out[:] = (np.angle(D_ij[:, :, idx]) - np.angle(self.X.T(p_crossing[idx])))[:, 0, :].T.flatten()[p_2]

  def sparsity(self, N, U):
    r"""
    Calculate the sparsity of the matrix

    :param N: Size of the matrix
    :param U: Matrix
    """
    return (N**2 - np.count_nonzero(np.round(U, 5)))/N**2

  def make_stride_array(self, N, alpha):
    r"""
    Calculates stride array from left to right for a given pruning parameter alpha

    :param N: Size of the matrix
    :param alpha: Pruning parameter
    """
    const = (N - 1)/N
    possible_strides = [2**i for i in range(0, int(np.log2(N)))]
    stride_numbers = [int(np.floor((N/2**i)**(alpha - 1) * 1/const)) for i in range(1, int(np.log2(N) + 1), 1)]
    stride_array = []
    for idx in range(N - 1):
      s = 2**np.binary_repr(idx + 1)[::-1].index('1')
      if stride_numbers[possible_strides.index(s)] > 0:
        stride_array.append(s)
        stride_numbers[possible_strides.index(s)] -=1
      else:
        pass
    return stride_array

    
