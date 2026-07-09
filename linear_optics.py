
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import block_diag
import scipy.stats
from functools import partial, reduce
from typing import Optional


class Linear_Optics:
    """Provides differentiable utilities for simulating linear optical networks.

    This class handles the creation and simulation of programmable photonic
    integrated circuits, including individual Mach-Zehnder Interferometers (MZIs)
    and full rectangular mesh decompositions.
    """

    def __init__(self, N_modes: Optional[int] = None):
        """Initializes the linear optics simulator.

        Args:
            N_modes: Number of spatial modes input to the network.
        """
        self.N_modes = N_modes

    def haar_mat(self, N_modes: int) -> jnp.ndarray:
        """Generates a Haar-random unitary matrix.

        Args:
            N_modes: Number of spatial modes for the unitary group.

        Returns:
            A complex JAX array representing a Haar-random unitary matrix.
        """
        # Generate the random unitary sample using SciPy, then cast to JAX array
        raw_matrix = scipy.stats.unitary_group.rvs(N_modes)
        return jnp.array(raw_matrix, dtype=jnp.complex64)

    @partial(jax.jit, static_argnums=(0,))
    def mzi(
        self,
        theta: jnp.ndarray,
        phi: jnp.ndarray,
        alpha: jnp.ndarray = 0.0 + 0.0j,
        beta: jnp.ndarray = 0.0 + 0.0j,
    ) -> jnp.ndarray:
        """Computes a single MZI transfer function for a 2x2 unitary matrix.

        Evaluates the matrix mathematical transformations corresponding to Eq. (5)
        in https://doi.org/10.1364/OPTICA.424052.

        Args:
            theta: Internal MZI phase shift parameter $\theta$.
            phi: External MZI phase shift parameter $\phi$.
            alpha: Directional coupler error $\alpha$ for the first coupler.
            beta: Directional coupler error $\beta$ for the second coupler.

        Returns:
            A 2x2 complex unitary matrix representing the component transformation.

        Raises:
            ValueError: If the configured number of modes in the instance is
                not equal to 2.
        """
        if self.N_modes != 2:
            raise ValueError(
                f"MZI evaluation requires exactly 2 modes. Current configuration "
                f"specifies {self.N_modes} modes."
            )

        t_00 = jnp.exp(1j * phi) * (
            jnp.cos(alpha - beta) * jnp.sin(theta / 2)
            + 1j * jnp.sin(alpha + beta) * jnp.cos(theta / 2)
        )
        t_01 = jnp.cos(alpha + beta) * jnp.cos(theta / 2) + 1j * jnp.sin(
            alpha - beta
        ) * jnp.sin(theta / 2)
        t_10 = jnp.exp(1j * phi) * (
            jnp.cos(alpha + beta) * jnp.cos(theta / 2)
            - 1j * jnp.sin(alpha - beta) * jnp.sin(theta / 2)
        )
        t_11 = -(
            jnp.cos(alpha - beta) * jnp.sin(theta / 2)
            - 1j * jnp.sin(alpha + beta) * jnp.cos(theta / 2)
        )

        transfer_matrix = jnp.array([[t_00, t_01], [t_10, t_11]])
        return 1j * jnp.exp(1j * theta / 2) * transfer_matrix

    def get_mzi_phases(self, unitary: jnp.ndarray):
        """Extracts individual MZI phases from a target unitary matrix.

        Args:
            unitary: A complex unitary matrix target for mesh decomposition.

        Raises:
            NotImplementedError: This method has not been implemented yet.
        """
        raise NotImplementedError("Phase extraction routine is not implemented.")

    def set_mzi(self, unitary: jnp.ndarray, alpha: jnp.ndarray, beta: jnp.ndarray):
        """Sets MZI parameters based on a target unitary matrix and error profiles.

        Args:
            unitary: Target unitary matrix configuration.
            alpha: Array of directional coupler errors for the first couplers.
            beta: Array of directional coupler errors for the second couplers.

        Raises:
            NotImplementedError: This method has not been implemented yet.
        """
        raise NotImplementedError("MZI phase setter routine is not implemented.")

    def scf_matrix(
        self,
        theta: jnp.ndarray,
        phi: jnp.ndarray,
        d: jnp.ndarray,
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
    ):
        """Computes the unitary transfer matrix of an SCF mesh layout.

        Args:
            theta: A 1D array of internal phase shift values for the mesh.
            phi: A 1D array of external phase shift values for the mesh.
            d: A 1D array representing the output phase screens.
            alpha: A 1D array of directional coupler errors (first coupler).
            beta: A 1D array of directional coupler errors (second coupler).

        Raises:
            NotImplementedError: This method has not been implemented yet.
        """
        raise NotImplementedError("SCF matrix assembly is not implemented.")

    @partial(jax.jit, static_argnums=(0,))
    def clements_matrix(
        self,
        theta: jnp.ndarray,
        phi: jnp.ndarray,
        d: jnp.ndarray,
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
    ) -> jnp.ndarray:
        """Computes the unitary transfer matrix of a Clements mesh network.

        Simulates a differentiable mesh network based on the rectangular Clements
        decomposition. Mach-Zehnder Interferometers (MZIs) are indexed from top to
        bottom and left to right.

        Args:
            theta: A 1D array of internal phase shift values for the MZIs.
            phi: A 1D array of external phase shift values for the MZIs.
            d: A 1D array representing the output phase screen.
            alpha: A 1D array of directional coupler errors (first coupler).
            beta: A 1D array of directional coupler errors (second coupler).

        Returns:
            A complex unitary matrix representing the full mesh transformation.

        Raises:
            ValueError: If the input arrays do not match the expected dimensions
                based on the configured number of modes.
        """
        N_modes = self.N_modes
        expected_mzis = (N_modes * (N_modes - 1)) // 2

        if theta.shape[0] != expected_mzis:
            raise ValueError(f"Expected theta length {expected_mzis}, got {theta.shape[0]}")
        if phi.shape[0] != expected_mzis:
            raise ValueError(f"Expected phi length {expected_mzis}, got {phi.shape[0]}")
        if alpha.shape[0] != expected_mzis:
            raise ValueError(f"Expected alpha length {expected_mzis}, got {alpha.shape[0]}")
        if beta.shape[0] != expected_mzis:
            raise ValueError(f"Expected beta length {expected_mzis}, got {beta.shape[0]}")
        if d.shape[0] != N_modes:
            raise ValueError(f"Expected d length {N_modes}, got {d.shape[0]}")

        unitary = jnp.eye(N_modes, dtype=jnp.complex64)
        idx = 0

        for i in range(N_modes):
            is_even_layer = (i % 2 == 0)
            if N_modes % 2 == 0:
                k = N_modes // 2 if is_even_layer else (N_modes // 2) - 1
                offset = 0 if is_even_layer else 1
            else:
                k = N_modes // 2
                offset = 0 if is_even_layer else 1

            t = theta[idx : idx + k]
            p = phi[idx : idx + k]
            a = alpha[idx : idx + k]
            b = beta[idx : idx + k]
            idx += k

            c_a = jnp.cos(jnp.pi / 4 + a)
            s_a = jnp.sin(jnp.pi / 4 + a)
            c_b = jnp.cos(jnp.pi / 4 + b)
            s_b = jnp.sin(jnp.pi / 4 + b)

            exp_t = jnp.exp(1j * t)
            exp_p = jnp.exp(1j * p)
            zero = jnp.zeros_like(c_a)
            one = jnp.ones_like(c_a)

            h_a = jnp.stack([
                jnp.stack([c_a, 1j * s_a], axis=-1),
                jnp.stack([1j * s_a, c_a], axis=-1)
            ], axis=1)

            h_b = jnp.stack([
                jnp.stack([c_b, 1j * s_b], axis=-1),
                jnp.stack([1j * s_b, c_b], axis=-1)
            ], axis=1)

            phi_mat = jnp.stack([
                jnp.stack([exp_p, zero], axis=-1),
                jnp.stack([zero, one], axis=-1)
            ], axis=1)

            theta_mat = jnp.stack([
                jnp.stack([exp_t, zero], axis=-1),
                jnp.stack([zero, one], axis=-1)
            ], axis=1)

            mzi_blocks = h_b @ theta_mat @ h_a @ phi_mat
            
            # Extract, update, and replace active rows
            unitary_active = unitary[offset : offset + 2 * k, :].reshape(k, 2, N_modes)
            unitary_active = jnp.einsum('kab,kbj->kaj', mzi_blocks, unitary_active)
            unitary = unitary.at[offset : offset + 2 * k, :].set(unitary_active.reshape(2 * k, N_modes))

        return jnp.exp(1j * d)[:, None] * unitary

    @partial(jit, static_argnums = (0, ))
    def clements_matrix_deprecated(self, theta, phi, D, alpha, beta) -> jnp.ndarray:
      r"""
      DEPRECATED: Differentiable mesh network based on the Clements (rectangular) decomposition,
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