#Jasvith Raj Basani
#
#
#History
# 29/1/2022 - Created this File
# 15/2/2023 - Working linear layer

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from .linear_optics import Linear_Optics
from .scatterer import TLE
import itertools

class Circuit:

    def __init__(self,
                 N_modes: jnp.int16 = None,
                 input_photons: tuple = None,
                 spectral_profile = None,
                 spectral_param = None,
                 S_matrix: jnp.ndarray = None,
                 k: jnp.ndarray = jnp.linspace(-6, 6, 70)
                 ):

        r"""
        Class to construct the circuit, using the mode update method.
        Presently supports a maximum of only 2 photons in the circuit

        :param N_modes: Number of spacial/waveguide modes in the circuit (i.e. how wide the circuit is)
        :param input_photons: Tuple with the number of photons being input into each waveguide
        :param spectral_profile: Function indicating the frequency domain distribution of the photons
                                 Presently, this code requires all input photons to have the same spectral profile
        :param spectral_param: Tuple of arguments that characterize spectral_profile
        :param S_matrix: Scattering matrix for the TLEs that will be used in the nonlinear layer
                         If None, it will be calculated everytime a circuit is initialized. Ideally, this is called only once per unique emitter in the circuit
        :param k: Frequency distribution of the photons, default jnp.arange(-6, 6, 70) in units of gamma
        """
        print ("Initializing Circuit, Please Wait . . . ")
        self.N_modes = N_modes
        self.input_photons = input_photons
        self.spectral_profile = spectral_profile
        self.k = k
        self.steps = len(self.k)
        self.dk = self.k[2] - self.k[1]
        if S_matrix is None:
            self.S_matrix = TLE.S_mat_tt(self.k, self.k, self.k, self.k)
        else:
            self.S_matrix = S_matrix

        self.state_defs = []
        self.mode_defs = []
        self.ones, self.twos, self.counts = [], [], []
        self.state_defs, self.mode_defs = [], []
        self.k_1, self.k_2 = jnp.meshgrid(self.k, self.k)

        if np.sum(self.input_photons) == 1:
            raise NotImplementedError()

        elif np.sum(self.input_photons) == 2:
            ones_temp = tuple(np.append(np.array([1,1]), np.zeros(len(self.input_photons) - 2)))
            twos_temp = tuple(np.append(np.array([2]), np.zeros(len(self.input_photons) - 1)))
            #Check for duplicate values in the permutations
            for state in list(itertools.permutations(ones_temp)):
                if state not in self.ones:
                    self.ones.append(state)
                else:
                    pass
            for state in (itertools.permutations(twos_temp)):
                if state not in self.twos:
                    self.twos.append(state)
                else:
                    pass

            for count, state in enumerate(self.twos):
                self.state_defs.append(state)
                if state == self.input_photons:
                    self.mode_defs.append(self.make_2D(self.spectral_profile, spectral_param))
                    self.counts.append(count)
                else:
                    self.mode_defs.append(jnp.zeros(self.k_1.shape))

            for count, state in enumerate(self.ones):
                self.state_defs.append(state)
                if state == self.input_photons:
                    self.mode_defs.append(self.make_2D(self.spectral_profile, spectral_param))
                    self.counts.append(count)
                else:
                    self.mode_defs.append(jnp.zeros(self.k_1.shape))

        self.lo = Linear_Optics(self.N_modes)
        self.tle = TLE()

        self.mode_defs = jnp.array(self.mode_defs)
        self.state_defs = jnp.array(self.state_defs)
        self.counts = jnp.array(self.counts)
        self.bogo_coeff = jnp.zeros((len(self.mode_defs), len(self.mode_defs))) + 0j
        self.single_transfer = self.tle.S_mat_t(self.k_1, self.k_1) * self.tle.S_mat_t(self.k_2, self.k_2)
        self.norm_coeff = jnp.array(np.append([2**-0.5] * len(self.twos), [1] * len(self.ones)))
        self.vmap_twos = jnp.arange(0, len(self.twos))
        self.vmap_ones = jnp.arange(0, len(self.ones))


        print ("***Circuit Compiled***")

    @partial(jit, static_argnums = (0, 1, ))
    def make_2D(self, spectral_profile, sigma):
        r"""
        Given a spectral profile for 2 input photons, returns the two-photon wavefunction as a function of k_1, k_2

        :param spectral_profile: Callable function for the spectral profile (typically Gaussian, Lorentzian atc)
        :param sigma: Bandwidth/characterizing width of the spectral_profile
        :return: 2-dimensional function of k_1 and k_2
        """
        spectrum = spectral_profile(self.k, sigma).reshape((self.steps, 1))
        return spectrum.dot(spectrum.T) + 0j

    # TODO: Do I even need this function?
    @partial(jit, static_argnums = (0, ))
    def init_modes(self, sigma):
        r"""
        Function call occurs during __init__(), to initialize the mode definitions when the circuit is first called

        :param sigma:
        :return:
        """
        input_modes = jnp.zeros(self.mode_defs.shape) + 0j
        input_modes = (lambda i: input_modes.at[i].set(self.make_2D(self.spectral_profile, sigma)))(self.counts)
        # TODO: if input_photons has multiple state, do normalization
        return input_modes + 0j

    @partial(jit, static_argnums = (0, ))
    def add_linear_layer(self, modes, theta, phi, D, alpha, beta):
        r"""
        User function call to add a linear optical mesh transformation to the circuit

        :param modes: Frequency domain mode definitions for each possible photon state in the system
        :param theta: Optimizable phase shift values of the mesh, first phase of all the MZIs
        :param phi: Optimizable phase shift values of the mesh, second phase of all the MZIs
        :param D: Optimizable phase shift values of the mesh, output phase screen
        :param alpha: Non-optimizable directional coupler error
        :param beta: Non-optimizable directional coupler error
        :return: output mode definitions for all the two-photon states in the system
        """

        weights = self.lo.clements_matrix(theta, phi, D, alpha, beta)
        out_modes = self.bogoluigov_coeffs(modes, weights)
        return out_modes

    @partial(jit, static_argnums=(0,))
    def bogoluigov_coeffs(self, modes, weights):
        r"""

        :param modes:
        :param weights:
        :return:
        """
        bogo_coeff_0 = jnp.zeros((len(modes), len(modes))) + 0j

        # for i in range(self.N_modes):
        #  bogo_transform = jnp.einsum('i, j -> ij', weights[i], weights[i])
        #  bogo_transform = bogo_transform + jnp.transpose(bogo_transform)
        #  bogo_coeff = bogo_coeff.at[i].set(jnp.append(2**0.5 * 0.5 * jnp.diag(bogo_transform), bogo_transform[jnp.triu_indices_from(bogo_transform, k=1)]))
        #   print (bogo_coeff)

        def two_photon_transform(idx, bogo_coeff, i, j, weights):
            bogo_transform = jnp.einsum('i, j -> ij', weights[i], weights[j])
            bogo_transform = bogo_transform + jnp.transpose(bogo_transform)
            bogo_coeff = bogo_coeff.at[idx].set(jnp.append(2 ** 0.5 * 0.5 * jnp.diag(bogo_transform), bogo_transform[jnp.triu_indices_from(bogo_transform, k=1)]))
            return bogo_coeff

        bogo_coeff_2 = jax.vmap(lambda idx: two_photon_transform(idx, bogo_coeff_0, idx, idx, weights))(
            self.vmap_twos)
        bogo_coeff = jnp.sum(bogo_coeff_2, axis=0)

        # for i in range(N_modes, len(modes)):
        #  for j in range(i - N_modes + 1):
        #    bogo_transform = jnp.einsum('i, j -> ij', weights[i], weights[j])
        #    bogo_transform = bogo_transform + jnp.transpose(bogo_transform)
        #    bogo_coeff = bogo_coeff.at[i].set(jnp.append(2**0.5 * 0.5 * jnp.diag(bogo_transform), bogo_transform[jnp.triu_indices_from(bogo_transform, k=1)]))

        idx_count = len(self.vmap_twos)
        for i in range(0, len(weights)):
            for j in range(i, len(weights)):
                if (i == j):
                    pass
                else:
                    bogo_coeff = two_photon_transform(idx_count, bogo_coeff, i, j, weights)
                    idx_count += 1

        '''Normalization'''
        modes = jnp.einsum('ijk, i -> ijk', modes, self.norm_coeff)
        '''Gotta love screwing with indexing'''
        modes = jnp.einsum('ijk, li -> ljk', modes, jnp.transpose(bogo_coeff))
        # return bogo_coeff
        return modes

    @partial(jit, static_argnums = (0, ))
    def add_TLE_layer(self, modes, emitter_bools):
        r"""

        :param modes:
        :param emitter_bools:
        :return:
        """
        modes_temp = jnp.zeros_like(modes) + 0j
        modes = modes + 0j
        assert len(emitter_bools) == self.N_modes, "The number of emitters does not equal the number of spatial modes"
    
        '''Field programmable two-photon transform'''
        def apply_two_photon_NL(idx, modes):
            modes = modes.at[idx].set(self.tle.psi_out_tt(modes[idx], self.S_matrix))
            return modes
        def pass_two_photon_NL(idx, modes):
            return modes
        def two_photon_NL(idx, emitter_bool, modes):
            modes = jax.lax.cond(emitter_bool == 1, apply_two_photon_NL, pass_two_photon_NL, idx, modes)
            return modes
    
        modes_temp = jax.vmap(lambda idx, emitter_bool: two_photon_NL(idx, emitter_bool, modes))(self.vmap_twos, emitter_bools)
        #Convert this into a jax.lax.scan loop
        for idx in range(len(self.vmap_twos)):
            modes = modes.at[idx].set(modes_temp[idx][idx])

        '''Field Programmable one-photon transform'''
        def apply_one_photon_NL_k1(idx, modes, transform):
            modes = modes.at[idx].set(modes[idx] * transform)
            return modes
        def apply_one_photon_NL_k2(idx, modes, transform):
            modes = jnp.transpose(modes, [0, 2, 1])
            modes = modes.at[idx].set(modes[idx] * transform)
            return jnp.transpose(modes, [0, 2, 1])
        def pass_one_photon_NL(idx, modes, transform):
            return modes
        def one_photon_NL(idx, emitter_bools, modes):
            indices = jnp.argwhere(jnp.array(self.ones[idx]) == 1, size = 2)
            bool_1, bool_2 = emitter_bools[indices[0][0]], emitter_bools[indices[1][0]]
            bool_3 = bool_1 * bool_2
            idx = idx + len(self.twos)

            modes = jax.lax.cond(bool_3 == 1, apply_one_photon_NL_k1, pass_one_photon_NL, idx, modes, self.single_transfer)
            modes = jax.lax.cond(((bool_1 == 1) & (bool_2 == 0)), apply_one_photon_NL_k1, pass_one_photon_NL, idx, modes, self.tle.S_mat_t(self.k))
            modes = jax.lax.cond(((bool_1 == 0) & (bool_2 == 1)), apply_one_photon_NL_k2, pass_one_photon_NL, idx, modes, self.tle.S_mat_t(self.k))
            modes = jax.lax.cond(((bool_1 == 0) & (bool_2 == 0)), pass_one_photon_NL, pass_one_photon_NL, idx, modes, 1)
            return modes

        for idx in range(len(self.vmap_ones)):
            modes = one_photon_NL(idx, emitter_bools, modes)
    
        return modes
