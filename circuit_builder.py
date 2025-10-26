# History
# 29/1/2022 - Created this File
# 15/2/2023 - Working linear layer
# 30/3/2023 - Working field programmability?
# 20/9/2023 - Single mode circuit class, with Boson Sampling and 3ls nonlinearity

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from .linear_optics import Linear_Optics
# from .scatterer import TLE
import itertools
from scipy.special import factorial
import matplotlib.pyplot as plt

class Circuit_singlemode:

    def __init__(self,
                 N_modes: jnp.int16 = None,
                 N_photons: jnp.int16 = None,
                 input_photons: jnp.ndarray = None
                 ):
        r"""
        Class to construct a single mode circuit

        :param N_modes: Number of spatial/waveguide modes in the circuit (i.e. how wide the circuit is)
        :param N_photons: Number of photons in the circuit (lossless, so photon number is conserved)
        :param input_photons: Tuple of length N_modes, with sum N_photons to indicate which initial state is populated
        """
        """TODO: Superpositions of states as input is not yet supported"""

        print("Initialzing Circuit, Please Wait .  .  .")
        self.N_modes = N_modes
        self.N_photons = N_photons
        self.input_photons = input_photons

        assert jnp.sum(jnp.array(self.input_photons)) == self.N_photons
        assert len(self.input_photons) == self.N_modes

        def generate_possible_states(N_modes, N_photons):
            r"""
            __init__ call to generate all possible states of N_photons in N_modes
            :param N_modes:
            :param N_photons:
            :return:
            """

            def generate_state(N_modes, N_photons, current_state):
                if N_modes == 0:
                    if N_photons == 0:
                        valid_state.append(current_state[:])
                    return
                for num in range(0, N_photons + 1):
                    if num <= N_photons:
                        current_state.append(num)
                        generate_state(N_modes - 1, N_photons - num, current_state)
                        current_state.pop()

            valid_state = []
            generate_state(N_modes, N_photons, [])
            return valid_state
        
        def generate_possible_states_backtrack(N_modes, N_photons):
            r"""
            __init__ call to generate all possible states of N_photons in N_modes
            Backtracking algorithm to generate lists of length N_modes that sums up to N_photons
            :param N_modes:
            :param N_photons:
            :return:
            """
            
            result_list, result_dict = [], {}
            def backtrack(remaining_sum, index):
                if index == N_modes:
                    if remaining_sum == 0:
                        result_list.append(current_state[:])
                        result_dict[tuple(current_state)] = 0 + 0j
                    return

                for i in range(remaining_sum + 1):
                    current_state[index] = i
                    backtrack(remaining_sum - i, index + 1)

            current_state = [0] * N_modes
            backtrack(N_photons, 0)
            return result_list, result_dict

        # self.possible_states_list = generate_possible_states(self.N_modes, self.N_photons)
        self.possible_states_list, self.state_amps = generate_possible_states_backtrack(self.N_modes, self.N_photons)
        self.state_amps[tuple(self.input_photons)] = 1 + 0j
        
        self.num_possible_states = len(self.possible_states_list)
        self.arange_possible_states = jnp.arange(0, self.num_possible_states)
        
        # self.possible_states_dict = {}
        # for s in self.possible_states_list:
        #     self.possible_states_dict[tuple(s)] = jnp.array(s)
        self.possible_states_dict = self.state_amps.copy()
        self.possible_states_dict.update(itertools.starmap(lambda k, v: (k, jnp.array(v)), zip(self.possible_states_dict.keys(), self.possible_states_list)))

        # self.state_amps = {}
        # for s in self.possible_states_list:
        #     if (tuple(s) == self.input_photons):
        #         self.state_amps[tuple(s)] = 1 + 0j
        #     else:
        #         self.state_amps[tuple(s)] = 0 + 0j

        self.states_idx = {}
        for s in self.possible_states_list:
            # Use list comprehension to precompute idx_array
            idx_array = jnp.concatenate([jnp.full(num, count) for count, num in enumerate(s)])
            # Convert to JAX array with the desired dtype
            self.states_idx[tuple(s)] = jnp.array(idx_array, dtype = jnp.int16)
        
        # for s in self.possible_states_list:
        #     idx_array, count = np.array([]), 0
        #     for num in s:
        #         idx = ([count] * num)
        #         if idx != []:
        #             idx_array = np.append(idx_array, idx)
        #         count = count + 1
        #     self.states_idx[tuple(s)] = jnp.array(idx_array, dtype=jnp.int16)

        # self.all_states_factorial = {}
        # self.all_states_idx = {}
        # for s1 in self.possible_states_list:
        #     states_factorial = []
        #     idx_vals = []
        #     for s2 in self.possible_states_list:
        #         states_factorial.append(
        #             jnp.prod(factorial(np.array(s1)), axis=0) * jnp.prod(factorial(np.array(s2)), axis=0))
        #         idx_vals.append([self.states_idx[tuple(s1)], self.states_idx[tuple(s2)]])
        #     self.all_states_factorial[tuple(s1)] = jnp.array(states_factorial)
        #     self.all_states_idx[tuple(s1)] = jnp.array(idx_vals)
        
        factorial_products = jnp.prod(factorial(self.possible_states_list), axis=1)
        all_states_factorial_matrix = jnp.outer(factorial_products, factorial_products)

        idx_list = [self.states_idx[tuple(state)] for state in self.possible_states_list]
        all_states_idx = jnp.array([[jnp.stack([idx1, idx2], axis = 0) for idx2 in idx_list] for idx1 in idx_list])

        # Construct all pairwise combinations of indices
        self.all_states_factorial = {tuple(state): all_states_factorial_matrix[idx] for idx, state in enumerate(self.possible_states_list)}
        self.all_states_idx = {tuple(state): all_states_idx[idx] for idx, state in enumerate(self.possible_states_list)}

        self.lo = Linear_Optics(self.N_modes)
        #self.tle = TLE()

        print("***Circuit Ready For Compilation***")

    @partial(jit, static_argnums=(0,))
    def add_linear_layer(self, state_amps, theta, phi, D, alpha, beta):
        r"""
        User function call to add a linear optical mesh transformation into the circuit

        :param state_amps: Probability amplitudes of the states being input into the mesh. Has to maintain pytree structure of self.possible_states_dict
        :param theta: Optimizable phase shift values of the mesh, first phase of all the MZIs
        :param phi: Optimizable phase shift values of the mesh, second phase of all the MZIs
        :param D: Optimizable phase shift values of the mesh, output phase screen
        :param alpha: Non-optimizable directional coupler error
        :param beta: Non-optimizable directional coupler error
        :return: Output probability amplitudes of all the states. Again, maintains pytree structure
        """

        weights = self.lo.clements_matrix(theta, phi, D, alpha, beta)
        new_amps = jax.tree_util.tree_map(lambda amp, states_array, states_factorial: self.bosonic_transform(amp, weights, states_array, states_factorial), state_amps, self.all_states_idx, self.all_states_factorial)
        new_amps_extract, pytree_struct = jax.tree_util.tree_flatten(new_amps)
        new_amps = jax.tree_util.tree_unflatten(pytree_struct, jnp.sum(jnp.array(new_amps_extract), axis=0))
        return new_amps
    
    # @partial(jit, static_argnums=(0,))
    # def linear_evolution(self, state_amps, U):
    #     r"""
    #     User function call to evolve a state under a pre-defined unitary

    #     :param state_amps: Probability amplitudes of the states being input into the mesh. Has to maintain pytree structure of self.possible_states_dict
    #     :param U: Unitary of size (N_modes x N_modes) that defines the evolution
    #     :return: Output probability amplitudes of all the states. Again, maintains pytree structure
    #     """
        
    #     weights = U + 0j
    #     new_amps = jax.tree_util.tree_map(lambda amp, states_array, states_factorial: self.bosonic_transform(amp, weights, states_array, states_factorial), state_amps, self.all_states_idx, self.all_states_factorial)
    #     new_amps_extract, pytree_struct = jax.tree_util.tree_flatten(new_amps)
    #     new_amps = jax.tree_util.tree_unflatten(pytree_struct, jnp.sum(jnp.array(new_amps_extract), axis=0))
    #     return new_amps
    
    def linear_evolution(self, state_amps, U, jit_compile = True):
        r"""
        User function call to evolve a state under a pre-defined unitary

        :param state_amps: Probability amplitudes of the states being input into the mesh. Has to maintain pytree structure of self.possible_states_dict
        :param U: Unitary of size (N_modes x N_modes) that defines the evolution
        :return: Output probability amplitudes of all the states. Again, maintains pytree structure
        """    
        def _core_func(state_amps, U):
            weights = U + 0j
            new_amps = jax.tree_util.tree_map(lambda amp, states_array, states_factorial: self.bosonic_transform(amp, weights, states_array, states_factorial), state_amps, self.all_states_idx, self.all_states_factorial)
            new_amps_extract, pytree_struct = jax.tree_util.tree_flatten(new_amps)
            new_amps = jax.tree_util.tree_unflatten(pytree_struct, jnp.sum(jnp.array(new_amps_extract), axis=0))
            return new_amps
        
        # Conditionally JIT the core function
        core_func = jit(_core_func, static_argnums=(0,)) if jit_compile else _core_func
        return core_func(state_amps, U)
        

    @partial(jit, static_argnums=(0,))
    def bosonic_transform(self, amp, U, states_array_idx, states_array_factorial):
        r"""
        Internal function (maybe) to calculate the probability amplitudes of the states

        :param amp: Probability amplitude of a given state, as passed via tree_map from add_linear_layer
        :param U: Unitary matrix weights, typically matrix implemented by MZI mesh derived from theta, phi, D, alpha, beta passed to add_linear_layer
        :param states_array_idx: Indices of states to extract sub-matrix U_st from matrix U
        :param states_array_factorial: product of factorial of corresponding states
        :return: Pytree (structure of self.possible_states_dict) of how every state maps to every other state
        """
        def make_U_st(U, idx_array):
            r"""
            Make U_st matrix from U - m_{k} copies of the k^{th} input state column and n_{k} copies of the k^{th} output state row

            :param U: Matrix to calculate the sub-matrices
            :param idx_array: Indices corresponding to the m_{k} and n_{k} states, generated in self.all_states_idx
            :return: submatrix U_{st}
            """
            idx_x, idx_y = idx_array[0], idx_array[1]
            return U[:, idx_x][idx_y]

        U_st = jax.vmap(lambda idx_val: make_U_st(U, states_array_idx[idx_val]))(self.arange_possible_states)
        perm_vals = jax.vmap(lambda idx: self.lo.calc_perm(U_st[idx]))(self.arange_possible_states)
        new_amps = amp * perm_vals / jnp.sqrt(states_array_factorial)
        # new_probs = amp**2 * (jnp.abs(perm_vals)/jnp.sqrt(states_array_factorial))**2
        return new_amps

    @partial(jit, static_argnums=(0,))
    def add_3ls_nonlinear_layer(self, state_amps, chi_1_array, chi_2_array):
        r"""
        User function call to add a layer of three-level system based photon subtraction/injection nonlinearity into the circuit

        :param state_amps: Probability amplitudes of type dict, maintain the pytree structure of self.possible_states_dict
        :param chi_1_array: Array of size N_modes, to indicate the phase-shifts on the single photon component
        :param chi_2_array: Array of size N_modes, to indicate the phase-shifts on the (N - 1) photon component
        """
        # assert len(chi_1) == N_modes
        # assert len(chi_2) == N_modes
        state_amps_out = jax.tree_util.tree_map(lambda amp, state_array: self.nonlinearity_3ls(amp, state_array, chi_1_array, chi_2_array), state_amps, self.possible_states_dict)
        return state_amps_out

    @partial(jit, static_argnums=(0,))
    def nonlinearity_3ls(self, state_amp, state_array, chi_1_array, chi_2_array):
        r"""
        Internal function, logic is a bit obscure. Note that this function is being called via tree_map
        Pytree structure is defined by state_array

        :param state_amp: Probability amplitude of a given state
        :param state_array: Defines the pytree structure to follow - by default self.possible_states_dict
        :param chi_1_array: Array of size N_modes, to indicate the phase-shifts on the single photon component
        :param chi_2_array: Array of size N_modes, to indicate the phase-shifts on the (N - 1) photon component

        """

        def zero_photon_phase(chi_1_val, chi_2_val, state_array_elem):
            return 1.0 + 0j

        @jit
        def multi_photon_phase(chi_1_val, chi_2_val, state_array_elem):
            return jnp.exp(1j * chi_1_val + 1j * (state_array_elem - 1) * chi_2_val)

        @jit
        def photon_num_nl(idx, state_array_elem):
            return jax.lax.cond(state_array_elem >= 1,
                                lambda _: multi_photon_phase(*_),
                                lambda _: zero_photon_phase(*_),
                                (chi_1_array[idx], chi_2_array[idx], state_array_elem),
                                )

        phase_nl = jax.vmap(lambda idx: photon_num_nl(idx, state_array[idx]))(self.arange_possible_states)
        return state_amp * jnp.prod(phase_nl, axis=0)
    
    def visualize_state(self, amps, figsize = None, cmap = None, x_fontsize = 9, bar_width = 0.8, ylim = (0.0, 1.05)):
        r"""
        Function to visualize a histogram of state amplitudes.
        
        :param amps: Probability amplitudes of type dict, maintain the pytree structure of self.possible_states_dict
        :param figsize:
        :param cmap:
        :param x_fontsize:
        :param bar_width:
        :param ylim:
        
        """
        if cmap == None:
            cmap = plt.colormaps.get_cmap('Blues')
        else:
            pass

        amps_array, basis_elements = [], []
        for idx, s in enumerate(amps):
            amps_array.append(np.abs(amps[s]))
            string = '$| '
            for jdx in range(len(s)):
                string += str(s[jdx])
            string = string + ' \\rangle $'
            basis_elements.append(string)  

        if figsize == None:
            fig, ax = plt.subplots(figsize = (len(basis_elements)/5, 3.5))
        else:
            fig, ax = plt.subplots(figsize = figsize)

        ax.bar(np.arange(0, len(basis_elements)), np.array(amps_array), facecolor = cmap(0.32), edgecolor = cmap(0.8), width = bar_width)
        ax.set_xticks(np.arange(0, len(basis_elements)), basis_elements, fontsize = x_fontsize, rotation = 90)
        ax.set_ylim(ylim)
        plt.show()


"""TODO: Update this class to operate on Pytrees rather than structured memory states"""
class Circuit_multimode:

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
                    self.counts.append(count + self.N_modes)
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


        print ("***Circuit Ready For Compilation***")

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
            modes = jax.lax.cond(((bool_1 == 1) & (bool_2 == 0)), apply_one_photon_NL_k1, pass_one_photon_NL, idx, modes, self.tle.S_mat_t(self.k, self.k))
            modes = jax.lax.cond(((bool_1 == 0) & (bool_2 == 1)), apply_one_photon_NL_k2, pass_one_photon_NL, idx, modes, self.tle.S_mat_t(self.k, self.k))
            modes = jax.lax.cond(((bool_1 == 0) & (bool_2 == 0)), pass_one_photon_NL, pass_one_photon_NL, idx, modes, 1)
            return modes

        for idx in range(len(self.vmap_ones)):
            modes = one_photon_NL(idx, emitter_bools, modes)
    
        return modes
