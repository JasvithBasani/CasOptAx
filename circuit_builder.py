import itertools
from functools import partial
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt

from .linear_optics import Linear_Optics

class Circuit_singlemode:
    """Constructs and manages a single-mode quantum optical circuit.

    Attributes:
        n_modes: Number of spatial/waveguide modes in the circuit.
        n_photons: Number of photons in the circuit.
        input_photons: Tuple indicating which initial state is populated.
    """

    def __init__(
        self,
        n_modes: int,
        n_photons: int,
        input_photons: Tuple[int, ...],
        verbose: bool = False,
    ):
        """Initializes the circuit and precomputes all required matrices.

        Args:
            n_modes: Number of spatial/waveguide modes.
            n_photons: Number of photons (conserved).
            input_photons: Tuple of length n_modes summing to n_photons.
            verbose: If True, prints initialization status.

        Raises:
            AssertionError: If photon sums or mode lengths do not match.
        """
        # TODO: Superpositions of states as input is not yet supported
        if verbose: 
            print("Initializing Circuit, Please Wait . . .")
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.input_photons = input_photons

        assert jnp.sum(jnp.array(self.input_photons)) == self.n_photons
        assert len(self.input_photons) == self.n_modes

        # def generate_possible_states(modes: int, photons: int) -> List[List[int]]:
        #     """DEPRECATED: Generates all possible states using recursion."""
        #     def generate_state(m: int, p: int, current_state: List[int]) -> None:
        #         if m == 0:
        #             if p == 0:
        #                 valid_state.append(current_state[:])
        #             return
        #         for num in range(0, p + 1):
        #             if num <= p:
        #                 current_state.append(num)
        #                 generate_state(m - 1, p - num, current_state)
        #                 current_state.pop()

        #     valid_state = []
        #     generate_state(modes, photons, [])
        #     return valid_state

        # def generate_possible_states_backtrack(
        #     modes: int, photons: int
        # ) -> Tuple[List[List[int]], Dict[Tuple[int, ...], complex]]:
        #     """DEPRECATED: Generates all possible states using backtracking."""
        #     result_list, result_dict = [], {}

        #     def backtrack(remaining_sum: int, index: int) -> None:
        #         if index == modes:
        #             if remaining_sum == 0:
        #                 result_list.append(current_state[:])
        #                 result_dict[tuple(current_state)] = 0 + 0j
        #             return

        #         for i in range(remaining_sum + 1):
        #             current_state[index] = i
        #             backtrack(remaining_sum - i, index + 1)

        #     current_state = [0] * modes
        #     backtrack(photons, 0)
        #     return result_list, result_dict

        def generate_possible_states_optimized(
            modes: int, photons: int
        ) -> Tuple[List[List[int]], Dict[Tuple[int, ...], complex]]:
            """Optimized combinations algorithm for state generation."""
            states = []
            n_positions = photons + modes - 1
            for bars in itertools.combinations(range(n_positions), modes - 1):
                padded_bars = (-1,) + bars + (n_positions,)
                state = [padded_bars[i + 1] - padded_bars[i] - 1 for i in range(modes)]
                states.append(state)
            return states, {tuple(state): 0j for state in states}

        self.possible_states_list, self.state_amps = generate_possible_states_optimized(
            self.n_modes, self.n_photons
        )
        self.state_amps[tuple(self.input_photons)] = 1 + 0j

        self.num_possible_states = len(self.possible_states_list)
        self.arange_possible_states = jnp.arange(0, self.num_possible_states)

        self.possible_states_dict = self.state_amps.copy()
        for s in self.possible_states_list:
            self.possible_states_dict[tuple(s)] = jnp.array(s)

        idx_matrix_np = np.zeros(
            (self.num_possible_states, self.n_photons), dtype=np.int16
        )
        for i, s in enumerate(self.possible_states_list):
            idx_array = np.concatenate(
                [np.full(num, count) for count, num in enumerate(s)]
            )
            idx_matrix_np[i] = idx_array
            
        self.idx_matrix = jnp.array(idx_matrix_np)
        
        states_matrix_np = np.array(self.possible_states_list)
        self.factorial_products = jnp.array(jnp.prod(factorial(states_matrix_np), axis=1))

        # d_size = self.num_possible_states
        # all_states_idx_np = np.empty((d_size, d_size, 2, self.n_photons), dtype=np.int16)
        # all_states_idx_np[:, :, 0, :] = idx_matrix_np[:, None, :]
        # all_states_idx_np[:, :, 1, :] = idx_matrix_np[None, :, :]

        # self.all_states_idx_matrix = jnp.array(all_states_idx_np)

        # states_matrix_np = np.array(self.possible_states_list)
        # factorial_products = jnp.prod(factorial(states_matrix_np), axis=1)

        # self.all_states_factorial_matrix = jnp.outer(
        #     factorial_products, factorial_products
        # )

        # self.all_states_factorial = {
        #     tuple(state): self.all_states_factorial_matrix[idx]
        #     for idx, state in enumerate(self.possible_states_list)
        # }

        # self.all_states_idx = {
        #     tuple(state): self.all_states_idx_matrix[idx]
        #     for idx, state in enumerate(self.possible_states_list)
        # }

        n_p = self.n_photons
        self.perm_sign = jnp.array(np.tile(np.array([-1.0, 1.0]), 2 ** (n_p - 1)))

        bin_idx = (np.arange(2**n_p) + 1) % (2**n_p)
        new_grey = bin_idx ^ (bin_idx // 2)
        new_grey_rolled = np.roll(new_grey, 1)
        self.perm_direction = jnp.array(
            np.where(new_grey < new_grey_rolled, 1.0, -1.0)
        )

        indices = [
            (i + 1 & -(i + 1)).bit_length() - 1 for i in range(2 ** (n_p - 1))
        ]
        self.perm_grey_diff_index = jnp.array(indices * 2, dtype=jnp.int16)

        self._linear_evolution_jit = jax.jit(self._core_linear_evolution)
        self.lo = Linear_Optics(self.n_modes)

        if verbose:
            print("***Circuit Ready For Compilation***")

    def __hash__(self) -> int:
        """Enables JAX to hash the class for static_argnums JIT compilation."""
        return hash((self.n_modes, self.n_photons, tuple(self.input_photons)))

    def __eq__(self, other: object) -> bool:
        """Checks equality for JAX JIT recompilation triggers."""
        if not isinstance(other, Circuit_singlemode):
            return False
        return (self.n_modes, self.n_photons, tuple(self.input_photons)) == (
            other.n_modes,
            other.n_photons,
            tuple(other.input_photons),
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def calc_perm(self, U_sub):
        r"""Calculates the permanent of a submatrix using the highly optimized Ryser formula.

        This implementation evaluates the permanent with zero overhead generation by 
        leveraging static Gray code and sign arrays pre-calculated during class 
        initialization. It is strictly compiled for XLA acceleration.

        Args:
            U_sub (jax.Array): A square submatrix of shape (N_photons, N_photons) extracted 
                from the spatial unitary matrix.

        Returns:
            jax.Array: A scalar JAX array representing the complex permanent of the submatrix.
        """
        # U_sub is indexed based on the pre-calculated gray code difference map
        new_vector = U_sub[self.perm_grey_diff_index]
        # Row-wise broadcast multiplication by the direction array
        scaled_vector = new_vector * self.perm_direction[:, None]
        # Cumulative sum and reduction
        reduced = jnp.prod(jnp.cumsum(scaled_vector, axis=0), axis=1)
        perm = jnp.sum(reduced * self.perm_sign)

        return perm
    
    @partial(jax.jit, static_argnums=(0,))
    def get_fock_transition_matrix(self, U: jnp.ndarray) -> jnp.ndarray:
        r"""Converts a spatial unitary matrix into a Fock-basis transition matrix.

        This function maps an N_modes x N_modes spatial unitary to a DxD Fock transition 
        matrix. It iterates over output states sequentially to strictly cap memory usage,
        extracting all necessary submatrices and calculating their permanents using 
        JAX `vmap` for the input state dimension.

        Args:
            U (jax.Array): The spatial unitary matrix of shape (N_modes, N_modes).

        Returns:
            jax.Array: The complex DxD transition matrix in the Fock basis.
        """
        def process_row(output_idx):
            # 1. Slice the rows of the Unitary based on the output state (shape: N_photons, N_modes)
            U_row_sliced = U[output_idx, :]
            
            # 2. Extract the columns for all possible input states (shape: D, N_photons, N_photons)
            U_st_row = jax.vmap(lambda input_idx: U_row_sliced[:, input_idx])(self.idx_matrix)
            
            # 3. Calculate permanents for this row
            return jax.vmap(self.calc_perm)(U_st_row)
            
        # lax.map loops over the D states sequentially to cap peak memory (replaces outer vmap)
        perms = jax.lax.map(process_row, self.idx_matrix)
        
        # Broadcast the 1D factorial array across rows and columns to reconstruct the DxD math
        f_sqrt = jnp.sqrt(self.factorial_products)
        U_fock = (perms / f_sqrt[:, None] / f_sqrt[None, :]).T
        
        return U_fock
    
    @partial(jax.jit, static_argnums=(0,))
    def add_linear_layer(
        self,
        state_amps: Dict[Tuple[int, ...], complex],
        theta: jnp.ndarray,
        phi: jnp.ndarray,
        d_phase: jnp.ndarray,
        alpha: float,
        beta: float,
    ) -> Dict[Tuple[int, ...], complex]:
        """Evolves the quantum state through a parameterized optical mesh.
        
        This function constructs the spatial unitary from the mesh parameters and physical 
        imperfections using the Clements decomposition. It then flattens the input state 
        PyTree, applies the Fock transition matrix via a single matrix-vector multiplication, 
        and reconstructs the output PyTree.

        Args:
            state_amps: Probability amplitudes of the input states.
            theta: Optimizable internal phase shifts of the mesh MZIs.
            phi: Optimizable external phase shifts of the mesh MZIs.
            d_phase: Optimizable phase shifts for the output phase screen.
            alpha: Non-optimizable directional coupler error.
            beta: Non-optimizable directional coupler error.

        Returns:
            The output probability amplitudes maintaining the PyTree structure.
        """
        u_matrix = self.lo.clements_matrix(theta, phi, d_phase, alpha, beta)
        amps_vec = jnp.array(jax.tree_util.tree_leaves(state_amps))
        u_fock = self.get_fock_transition_matrix(u_matrix)
        new_amps_vec = u_fock @ amps_vec
        pytree_struct = jax.tree_util.tree_structure(state_amps)
        return jax.tree_util.tree_unflatten(pytree_struct, new_amps_vec)
    
    @partial(jax.jit, static_argnums=(0, 3))
    def _core_linear_evolution(
        self,
        state_amps: Dict[Tuple[int, ...], complex],
        u_matrix: jnp.ndarray,
        use_checkpoint: bool = False,
    ) -> Dict[Tuple[int, ...], complex]:
        """Core implementation for linear state evolution under a unitary.
        
        This method flattens the PyTree dictionary into a 1D state vector, calculates 
        the associated DxD Fock transition matrix, performs a matrix-vector dot product 
        for ultra-fast evolution, and re-packages the state into its original PyTree 
        structure.

        Args:
            state_amps (dict): Probability amplitudes of the input states. Must maintain 
                the PyTree structure of `self.possible_states_dict`.
            U (jax.Array): The predefined spatial unitary matrix of shape (N_modes, N_modes) 
                dictating the evolution.
            use_checkpoint (bool, optional): If True, uses gradient checkpointing 
                (rematerialization) to significantly reduce memory usage during 
                backpropagation at the cost of extra compute. Defaults to False.

        Returns:
            dict: The evolved probability amplitudes, maintaining the exact same PyTree 
                dictionary structure as the input `state_amps`.
        """
        amps_vec = jnp.array(jax.tree_util.tree_leaves(state_amps))
        if use_checkpoint:
            fock_fn = jax.checkpoint(self.get_fock_transition_matrix)
        else:
            fock_fn = self.get_fock_transition_matrix

        u_fock = fock_fn(u_matrix)
        new_amps_vec = u_fock @ amps_vec
        pytree_struct = jax.tree_util.tree_structure(state_amps)
        return jax.tree_util.tree_unflatten(pytree_struct, new_amps_vec)

    def linear_evolution(
        self,
        state_amps: Dict[Tuple[int, ...], complex],
        u_matrix: jnp.ndarray,
        jit_compile: bool = True,
        use_checkpoint: bool = False,
    ) -> Dict[Tuple[int, ...], complex]:
        """User-facing wrapper for linear evolution.

        Args:
            state_amps: Input state amplitudes.
            u_matrix: Spatial unitary matrix.
            jit_compile: Whether to use the JIT-compiled execution path.
            use_checkpoint: Whether to enable memory-saving checkpointing.

        Returns:
            Evolved state amplitudes.
        """
        if jit_compile:
            return self._core_linear_evolution(state_amps, u_matrix, use_checkpoint)
        
        return self._core_linear_evolution.__wrapped__(
            self, state_amps, u_matrix, use_checkpoint
        )

    @partial(jax.jit, static_argnums=(0,))
    def add_linear_layer_deprecated(self, state_amps, theta, phi, D, alpha, beta):
        r"""
        DEPRECATED: User function call to add a linear optical mesh transformation into the circuit
        """

        weights = self.lo.clements_matrix(theta, phi, D, alpha, beta)
        new_amps = jax.tree_util.tree_map(lambda amp, states_array, states_factorial: self.bosonic_transform_deprecated(amp, weights, states_array, states_factorial), state_amps, self.all_states_idx, self.all_states_factorial)
        new_amps_extract, pytree_struct = jax.tree_util.tree_flatten(new_amps)
        new_amps = jax.tree_util.tree_unflatten(pytree_struct, jnp.sum(jnp.array(new_amps_extract), axis=0))
        return new_amps
    
    def _core_linear_evolution_deprecated(self, state_amps, U):
        r"""
        DEPRECATED: Core function implementation of linear_evolution. Abstracted away to allow conditional jitting
        """
        weights = U + 0j
        new_amps = jax.tree_util.tree_map(lambda amp, states_array, states_factorial: self.bosonic_transform_deprecated(amp, weights, states_array, states_factorial), state_amps, self.all_states_idx, self.all_states_factorial)
        new_amps_extract, pytree_struct = jax.tree_util.tree_flatten(new_amps)
        new_amps = jax.tree_util.tree_unflatten(pytree_struct, jnp.sum(jnp.array(new_amps_extract), axis=0))
        return new_amps

    @partial(jax.jit, static_argnums=(0,))
    def bosonic_transform_deprecated(self, amp, U, states_array_idx, states_array_factorial):
        r"""
        DEPRECATED: Internal function (maybe) to calculate the probability amplitudes of the states
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
    
    @partial(jax.jit, static_argnums=(0,))
    def add_3ls_nonlinear_layer(
        self,
        state_amps: Dict[Tuple[int, ...], complex],
        chi_1_array: jnp.ndarray,
        chi_2_array: jnp.ndarray,
    ) -> Dict[Tuple[int, ...], complex]:
        """Adds a layer of 3LS photon subtraction/injection nonlinearity.

        Args:
            state_amps: Probability amplitudes maintaining PyTree structure.
            chi_1_array: Phase-shifts on the single photon component (size n_modes).
            chi_2_array: Phase-shifts on the (N-1) photon component (size n_modes).

        Returns:
            The updated probability amplitudes.
        """
        return jax.tree_util.tree_map(
            lambda amp, state_array: self.nonlinearity_3ls(
                amp, state_array, chi_1_array, chi_2_array
            ),
            state_amps,
            self.possible_states_dict,
        )

    @partial(jax.jit, static_argnums=(0,))
    def nonlinearity_3ls(
        self,
        state_amp: complex,
        state_array: jnp.ndarray,
        chi_1_array: jnp.ndarray,
        chi_2_array: jnp.ndarray,
    ) -> complex:
        """Applies mode-wise nonlinear phase shifts to a given state.

        Args:
            state_amp: Probability amplitude of a given state.
            state_array: Photon numbers for each mode in this state.
            chi_1_array: Phase-shifts on the single photon component.
            chi_2_array: Phase-shifts on the multi-photon components.

        Returns:
            The newly evaluated complex amplitude for this state.
        """
        # A simple mask cleanly resolves the logic without needing lax.cond
        # or indexing that could throw out-of-bounds errors.
        mask = state_array >= 1
        multi_phase = jnp.exp(1j * chi_1_array + 1j * (state_array - 1) * chi_2_array)
        zero_phase = 1.0 + 0j
        
        # Applies phase per mode simultaneously 
        phase_nl = jnp.where(mask, multi_phase, zero_phase)
        return state_amp * jnp.prod(phase_nl)

    @partial(jax.jit, static_argnums=(0,))
    def add_3ls_nonlinear_layer_deprecated(self, state_amps, chi_1_array, chi_2_array):
        r"""
        DEPRECATED: User function call to add a layer of three-level system based photon subtraction/injection nonlinearity into the circuit

        :param state_amps: Probability amplitudes of type dict, maintain the pytree structure of self.possible_states_dict
        :param chi_1_array: Array of size N_modes, to indicate the phase-shifts on the single photon component
        :param chi_2_array: Array of size N_modes, to indicate the phase-shifts on the (N - 1) photon component
        """
        # assert len(chi_1) == N_modes
        # assert len(chi_2) == N_modes
        state_amps_out = jax.tree_util.tree_map(lambda amp, state_array: self.nonlinearity_3ls(amp, state_array, chi_1_array, chi_2_array), state_amps, self.possible_states_dict)
        return state_amps_out

    @partial(jax.jit, static_argnums=(0,))
    def nonlinearity_3ls_deprecated(self, state_amp, state_array, chi_1_array, chi_2_array):
        r"""
        DEPRECATED: Internal function, logic is a bit obscure. Note that this function is being called via tree_map
        Pytree structure is defined by state_array

        :param state_amp: Probability amplitude of a given state
        :param state_array: Defines the pytree structure to follow - by default self.possible_states_dict
        :param chi_1_array: Array of size N_modes, to indicate the phase-shifts on the single photon component
        :param chi_2_array: Array of size N_modes, to indicate the phase-shifts on the (N - 1) photon component

        """

        def zero_photon_phase(chi_1_val, chi_2_val, state_array_elem):
            return 1.0 + 0j

        @jax.jit
        def multi_photon_phase(chi_1_val, chi_2_val, state_array_elem):
            return jnp.exp(1j * chi_1_val + 1j * (state_array_elem - 1) * chi_2_val)

        @jax.jit
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

    @partial(jax.jit, static_argnums = (0, 1, ))
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
    @partial(jax.jit, static_argnums = (0, ))
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

    @partial(jax.jit, static_argnums = (0, ))
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

    @partial(jax.jit, static_argnums=(0,))
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

    @partial(jax.jit, static_argnums = (0, ))
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
