import itertools
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

# Enforce 64-bit precision to prevent catastrophic numerical cancellation in permanents.
jax.config.update("jax_enable_x64", True)


# ==============================================================================
# PURE JAX FUNCTIONS (Out-of-class Helper Functions)
# ==============================================================================

def _calc_perm_prod(
    u_sub: jnp.ndarray,
    perm_grey_diff_index: jnp.ndarray,
    perm_direction: jnp.ndarray,
    perm_sign: jnp.ndarray,
) -> jnp.ndarray:
    """Computes the permanent of a square matrix using the Ryser-Glynn algorithm.

    This function is designed to be pure and stateless so it can be efficiently
    vectorized across batches using `jax.vmap`.

    Args:
        u_sub: A 2D complex JAX array representing the square submatrix.
        perm_grey_diff_index: A 1D integer array mapping the Gray code index
            differences for row scaling.
        perm_direction: A 1D float array containing 1.0 or -1.0 indicating the
            traversal direction through the Gray code.
        perm_sign: A 1D float array of alternating signs used for computing the
            final Ryser sum.

    Returns:
        A complex scalar representing the permanent of the submatrix.
    """
    new_vector = u_sub[perm_grey_diff_index]
    scaled_vector = new_vector * perm_direction[:, None]
    reduced = jnp.prod(jnp.cumsum(scaled_vector, axis=0), axis=1)
    return jnp.sum(reduced * perm_sign)


# def _get_fock_transition_matrix(
#     u_matrix: jnp.ndarray,
#     idx_matrix: jnp.ndarray,
#     factorial_products: jnp.ndarray,
#     perm_grey_diff_index: jnp.ndarray,
#     perm_direction: jnp.ndarray,
#     perm_sign: jnp.ndarray,
# ) -> jnp.ndarray:
#     """Constructs the complete Fock-space transition matrix.

#     Maps the single-photon unitary matrix to the multiphoton Fock space by
#     evaluating the matrix permanent for all combinations of input and output
#     states. Uses an internal router to switch between `vmap` and `lax.map`
#     to prevent memory crashes on large state spaces.

#     Args:
#         u_matrix: The single-photon mode-to-mode complex transition matrix.
#         idx_matrix: A 2D integer array mapping each Fock state to its constituent
#             waveguide mode indices.
#         factorial_products: A 1D float array containing the product of state
#             occupation factorials used for normalization.
#         perm_grey_diff_index: Precomputed Ryser Gray code index metadata.
#         perm_direction: Precomputed Ryser traversal direction tracking array.
#         perm_sign: Precomputed Ryser alternating sign sequence array.

#     Returns:
#         A 2D complex array representing the full transition amplitudes between
#         all possible configurations in the Fock space.
#     """
#     def process_row(out_state_idx: jnp.ndarray) -> jnp.ndarray:
#         u_row_sliced = u_matrix[out_state_idx, :]
#         u_st_row = jax.vmap(lambda input_idx: u_row_sliced[:, input_idx])(idx_matrix)
#         calc_fn = partial(
#             _calc_perm_prod,
#             perm_grey_diff_index=perm_grey_diff_index,
#             perm_direction=perm_direction,
#             perm_sign=perm_sign,
#         )
#         return jax.vmap(calc_fn)(u_st_row)

#     num_states = idx_matrix.shape[0]

#     if num_states <= 1024:
#         perms = jax.vmap(process_row)(idx_matrix)
#     else:
#         perms = jax.lax.map(process_row, idx_matrix)

#     f_sqrt = jnp.sqrt(factorial_products)
#     return perms / f_sqrt[:, None] / f_sqrt[None, :]


def _get_fock_transition_matrix(
    u_matrix: jnp.ndarray,
    idx_matrix: jnp.ndarray,
    factorial_products: jnp.ndarray,
    perm_grey_diff_index: jnp.ndarray,
    perm_direction: jnp.ndarray,
    perm_sign: jnp.ndarray,
    batch_size: int = 256,
) -> jnp.ndarray:
    """Constructs the complete Fock-space transition matrix.

    Maps the single-photon unitary matrix to the multiphoton Fock space by
    evaluating the matrix permanent for all combinations of input and output
    states. Uses an internal router to switch between `vmap` and `lax.map`
    to prevent memory crashes on large state spaces.

    Args:
        u_matrix: The single-photon mode-to-mode complex transition matrix.
        idx_matrix: A 2D integer array mapping each Fock state to its constituent
            waveguide mode indices.
        factorial_products: A 1D float array containing the product of state
            occupation factorials used for normalization.
        perm_grey_diff_index: Precomputed Ryser Gray code index metadata.
        perm_direction: Precomputed Ryser traversal direction tracking array.
        perm_sign: Precomputed Ryser alternating sign sequence array.

    Returns:
        A 2D complex array representing the full transition amplitudes between
        all possible configurations in the Fock space.
    """
    def process_row(out_state_idx: jnp.ndarray) -> jnp.ndarray:
        u_row_sliced = u_matrix[out_state_idx, :]
        u_st_row = jax.vmap(lambda input_idx: u_row_sliced[:, input_idx])(idx_matrix)
        calc_fn = partial(
            _calc_perm_prod,
            perm_grey_diff_index=perm_grey_diff_index,
            perm_direction=perm_direction,
            perm_sign=perm_sign,
        )
        return jax.vmap(calc_fn)(u_st_row)

    num_states = idx_matrix.shape[0]

    # Calculate how much padding is needed to make num_states divisible by batch_size
    pad_len = (batch_size - (num_states % batch_size)) % batch_size
    
    # Pad the matrix with duplicate rows (we will discard the results later)
    padded_idx_matrix = jnp.pad(idx_matrix, ((0, pad_len), (0, 0)), mode="wrap")
    num_batches = padded_idx_matrix.shape[0] // batch_size
    
    # Reshape into chunks: (num_batches, batch_size, n_photons)
    batched_idx = padded_idx_matrix.reshape(num_batches, batch_size, -1)

    # vmap over the batch, map over the chunks
    def process_batch(batch_idx):
        return jax.vmap(process_row)(batch_idx)

    # Execute the chunked operation
    padded_perms = jax.lax.map(process_batch, batched_idx)
    
    # Flatten the batches back out and slice off the padding
    perms = padded_perms.reshape(-1, num_states)[:num_states, :]

    # Apply factorial normalizations
    f_sqrt = jnp.sqrt(factorial_products)
    return perms / f_sqrt[:, None] / f_sqrt[None, :]


@partial(jax.jit, static_argnames=["use_checkpoint"])
def _evolve_state_jax(
    amps_vec: jnp.ndarray,
    u_matrix: jnp.ndarray,
    idx_matrix: jnp.ndarray,
    factorial_products: jnp.ndarray,
    perm_grey_diff_index: jnp.ndarray,
    perm_direction: jnp.ndarray,
    perm_sign: jnp.ndarray,
    use_checkpoint: bool = False,
) -> jnp.ndarray:
    """Evolves a 1D state vector through a transformation matrix using XLA compilation.

    Args:
        amps_vec: A 1D complex array of the current state amplitudes.
        u_matrix: The single-photon transformation matrix.
        idx_matrix: A 2D integer array mapping state mode allocations.
        factorial_products: A 1D float array of precomputed factorial normalizations.
        perm_grey_diff_index: Precomputed Ryser Gray code index metadata.
        perm_direction: Precomputed Ryser traversal direction tracking array.
        perm_sign: Precomputed Ryser alternating sign sequence array.
        use_checkpoint: If True, applies JAX gradient checkpointing to conserve
            device memory during backpropagation.

    Returns:
        A 1D complex array containing the evolved state amplitudes.
    """
    fock_fn = _get_fock_transition_matrix
    if use_checkpoint:
        fock_fn = jax.checkpoint(fock_fn)

    u_fock = fock_fn(
        u_matrix,
        idx_matrix,
        factorial_products,
        perm_grey_diff_index,
        perm_direction,
        perm_sign,
    )
    return u_fock @ amps_vec


def _core_3ls_nonlinearity(
    amps_vec: jnp.ndarray,
    states_matrix: jnp.ndarray,
    chi_1_array: jnp.ndarray,
    chi_2_array: jnp.ndarray,
) -> jnp.ndarray:
    """Applies mode-wise nonlinear phase shifts simultaneously across all states.

    Uses JAX array broadcasting to completely bypass dictionary mapping loops,
    evaluating multi-photon phase interactions in a single vectorized pass.

    Args:
        amps_vec: A 1D complex array of the current state amplitudes.
        states_matrix: A 2D integer array mapping states to mode distributions.
        chi_1_array: A 1D float array of phase shifts applied to single-photon
            components (shape: n_modes).
        chi_2_array: A 1D float array of phase shifts applied to multi-photon
            components (shape: n_modes).

    Returns:
        A 1D complex array representing the updated state amplitudes.
    """
    mask = states_matrix >= 1
    multi_phase = jnp.exp(
        1j * chi_1_array[None, :] + 1j * (states_matrix - 1) * chi_2_array[None, :]
    )
    zero_phase = 1.0 + 0j
    phase_nl = jnp.where(mask, multi_phase, zero_phase)
    total_phase = jnp.prod(phase_nl, axis=1)
    return amps_vec * total_phase


_core_3ls_nonlinearity_jit = jax.jit(_core_3ls_nonlinearity)


@jax.jit
def _calc_targeted_amplitudes(
    u_matrix: jnp.ndarray,
    in_idx_matrix: jnp.ndarray,
    out_idx_matrix: jnp.ndarray,
    in_facts: jnp.ndarray,
    out_facts: jnp.ndarray,
    perm_grey_diff_index: jnp.ndarray,
    perm_direction: jnp.ndarray,
    perm_sign: jnp.ndarray,
) -> jnp.ndarray:
    """Calculates an M x N submatrix of specific input-to-output transitions.

    Args:
        u_matrix: The single-photon complex transformation matrix.
        in_idx_matrix: A 2D integer array of mode indices for the M input states.
        out_idx_matrix: A 2D integer array of mode indices for the N output states.
        in_facts: A 1D float array of factorial products for the M input states.
        out_facts: A 1D float array of factorial products for the N output states.
        perm_grey_diff_index: Precomputed Ryser Gray code index metadata.
        perm_direction: Precomputed Ryser traversal direction tracking array.
        perm_sign: Precomputed Ryser alternating sign sequence array.

    Returns:
        A 2D complex array of shape (M, N) containing transition amplitudes.
    """
    def process_input(in_idx: jnp.ndarray, in_f: float) -> jnp.ndarray:
        def single_transition(out_idx: jnp.ndarray, out_f: float) -> complex:
            u_sub = u_matrix[out_idx, :][:, in_idx]
            perm = _calc_perm_prod(
                u_sub, perm_grey_diff_index, perm_direction, perm_sign
            )
            return perm / jnp.sqrt(in_f * out_f)
        
        return jax.vmap(single_transition)(out_idx_matrix, out_facts)

    return jax.vmap(process_input)(in_idx_matrix, in_facts)


# ==============================================================================
# MAIN CIRCUIT CLASS
# ==============================================================================

class Circuit_singlemode:
    """Constructs and manages a single-mode quantum optical circuit.

    Provides a dual-execution architecture: a fully JIT-compiled XLA path for
    high-performance GPU/CPU execution, and an eager-execution Python loop path
    for instant debugging and verification.

    Attributes:
        n_modes: The number of spatial/waveguide modes in the circuit.
        n_photons: The total conserved number of photons.
        input_photons: A tuple indicating the initial populated state.
        debug_mode: If True, bypasses XLA compilation for eager execution.
        possible_states_list: A list of all valid photon state distributions.
        possible_states_dict: A dictionary mapping state tuples to amplitudes.
    """

    def __init__(
        self,
        n_modes: int,
        n_photons: int,
        input_photons: Tuple[int, ...],
        linear_optics_instance: Optional[Any] = None,
        verbose: bool = False,
        debug_mode: bool = False,
    ):
        """Initializes the circuit and precomputes required indices and metadata.

        Args:
            n_modes: Number of spatial/waveguide modes. Must be > 0.
            n_photons: Number of photons (conserved). Must be >= 0.
            input_photons: Tuple of length n_modes summing to n_photons.
            linear_optics_instance: An optional pre-initialized instance of a
                Linear_Optics class.
            verbose: If True, prints initialization and execution status.
            debug_mode: If True, bypasses JIT compilation for tracking.

        Raises:
            ValueError: If parameters are logically or physically invalid.
        """
        if verbose:
            print("Initializing Circuit, Please Wait . . .")

        # Compact validation checks
        if not isinstance(n_modes, int) or n_modes <= 0: raise ValueError(f"n_modes must be a positive integer, got {n_modes}")
        if not isinstance(n_photons, int) or n_photons < 0: raise ValueError(f"n_photons must be a non-negative integer, got {n_photons}")
        if not isinstance(input_photons, tuple) or len(input_photons) != n_modes: raise ValueError(f"input_photons must be a tuple of length {n_modes}")
        if sum(input_photons) != n_photons or any(p < 0 or not isinstance(p, int) for p in input_photons): raise ValueError("input_photons must sum to n_photons and contain non-negative integers.")

        self.n_modes = n_modes
        self.n_photons = n_photons
        self.input_photons = input_photons
        self.debug_mode = debug_mode
        self._linear_optics_instance = linear_optics_instance

        self.possible_states_list, self.state_amps = self._generate_states(
            self.n_modes, self.n_photons
        )
        self.state_amps[tuple(self.input_photons)] = 1.0 + 0j

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
        self.states_matrix = jnp.array(states_matrix_np)
        
        self.factorial_products = jnp.array(
            jnp.prod(scipy.special.factorial(states_matrix_np), axis=1)
        )

        self._setup_ryser_arrays()

        if verbose:
            if self.debug_mode:
                print("🔧 Circuit initialized in DEBUG MODE (Eager/CPU)")
            else:
                print("🚀 Circuit initialized in PRODUCTION MODE (JIT/GPU)")

    @property
    def lo(self) -> Any:
        """Lazy loader for the Linear Optics compilation module.

        Returns:
            An instance of the Linear_Optics class.
        """
        if self._linear_optics_instance is None:
            from .linear_optics import Linear_Optics  
            self._linear_optics_instance = Linear_Optics(N_modes=self.n_modes)
        return self._linear_optics_instance

    def _generate_states(
        self, modes: int, photons: int
    ) -> Tuple[List[List[int]], Dict[Tuple[int, ...], complex]]:
        """Generates the full Fock state combinations using stars and bars.

        Args:
            modes: The number of available modes.
            photons: The total number of photons to distribute.

        Returns:
            A tuple containing a list of all valid state integer arrays, and a
            dictionary mapping those states as tuples to an initial amplitude.
        """
        states = []
        n_positions = photons + modes - 1
        for bars in itertools.combinations(range(n_positions), modes - 1):
            padded_bars = (-1,) + bars + (n_positions,)
            state = [padded_bars[i + 1] - padded_bars[i] - 1 for i in range(modes)]
            states.append(state)
        return states, {tuple(state): 0j for state in states}

    def _setup_ryser_arrays(self) -> None:
        """Precomputes structural mapping sequences for permanent calculation.

        Sets up Gray code tracking, alternating sign layouts, and column indices
        required by the Ryser permanent function, binding them to the instance
        so they are only evaluated once upon initialization.
        """
        n_p = self.n_photons
        self.perm_sign = jnp.array(np.tile(np.array([-1.0, 1.0]), 2 ** (n_p - 1)))

        bin_idx = (np.arange(2**n_p) + 1) % (2**n_p)
        new_grey = bin_idx ^ (bin_idx // 2)
        new_grey_rolled = np.roll(new_grey, 1)
        self.perm_direction = jnp.array(
            np.where(new_grey < new_grey_rolled, 1.0, -1.0)
        )

        indices = [(i + 1 & -(i + 1)).bit_length() - 1 for i in range(2 ** (n_p - 1))]
        self.perm_grey_diff_index = jnp.array(indices * 2, dtype=jnp.int16)

    # ==============================================================================
    # USER-FACING CIRCUIT ROUTING METHODS
    # ==============================================================================

    def linear_evolution(
        self,
        state_amps: Dict[Tuple[int, ...], complex],
        u_matrix: jnp.ndarray,
        use_checkpoint: bool = False,
    ) -> Dict[Tuple[int, ...], complex]:
        """Evolves full state amplitudes through an explicit transformation matrix.

        Args:
            state_amps: Dictionary mapping Fock state tuples to complex amplitudes.
            u_matrix: A 2D single-photon transformation matrix.
            use_checkpoint: If True, enables JAX gradient checkpointing during backprop.

        Returns:
            A dictionary mapping the newly evolved Fock states to their amplitudes.
        """
        amps_vec = jnp.array(jax.tree_util.tree_leaves(state_amps))

        if self.debug_mode:
            new_amps_vec = self._core_evolution_debug(amps_vec, u_matrix)
        else:
            new_amps_vec = _evolve_state_jax(
                amps_vec,
                u_matrix,
                self.idx_matrix,
                self.factorial_products,
                self.perm_grey_diff_index,
                self.perm_direction,
                self.perm_sign,
                use_checkpoint=use_checkpoint,
            )

        pytree_struct = jax.tree_util.tree_structure(state_amps)
        return jax.tree_util.tree_unflatten(pytree_struct, new_amps_vec)

    def add_linear_layer(
        self,
        state_amps: Dict[Tuple[int, ...], complex],
        theta: jnp.ndarray,
        phi: jnp.ndarray,
        d_phase: jnp.ndarray,
        alpha: Any,
        beta: Any,
        use_checkpoint: bool = False,
    ) -> Dict[Tuple[int, ...], complex]:
        """Evolves the state through a parameterized Clements linear mesh.

        Args:
            state_amps: Dictionary mapping Fock state tuples to complex amplitudes.
            theta: Parameter array for the beam splitter angles.
            phi: Parameter array for internal phase shifts.
            d_phase: Parameter array for external phase shifts.
            alpha: Component configuration parameter for directional coupler errors.
            beta: Component configuration parameter for directional coupler errors.
            use_checkpoint: If True, enables JAX gradient checkpointing.

        Returns:
            A dictionary of the updated optical state amplitudes.
        """
        u_matrix = self.lo.clements_matrix(theta, phi, d_phase, alpha, beta)
        return self.linear_evolution(state_amps, u_matrix, use_checkpoint)

    def add_3ls_nonlinear_layer(
        self,
        state_amps: Dict[Tuple[int, ...], complex],
        chi_1_array: jnp.ndarray,
        chi_2_array: jnp.ndarray,
    ) -> Dict[Tuple[int, ...], complex]:
        """Adds a layer of 3LS photon subtraction/injection nonlinearity.

        Args:
            state_amps: Probability amplitudes maintaining PyTree structure.
            chi_1_array: Phase shifts on single-photon components (size n_modes).
            chi_2_array: Phase shifts on multi-photon components (size n_modes).

        Returns:
            The updated probability amplitudes mapping back to the state tuples.
            
        Raises:
            ValueError: If phase shift arrays do not match the number of modes.
        """
        if getattr(chi_1_array, "shape", None) != (self.n_modes,): raise ValueError(f"chi_1_array shape must be {(self.n_modes,)}")
        if getattr(chi_2_array, "shape", None) != (self.n_modes,): raise ValueError(f"chi_2_array shape must be {(self.n_modes,)}")

        amps_vec = jnp.array(jax.tree_util.tree_leaves(state_amps))
        
        if self.debug_mode:
            new_amps_vec = _core_3ls_nonlinearity(
                amps_vec, self.states_matrix, chi_1_array, chi_2_array
            )
        else:
            new_amps_vec = _core_3ls_nonlinearity_jit(
                amps_vec, self.states_matrix, chi_1_array, chi_2_array
            )
            
        pytree_struct = jax.tree_util.tree_structure(state_amps)
        return jax.tree_util.tree_unflatten(pytree_struct, new_amps_vec)

    def get_targeted_transitions(
        self,
        input_states: List[Tuple[int, ...]],
        output_states: List[Tuple[int, ...]],
        u_matrix: jnp.ndarray,
    ) -> Dict[Tuple[int, ...], Dict[Tuple[int, ...], complex]]:
        """Rapidly computes transition amplitudes for lists of input/output pairs.

        Args:
            input_states: A list of initial Fock state tuples.
            output_states: A list of target output Fock state tuples.
            u_matrix: The single-photon transformation matrix.

        Returns:
            A nested dictionary structured as {input_state: {output_state: amplitude}}.

        Raises:
            ValueError: If a provided state configuration falls outside the valid
                Fock space bounds of the initialized circuit.
        """
        in_list_indices = []
        for state in input_states:
            try:
                in_list_indices.append(self.possible_states_list.index(list(state)))
            except ValueError:
                raise ValueError(f"Input state {state} is not in the valid state space.")

        out_list_indices = []
        for state in output_states:
            try:
                out_list_indices.append(self.possible_states_list.index(list(state)))
            except ValueError:
                raise ValueError(f"Output state {state} is not in the valid state space.")

        in_idx_jnp = jnp.array(in_list_indices)
        out_idx_jnp = jnp.array(out_list_indices)

        in_idx_matrix = self.idx_matrix[in_idx_jnp]
        in_facts = self.factorial_products[in_idx_jnp]

        out_idx_matrix = self.idx_matrix[out_idx_jnp]
        out_facts = self.factorial_products[out_idx_jnp]

        if self.debug_mode:
            amps_matrix = []
            for i in range(len(input_states)):
                row_amps = []
                for j in range(len(output_states)):
                    u_sub = u_matrix[out_idx_matrix[j], :][:, in_idx_matrix[i]]
                    perm = _calc_perm_prod(
                        u_sub, self.perm_grey_diff_index, self.perm_direction, self.perm_sign
                    )
                    amp = perm / jnp.sqrt(in_facts[i] * out_facts[j])
                    row_amps.append(amp)
                amps_matrix.append(row_amps)
            amps_matrix_jnp = jnp.array(amps_matrix)
        else:
            amps_matrix_jnp = _calc_targeted_amplitudes(
                u_matrix,
                in_idx_matrix,
                out_idx_matrix,
                in_facts,
                out_facts,
                self.perm_grey_diff_index,
                self.perm_direction,
                self.perm_sign,
            )

        result = {}
        for i, in_state in enumerate(input_states):
            result[in_state] = {
                out_state: amps_matrix_jnp[i, j] 
                for j, out_state in enumerate(output_states)
            }
            
        return result

    # ==============================================================================
    # DEBUG PATH METHODS (Dynamic VMAP execution, No JIT tracing)
    # ==============================================================================

    def _core_evolution_debug(
        self, amps_vec: jnp.ndarray, u_matrix: jnp.ndarray
    ) -> jnp.ndarray:
        """Eager execution fallback path bypassing JIT for sequential debugging.

        Args:
            amps_vec: 1D complex array of the current state amplitudes.
            u_matrix: The single-photon transformation matrix.

        Returns:
            A 1D complex array of the newly evaluated state amplitudes.
        """
        u_fock = _get_fock_transition_matrix(
            u_matrix,
            self.idx_matrix,
            self.factorial_products,
            self.perm_grey_diff_index,
            self.perm_direction,
            self.perm_sign,
        )
        return jnp.dot(u_fock, amps_vec)

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
