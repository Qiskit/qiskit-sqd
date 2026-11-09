# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Functions for performing self-consistent configuration recovery."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def recover_configurations(
    bitstring_matrix: np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    avg_occupancies: tuple[np.ndarray, np.ndarray],
    num_elec_a: int,
    num_elec_b: int,
    rand_seed: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine bitstrings based on average orbital occupancy and a target hamming weight for each spin species.

    This function refines each bit in isolation in an attempt to transform the Hilbert space
    represented by the input ``bitstring_matrix`` into a space closer to that which supports
    the ground state.

    .. note::

        This function makes the assumption that bit ``i`` represents the spin-down orbital
        corresponding to the spin-up orbital in bit ``i + N`` where ``N`` is the number of
        spatial orbitals and ``i < N``.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring
        probabilities: A 1D array specifying a probability distribution over the bitstrings
        avg_occupancies: A length-2 tuple of arrays holding the mean occupancy of the spin-up
            and spin-down orbitals, respectively. The occupancies should be formatted as:
            ``(array([occ_a_0, ..., occ_a_N]), array([occ_b_0, ..., occ_b_N]))``
        num_elec_a: The number of spin-up electrons in the system.
        num_elec_b: The number of spin-down electrons in the system.
        rand_seed: A seed for controlling randomness

    Returns:
        A refined bitstring matrix and an updated probability array.

    References:
        [1]: J. Robledo-Moreno, et al., `Chemistry Beyond Exact Solutions on a Quantum-Centric Supercomputer <https://arxiv.org/abs/2405.05068>`_,
             arXiv:2405.05068 [quant-ph].
    """
    rng = np.random.default_rng(rand_seed)

    if num_elec_a < 0 or num_elec_b < 0:
        raise ValueError("The numbers of electrons must be specified as non-negative integers.")

    corrected_dict: defaultdict[str, float] = defaultdict(float)
    occs_array = np.flip(avg_occupancies).flatten()
    for bitstring, freq in zip(bitstring_matrix, probabilities):
        bs_corrected = _bitstring_correcting(
            bitstring,
            occs_array,
            [num_elec_b, num_elec_a],
            rng=rng,
        )
        bs_str = "".join("1" if bit else "0" for bit in bs_corrected)
        corrected_dict[bs_str] += freq
    bs_mat_out = np.array([[bit == "1" for bit in bs] for bs in corrected_dict])
    freqs_out = np.array([f for f in corrected_dict.values()])
    freqs_out = np.abs(freqs_out) / np.sum(np.abs(freqs_out))

    return bs_mat_out, freqs_out


def recover_configurations_single_species(
    bitstring_matrix: np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    avg_occupancies: np.ndarray,
    num_elec: int,
    rand_seed: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine bitstrings based on average orbital occupancy and a target hamming weight for a single species.

    This function refines each bit in isolation in an attempt to transform the Hilbert space
    represented by the input ``bitstring_matrix`` into a space closer to that which supports
    the ground state.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring
        probabilities: A 1D array specifying a probability distribution over the bitstrings
        avg_occupancies: A 1D array holding the mean occupancy each orbital. The occupancies
            should be formatted as: ``array([occ_0, ..., occ_N])``
        num_elec: The number of electrons in the system.
        rand_seed: A seed for controlling randomness

    Returns:
        A refined bitstring matrix and an updated probability array.

    References:
        [1]: J. Robledo-Moreno, et al., `Chemistry Beyond Exact Solutions on a Quantum-Centric Supercomputer <https://arxiv.org/abs/2405.05068>`_,
             arXiv:2405.05068 [quant-ph].
    """
    rng = np.random.default_rng(rand_seed)

    if num_elec < 0:
        raise ValueError("The numbers of electrons must be specified as a non-negative integer.")

    corrected_dict: defaultdict[str, float] = defaultdict(float)
    occs_array = np.flip(avg_occupancies)
    for bitstring, freq in zip(bitstring_matrix, probabilities):
        bs_corrected = _bitstring_correcting(
            bitstring,
            occs_array,
            [num_elec],
            rng=rng,
        )
        bs_str = "".join("1" if bit else "0" for bit in bs_corrected)
        corrected_dict[bs_str] += freq
    bs_mat_out = np.array([[bit == "1" for bit in bs] for bs in corrected_dict])
    freqs_out = np.array([f for f in corrected_dict.values()])
    freqs_out = np.abs(freqs_out) / np.sum(np.abs(freqs_out))

    return bs_mat_out, freqs_out


def _p_flip_0_to_1(ratio_exp: float, occ: float, eps: float = 0.01) -> float:  # pragma: no cover
    """Calculate the probability of flipping a bit from 0 to 1.

    This function will more aggressively flip bits which are in disagreement
    with the occupation information.

    Args:
        ratio_exp: The ratio of 1's expected in the set of bits
        occ: The occupancy of a particular bit, based estimated ground
            state found at the end of each configuration recovery iteration.
        eps: A value for controlling how aggressively to flip bits

    Returns:
        The probability with which to flip the bit

    """
    # Occupancy is < than naive expectation.
    # Flip 0s to 1 with small (<eps) probability in this case
    if occ < ratio_exp:
        return occ * eps / ratio_exp

    # Occupancy is >= naive expectation.
    # The probability weight to flip the bit increases linearly from ``eps`` to
    # ``1.0`` as the occupation deviates further from the expected ratio
    if ratio_exp == 1.0:
        return eps
    slope = (1 - eps) / (1 - ratio_exp)
    intercept = 1 - slope
    return occ * slope + intercept


def _p_flip_1_to_0(ratio_exp: float, occ: float, eps: float = 0.01) -> float:  # pragma: no cover
    """Calculate the probability of flipping a bit from 1 to 0.

    This function will more aggressively flip bits which are in disagreement
    with the occupation information.

    Args:
        ratio_exp: The ratio of 1's expected in the set of bits
        occ: The occupancy of a particular bit, based estimated ground
            state found at the end of each configuration recovery iteration.
        eps: A value for controlling how aggressively to flip bits

    Returns:
        The probability with which to flip the bit

    """
    return _p_flip_0_to_1(1 - ratio_exp, 1 - occ, eps)


def _bitstring_correcting(
    bit_array: np.ndarray,
    avg_occupancies: np.ndarray,
    hamming_weights: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Use occupancy information and target hamming weight(s) to correct a bitstring.

    Args:
        bit_array: A 1D array of ``bool`` representations of bit values
        avg_occupancies: A 1D array containing the mean occupancy of each orbital.
        hamming_weights: A 1D array containing the target hamming weight(s). In the
            case of a 2 particle system, the target hamming weight of the left half
            of the bitstring comes first.
        rng: A random number generator

    Returns:
        A corrected bitstring

    """
    # This function must not mutate the input arrays.
    bit_array = bit_array.copy()

    num_bits = bit_array.shape[0]
    num_species = len(hamming_weights)

    partition_size = num_bits // num_species

    # Get the probability of flipping each bit based on the avg occupancy of each bit and the target hamming weight
    probs_table = np.zeros((num_species, partition_size))
    offset = 0
    for s in range(num_species):
        for i in range(partition_size):
            if bit_array[i + offset]:
                probs_table[s][i] = _p_flip_1_to_0(
                    hamming_weights[s] / partition_size, avg_occupancies[i + offset], 0.01
                )
            else:
                probs_table[s][i] = _p_flip_0_to_1(
                    hamming_weights[s] / partition_size, avg_occupancies[i + offset], 0.01
                )

        # Force probabilities to be beween 0 and 1, which can fail due to floating point precision
        probs_table[s] = np.minimum(1, np.maximum(0, probs_table[s]))
        offset += partition_size

    ######################## Handle bits starting with left then right ########################
    start_idx = 0
    end_idx = partition_size
    for s in range(num_species):
        if np.any(probs_table[s]):
            # Normalize
            probs_table[s] /= np.sum(probs_table[s])

            # Get difference between # of 1s and expected # of 1s
            n = np.sum(bit_array[start_idx:end_idx])
            n_diff = n - hamming_weights[s]

            # Too many electrons
            if n_diff > 0:
                indices_occupied = np.where(bit_array[start_idx:end_idx])[0]
                # Get the probabilities that each 1 should be flipped to 0
                p_choice = probs_table[s][bit_array[start_idx:end_idx]] / np.sum(
                    probs_table[s][bit_array[start_idx:end_idx]]
                )
                # Correct the hamming by probabilistically flipping some bits to flip to 0
                indices_to_flip = rng.choice(
                    indices_occupied, size=round(n_diff), replace=False, p=p_choice
                )
                bit_array[start_idx:end_idx][indices_to_flip] = False

            # too few electrons
            if n_diff < 0:
                indices_empty = np.where(np.logical_not(bit_array[start_idx:end_idx]))[0]
                # Get the probabilities that each 0 should be flipped to 1
                p_choice = probs_table[s][np.logical_not(bit_array[start_idx:end_idx])] / np.sum(
                    probs_table[s][np.logical_not(bit_array[start_idx:end_idx])]
                )
                # Correct the hamming by probabilistically flipping some bits to flip to 1
                indices_to_flip = rng.choice(
                    indices_empty, size=round(np.abs(n_diff)), replace=False, p=p_choice
                )
                bit_array[start_idx:end_idx][indices_to_flip] = np.logical_not(
                    bit_array[start_idx:end_idx][indices_to_flip]
                )

        start_idx += partition_size
        end_idx += partition_size

    return bit_array
