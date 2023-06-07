import numpy as np
from pyscf import ao2mo
from functools import reduce

import mimiqcircuits as mc


def normal_ordering_swap(orbitals):
    """
    This function normal calculates the phase coefficient (-1)^swaps, where swaps is the number of
    swaps required to normal order a given list of numbers.
    :param orbitals: list of numbers, e.g. orbitals
    :returns: number of required swaps
    """
    count_swaps = 0
    for i in range(len(orbitals)):
        for j in range(i + 1, len(orbitals)):
            if orbitals[i] > orbitals[j]:
                count_swaps += 1

    return count_swaps


def fix_nelec_in_state_vector(final_state_vector, nelec):
    """
    Projects the wave function final_state_vector in the subspace with the fix number of electrons given by nelec
    :param final_state_vector Cirq object representing the state vector from a VQE simulation
    :param nelec (tuple) with n_alpha, n_beta number of electrons
    return: state vector (correctly normalized) with fixed number of electrons
    """
    n_alpha, n_beta = nelec
    n_qubits = int(np.log2(len(final_state_vector)))
    projected_vector = np.array(final_state_vector)
    for decimal_ket, coeff in enumerate(final_state_vector):
        string_ket = bin(decimal_ket)[2:].zfill(n_qubits)
        string_alpha = string_ket[::2]  # alpha orbitals occupy the even positions
        string_beta = string_ket[1::2]  # beta orbitals occupy the odd positions
        alpha_occ = [pos for pos, char in enumerate(string_alpha) if char == '1']
        beta_occ = [pos for pos, char in enumerate(string_beta) if char == '1']
        if (len(alpha_occ) != n_alpha) or (len(beta_occ) != n_beta):
            projected_vector[decimal_ket] = 0.0

    normalization = np.sqrt(np.dot(projected_vector.conj(), projected_vector))
    return projected_vector / normalization


def compute_integrals(pyscf_molecule, pyscf_scf):
    """
    Compute the 1-electron and 2-electron integrals.

    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.

    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    # Get one electrons integrals.
    n_orbitals = pyscf_scf.mo_coeff.shape[1]
    one_electron_compressed = reduce(np.dot, (pyscf_scf.mo_coeff.T,
                                              pyscf_scf.get_hcore(),
                                              pyscf_scf.mo_coeff))

    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    two_electron_compressed = ao2mo.kernel(pyscf_molecule,
                                           pyscf_scf.mo_coeff)

    two_electron_integrals = ao2mo.restore(
        1,  # no permutation symmetry
        two_electron_compressed, n_orbitals)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = np.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')

    # Return.
    return one_electron_integrals, two_electron_integrals


def from_string_to_spin(pauli_string, qubit):
    if pauli_string.lower() == 'x':
        return [mc.GateH(), qubit]
    elif pauli_string.lower() == 'y':
        return [mc.GateRX(np.pi / 2), qubit]
    else:
        return []


def get_basis_for_measurement(pauli_operator):
    """ Returns the basis rotation for measuring jw_hamiltonian_term: e.g. if
    jw_hamiltonian_term = Z_0 X_1 Y_2
    returns [[mc.GateH(), 1], [mc.GateRX(np.pi/2), 2]]
    """
    if len(pauli_operator):
        measurement_basis = []
        for qubit_index, pauli_op in pauli_operator:
            measurement_basis.append(from_string_to_spin(pauli_op, qubit_index))
    else:
        measurement_basis = [[]]
    return measurement_basis
