import numpy as np
import mimiqcircuits as mc
from src.utils_vqe import get_basis_for_measurement


# import scipy
# import cirq
#
# from openfermion import expectation
# from openfermion.linalg import get_sparse_operator
# from src.utils_vqe import fix_nelec_in_state_vector


class VqeHardwareEfficient(object):
    def __init__(self, n_qubits, n_layers, n_electrons=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qubits = list(range(n_qubits))
        self.num_params = n_qubits * (n_layers + 1)
        self.final_state_vector_best = None
        self.best_vqe_params = None
        self.best_vqe_energy = None
        self.n_electrons = n_electrons

    def ansatz(self, params):
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        hw_eff_cirq = mc.Circuit()

        for q in range(n_qubits):
            hw_eff_cirq.add_gate(mc.GateRY(params[q, 0]), self.qubits[q])
            hw_eff_cirq.add_gate(mc.GateRZ(params[q, 0]), self.qubits[q])

        for i in range(n_layers):
            for q in range(n_qubits - 1):
                hw_eff_cirq.add_gate(mc.GateCX(), self.qubits[q], self.qubits[q + 1])

            for q in range(n_qubits):
                hw_eff_cirq.add_gate(mc.GateRY(params[q, i + 1]), self.qubits[q])
                hw_eff_cirq.add_gate(mc.GateRZ(params[q, i + 1]), self.qubits[q])

        return hw_eff_cirq

    def compute_energy_random_params(self, hamiltonian, nsamples=10000):
        conn = mc.MimiqConnection()

        # create a token - first time only
        # conn.savetoken()

        # load the saved token
        conn.loadtoken(filepath="/Users/vodola/try_qperfect/vqe/qperfect.json")
        params = 2 * np.pi * np.random.random_sample((self.n_qubits, self.n_layers + 1))
        estimate_energy = 0.0
        for ham_term in hamiltonian:
            [(pauli_operator, ham_coeff)] = ham_term.terms.items()
            circuit_with_measurement = self.ansatz(params)
            print("measuring the pauli string:", pauli_operator, "with coefficient:", ham_coeff)
            measurements = get_basis_for_measurement(pauli_operator)
            for measurement in measurements:
                if len(measurement):
                    basis_change, qubit_number = measurement
                    print(basis_change, qubit_number)
                    circuit_with_measurement.add_gate(basis_change, qubit_number)
            print(circuit_with_measurement)

            job = conn.execute(circuit_with_measurement,
                               label="vqe",
                               algorithm="mps",
                               nsamples=nsamples,
                               bonddim=10)
            res = conn.get_results(job, interval=1)
            print(res.samples)
            expec_value_single_pauli = 0.0
            for ket_measured, count_measured in res.samples.items():
                print(ket_measured, count_measured)
                print([(1 - 2 * bit) for bit in ket_measured])
                expec_value_single_pauli += ham_coeff * count_measured * np.product(
                    [(1 - 2 * bit) for bit in ket_measured]) / nsamples
            print(expec_value_single_pauli)
            estimate_energy += expec_value_single_pauli

            conn.close()
            return estimate_energy
