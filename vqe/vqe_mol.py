import numpy as np
import pickle
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

from src.vqe import VqeHardwareEfficient

# from pyscf.scf.chkfile import dump_scf

if __name__ == "__main__":
    np.set_printoptions(precision=6,
                        suppress=True,
                        linewidth=10000)

    np.random.seed(12)
    system_label = "medium"  # "big"
    multiplicity = 1
    charge = 0
    spin = 0
    if system_label == "small":
        basis = "sto-3g"
        geometry = [('H', (0, 0, 0)),
                    ('H', (0, 0, 1.23))]

    elif system_label == "medium":
        basis = "sto-3g"
        geometry = [('O', (-3.1866592, 3.8969550, 0)),
                    ('O', (-2.1048336, 3.2644638, 0)),
                    ('O', (-1.0356472, 3.9180912, 0))]

    elif system_label == "big":
        basis = "cc-pvdz"
        geometry = [('O', (-3.1866592, 3.8969550, 0)),
                    ('O', (-2.1048336, 3.2644638, 0)),
                    ('O', (-1.0356472, 3.9180912, 0))]
    else:
        print("Choose system_label: small, medium, big")
        exit()

    molecule = MolecularData(geometry, basis, multiplicity, charge)

    molecule = run_pyscf(molecule, run_scf=True, run_fci=False)
    mf = molecule._pyscf_data['scf']
    mol = molecule._pyscf_data['mol']
    noccas, noccbs = mol.nelec
    # dump_scf(mol, "new.chk", mf.e_tot, mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    print(f"Basis set dimension: {len(mf.mo_occ)}")
    print(f"SCF energy: {molecule.hf_energy}")
    # print(f"FCI energy: {molecule.fci_energy}")
    n_qubit = molecule.n_qubits
    n_electron = molecule.n_electrons
    print("creating fermionic Hamiltonians - this can require time")
    fermionic_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
    try:
        with open(f'{system_label}.pickle', 'rb') as handle:
            jw_hamiltonian = pickle.load(handle)
    except:
        jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)
        with open(f'{system_label}.pickle', 'wb') as handle:
            pickle.dump(jw_hamiltonian, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("starting VQE")
    vqe = VqeHardwareEfficient(n_qubits=n_qubit, n_layers=1, n_electrons=mol.nelec)
    random_energy = vqe.compute_energy_random_params(jw_hamiltonian)
    print(f"Energy with random VQE params: {random_energy}")
