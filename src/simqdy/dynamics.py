from simqdy.hamiltonian import Hamiltonian
import qulacs


class QuantumDynamics(object):
    def __init__(self, hamiltonian: Hamiltonian, t: float, nt: int, target_obs):
        self.hamiltonian = hamiltonian
        self.t = t
        self.nt = nt
        self.delta = self.t/self.nt
        self.target_obs = target_obs
        self.state = qulacs.QuantumState(hamiltonian.n_qubits)

    def state(self):
        return self.state

    def calc_target(self):
        return self.target_obs.get_expectation_value(self.state)

    def run(self):
        t = [i*self.delta for i in range(self.nt+1)]
        y = list()
        y.append(self.calc_target())
        for i in range(self.nt):
            self.hamiltonian.t = t[i]
            circuit = self.hamiltonian.time_evolution_circuit(self.delta)
            circuit.update_quantum_state(self.state)
            y.append(self.calc_target())
        return t, y
