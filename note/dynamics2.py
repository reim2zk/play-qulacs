# design for quantum dynamics library using qulacs
from typing import List, Tuple, Dict, Union, Optional
import qulacs
import matplotlib.pyplot as plt


class Hamiltonian(object):
    def __init__(self, n_qubits):
        self.t = 0.0
        self.n_qubits = n_qubits

    def time_evolution_circuit(self, delta: float) -> qulacs.ParametricQuantumCircuit:
        raise NotImplementedError()


class TIDHamiltonian(Hamiltonian):
    def __init__(self, n_qubits: int,
                 coefs: List[Tuple[float, List[Tuple[str, int]]]],
                 circuit: Optional[qulacs.ParametricQuantumCircuit] = None):
        super().__init__(n_qubits)
        self.coefs = coefs
        self.circuit = circuit

    def time_evolution_circuit(self, delta: float) -> qulacs.ParametricQuantumCircuit:
        if self.circuit is not None:
            return self.circuit

        self.circuit = qulacs.ParametricQuantumCircuit(self.n_qubits)
        circuit = self.circuit

        for (c, tis) in self.coefs:
            if len(tis) == 1:
                (t, i) = tis[0]
                if t == "X":
                    circuit.add_RX_gate(i, 2*delta*c)
                elif t == "Z":
                    circuit.add_RZ_gate(i, 2*delta*c)

            elif len(tis) == 2:
                (t0, i0) = tis[0]
                (t1, i1) = tis[1]
                if t0 != t1:
                    raise RuntimeError("t0!=t1 does not supported")
                circuit.add_CNOT_gate(i0, i1)
                if t0 == "X":
                    circuit.add_RX_gate(i1, 2*delta*c)
                elif t0 == "Z":
                    circuit.add_RZ_gate(i1, 2*delta*c)
                circuit.add_CNOT_gate(i0, i1)
            else:
                raise RuntimeError("Three bit interaction is not supported")
        return circuit


def ising_hamiltonian(n_qubits: int, j_ij: Dict[Tuple[int, int], float], h_i: List[float]):
    coefs = []
    for ((i, j), v) in j_ij.items():
        coefs.append((v, [("Z", i), ("Z", j)]))
        coefs.append((v, [("Z", j), ("Z", i)]))
    for (i, v) in enumerate(h_i):
        coefs.append((v, [("Z", i)]))
    return TIDHamiltonian(n_qubits, coefs)


def traverse_magnetic_hamiltonian(n_qubits: int):
    coefs = [(1.0, [("X", i)]) for i in range(n_qubits)]
    return TIDHamiltonian(n_qubits, coefs)


class LinearlyTDHamiltonian(Hamiltonian):
    """
    represent linear combination of time-independent Hamiltonian with time-dependent coefficient
        H(t) = \sum_k a_k(t) H_k
    """
    def __init__(self, ats, hs: List[TIDHamiltonian]):
        super().__init__(hs[0].n_qubits)
        self.ats = ats
        self.hs = hs
        self.n_comb = len(hs)

        self.circuit = qulacs.QuantumCircuit(self.n_qubits)
        self.num_paras_list = [0]
        for h in self.hs:
            circuit = h.time_evolution_circuit(1.0)
            for i in range(circuit.get_gate_count()):
                gate = circuit.get_gate(i)
                self.circuit.add_gate(gate)
                n = circuit.get_gate_count() + self.num_paras_list[-1]
                self.num_paras_list.append(n)
        self.angles = []
        for i in range(self.circuit.get_parameter_count()):
            self.angles.append(self.circuit.get_parameter(i))

    def time_evolution_circuit(self, delta: float, circuit=None):
        for i in range(self.n_comb):
            amp = self.ats[i](self.t) * delta
            n0 = self.num_paras_list[i]
            n1 = self.num_paras_list[i+1]
            for j in range(n0, n1):
                angle = self.angles[j] * amp
                self.circuit.set_parameter(j, angle)
        return self.circuit


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


#%% class (Ising + Traverse magnectic)
def run2():

    # Hamiltonian
    n_qubits = 6
    h = 3.0
    coefs = ([(1.0, [("Z", i), ("Z", (i+1) % n_qubits)]) for i in range(n_qubits)] +
             [(h, [("X", i)]) for i in range(n_qubits)])
    hamiltonian = TIDHamiltonian(n_qubits, coefs)

    # target observable
    z_obs = qulacs.Observable(n_qubits)
    for i in range(n_qubits):
        z_obs.add_operator(qulacs.PauliOperator("Z "+str(i), 1.0/n_qubits))

    # dynamics
    t = 3.0
    nt = 100
    dy = QuantumDynamics(hamiltonian, t, nt, z_obs)
    (t, y) = dy.run()

    # plot
    plt.plot(t, y)
    plt.show()


run2()


#%% Ising + Traverse magnetic
def run1():
    t = 3.0
    M = 100
    delta = t/M
    h = 3.0
    n_qubits = 6
    coefs = ([(1.0, [("Z", i), ("Z", (i+1) % n_qubits)]) for i in range(n_qubits)] +
             [(h, [("X", i)]) for i in range(n_qubits)])
    hamiltonian = Hamiltonian(n_qubits, coefs)
    circuit = hamiltonian.time_evolution_circuit(delta)

    # target observable
    z_obs = qulacs.Observable(n_qubits)
    for i in range(n_qubits):
        z_obs.add_operator(qulacs.PauliOperator("Z "+str(i), 1.0/n_qubits))

    # target observable
    z_obs = qulacs.Observable(n_qubits)
    for i in range(n_qubits):
        z_obs.add_operator(qulacs.PauliOperator("Z "+str(i), 1.0/n_qubits))

    # initial state
    state = qulacs.QuantumState(n_qubits)
    state.set_zero_state()

    # time evolution
    t = [i*delta for i in range(M+1)]
    y = list()
    y.append(z_obs.get_expectation_value(state))
    for i in range(M):
        circuit.update_quantum_state(state)
        y.append(z_obs.get_expectation_value(state))
    plt.plot(t, y)
    plt.show()


run1()