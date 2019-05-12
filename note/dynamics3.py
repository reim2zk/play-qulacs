import qulacs
import matplotlib.pyplot as plt
import sys

sys.path.append("src")
import simqdy as sq


#%% Ising + Tracerse
def run1():
    n_qubits = 6
    h = 3.0
    coefs = ([(1.0, [("Z", i), ("Z", (i + 1) % n_qubits)]) for i in range(n_qubits)] +
             [(h, [("X", i)]) for i in range(n_qubits)])
    hamiltonian = sq.TIDHamiltonian(n_qubits, coefs)

    # target observable
    z_obs = qulacs.Observable(n_qubits)
    for i in range(n_qubits):
        z_obs.add_operator(qulacs.PauliOperator("Z " + str(i), 1.0 / n_qubits))

    # dynamics
    t = 3.0
    nt = 100
    dy = sq.QuantumDynamics(hamiltonian, t, nt, z_obs)
    (t, y) = dy.run()

    # plot
    plt.plot(t, y)
    plt.show()


run1()


#%% Annealing machine
def run2():
    t1 = 3.0
    n_qubits = 6
    h = 3.0
    j_ij = {(i, i % n_qubits): 1.0 for i in range(n_qubits)}
    h_i = [h for _ in range(n_qubits)]
    ats = [lambda t: t, lambda t: t1-t]
    hs = [sq.ising_hamiltonian(n_qubits, j_ij, h_i), sq.traverse_magnetic_hamiltonian(n_qubits)]
    hamiltonian = sq.LinearlyTDHamiltonian(ats, hs)

    # target observable
    z_obs = qulacs.Observable(n_qubits)
    for i in range(n_qubits):
        z_obs.add_operator(qulacs.PauliOperator("Z " + str(i), 1.0 / n_qubits))

    # dynamics
    nt = 100
    dy = sq.QuantumDynamics(hamiltonian, t1, nt, z_obs)
    (ts, ys) = dy.run()

    # plot
    plt.plot(ts, ys)
    plt.show()


run2()
