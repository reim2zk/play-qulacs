import numpy as np
import qulacs
import matplotlib.pyplot as plt

#%% plot test
xs = np.linspace(0, 1)
ys = np.sin(np.pi*xs)
plt.plot(xs, ys)
plt.show()


#%% define
# state
n = 2
zero_index = 0
psi_index_list = list(range(1, n))
state0 = qulacs.QuantumState(n)
state0.set_computational_basis(0b011)
state1 = state0.copy()

# define Adamal test.
# Hamiltonian = H x I + \Lambda(U) + H x I
# \Lambda(U) = |0><0| x I + |1><1| x U
circuit = qulacs.QuantumCircuit(n)
lambda_U = qulacs.gate.to_matrix_gate(qulacs.gate.RZ(1, np.pi/2))
lambda_U.add_control_qubit(0, 1)

circuit.add_H_gate(0)
circuit.add_gate(lambda_U)
circuit.add_H_gate(0)

circuit.update_quantum_state(state0)
print("<\psi|U|\psi>", qulacs.state.inner_product(state1, state0))

