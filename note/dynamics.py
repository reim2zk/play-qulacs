# cf: http://dojo.qulacs.org/ja/latest/notebooks/4.2_trotter_decomposition.html
import numpy as np
import qulacs
import matplotlib.pyplot as plt

#%% Ising + Traverse magnetise
# setting
n_qubits = 6
t = 3.0
M = 100
delta = t/M
h = 3.0

# observable
z_obs = qulacs.Observable(n_qubits)
for i in range(n_qubits):
    z_obs.add_operator(qulacs.PauliOperator("Z "+str(i), 1.0/n_qubits))

# initial state
state = qulacs.QuantumState(n_qubits)
state.set_zero_state()

# quantum circuit
circuit = qulacs.QuantumCircuit(n_qubits)
for i in range(n_qubits):
    circuit.add_CNOT_gate(i, (i+1) % n_qubits)
    circuit.add_RZ_gate((i+1) % n_qubits, 2*delta)
    circuit.add_CNOT_gate(i, (i+1) % n_qubits)
    circuit.add_RX_gate(i, 2*h*delta)

# time evolution
t = [i*delta for i in range(M+1)]
y = list()
y.append(z_obs.get_expectation_value(state))
for i in range(M):
    circuit.update_quantum_state(state)
    y.append(z_obs.get_expectation_value(state))

#%% plot
plt.plot(t, y)
plt.show()