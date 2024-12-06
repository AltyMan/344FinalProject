import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np

NUM_QUBITS = 3
dev = qml.device("default.qubit", wires=NUM_QUBITS)
wires = list(range(NUM_QUBITS))

def equal_superposition(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)

def oracle(wires, omega):
    qml.FlipSign(omega, wires=wires)

def diffusion_operator(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)
        qml.PauliX(wires=wire)
    qml.ctrl(qml.PauliZ, 0)(wires=1)
    for wire in wires:
        qml.PauliX(wires=wire)
        qml.Hadamard(wires=wire)

@qml.qnode(dev)
def grover_circuit():
    equal_superposition(wires)
    qml.Snapshot("Uniform superposition |s>")

    omega = np.array([1, 1, 1])  # Marked state |111>
    oracle(wires, omega)
    qml.Snapshot("State marked by Oracle")
    diffusion_operator(wires)
    qml.Snapshot("Amplitude after diffusion")

    return qml.probs(wires=wires)

results = qml.snapshots(grover_circuit)()

for k, result in results.items():
    print(f"{k}: {result}")

y = np.real(results["Amplitude after diffusion"])
bit_strings = [f"{x:0{NUM_QUBITS}b}" for x in range(len(y))]

plt.bar(bit_strings, y, color="#70CEFF")

plt.xticks(rotation="vertical")
plt.xlabel("State label")
plt.ylabel("Probability Amplitude")
plt.title("States probabilities amplitudes")
plt.show()