import pennylane as qml
from pennylane import numpy as np

dev_gaussian = qml.device("default.gaussian", wires=1)

@qml.qnode(dev_gaussian)
def mean_photon_gaussian(mag_alpha, phase_alpha, phi):
    qml.Displacement(mag_alpha, phase_alpha, wires=0)
    qml.Rotation(phi, wires=0)
    return qml.expval(qml.NumberOperator(0))

def cost(params):
    return (mean_photon_gaussian(params[0], params[1], params[2]) - 1.0) ** 2

init_params = np.array([0.5, 0.29, 0.015])
print(cost(init_params))

# initialise the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.1)

# set the initial parameter values
params = init_params

i = 0
while cost(params) != 0:
    # update the circuit parameters
    params = opt.step(lambda v: cost(v), params)
    print("Cost after step {:5d}: {:8f}".format(i + 1, cost(params)))
    print("Optimized mag_alpha:{:8f}".format(params[0]))
    print("Optimized phase_alpha:{:8f}".format(params[1]))
    print("Optimized phi:{:8f}".format(params[2]))
    i += 1

print("Optimized mag_alpha:{:8f}".format(params[0]))
print("Optimized phase_alpha:{:8f}".format(params[1]))
print("Optimized phi:{:8f}".format(params[2]))