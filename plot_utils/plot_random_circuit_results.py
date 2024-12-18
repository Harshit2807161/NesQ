
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = 'results_random_circuits_opt.csv'
data = pd.read_csv(file_path)

# Extract the number of gates, time, and the different circuit depths
gates = data['gates']
cd_N = data['cd_N']
Cirq = data['Cirq']
Qiskit_basic = data['Qiskit_basic']
Qiskit_stochastic = data['Qiskit_stochastic']
Qiskit_sabre = data['Qiskit_sabre']
PyTket = data['PyTket']
cd_M = data['cd_M']
cd_M_t = data['cd_M_t']
cd_G = data['cd_G']
cd_NO = data['cd_NO']

# Compute the mean and standard deviation for each algorithm
algorithms = {
    'NesQ': cd_N,
    'Cirq': Cirq,
    'Qiskit (basic)': Qiskit_basic,
    'Qiskit (stochastic)': Qiskit_stochastic,
    'Qiskit (sabre)': Qiskit_sabre,
    't|ket>': PyTket,
    #'MCTS aided GNN (pre-trained)': cd_M,
    'Qroute': cd_M_t,
    #'GNRPA (level 1)': cd_G,
    'NesQ+': cd_NO
}

# Plotting
plt.figure(figsize=(10, 6))

window_size = 5

for name, values in algorithms.items():
    moving_avg = values
    std_dev = np.std(moving_avg)
    plt.plot(gates, moving_avg, label=name)
    plt.fill_between(gates, moving_avg - 10, moving_avg + 10, alpha=0.1)

# Labels and title
plt.xlabel('No. of input gates',fontsize=15)
plt.ylabel('Average routed circuit depth for 10 runs',fontsize=15)
plt.legend()
plt.grid(True)
plt.savefig("random_circuit_depth_30_150_opt.png")
plt.show()
