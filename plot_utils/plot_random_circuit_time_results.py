

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = 'results_random_circuits_time_opt.csv'
data = pd.read_csv(file_path)

# Extract the number of gates, time, and the different circuit depths
gates = data['gates']
cd_N = data['t_N']
Cirq = data['t_Cirq']
Qiskit_basic = data['t_Qiskit_basic']
Qiskit_stochastic = data['t_Qiskit_stochastic']
Qiskit_sabre = data['t_Qiskit_sabre']
PyTket = data['t_PyTket']
cd_M = data['t_M']
cd_M_t = data['t_M_t']
cd_G = data['t_G']
cd_NO = data['t_NO']

# Compute the mean and standard deviation for each algorithm
algorithms = {
    'NesQ': cd_N,
    #'Cirq': Cirq,
    #'Qiskit (basic)': Qiskit_basic,
    #'Qiskit (stochastic)': Qiskit_stochastic,
    #'Qiskit (sabre)': Qiskit_sabre,
    #'t|ket>': PyTket,
    #'Qroute (pre-trained)': cd_M,
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

# Labels and title
plt.xlabel('No. of input gates', fontsize=15)
plt.ylabel('Cummulative runtime for 10 runs (in seconds)', fontsize=15)
plt.legend()
plt.grid(True)
plt.savefig("random_circuit_runtime_30_150_opt.png")
plt.show()



