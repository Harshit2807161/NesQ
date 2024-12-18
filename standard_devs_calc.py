
import pandas as pd
import numpy as np


df = pd.read_csv('results_small_circuits_opt.csv')

print(np.std(df["cd_N"]))



print(np.std(df["Cirq"]))

print(np.std(df["Qiskit_basic"]))


print(np.std(df["Qiskit_stochastic"]))


print(np.std(df["Qiskit_sabre"]))


print(np.std(df["PyTket"]))


print(np.std(df["cd_M_t"]))

print(np.std(df["cd_G"]))

print(np.std(df["cd_NO"]))
