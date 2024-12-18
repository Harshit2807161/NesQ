import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the provided CSV file
df = pd.read_csv("results_time_data_largedata_new_opt.csv")

# Data provided by the user
data = {
    "circuit": df["circuit_name"].tolist(),
    "t_N": df["t_N"].tolist(),
    "t_Cirq": df["t_Cirq"].tolist(),
    "t_Qiskit_basic": df["t_Qiskit_basic"].tolist(),
    "t_Qiskit_stochastic": df["t_Qiskit_stochastic"].tolist(),
    "t_Qiskit_sabre": df["t_Qiskit_sabre"].tolist(),
    "t_PyTket": df["t_PyTket"].tolist(),
    "t_M": df["t_M"].tolist(),
    "t_M_t": df["t_M_t"].tolist(),
    "t_G": df["t_G"].tolist(),
    "t_NO": df["t_NO"].tolist(),
    "gates_count": df["gates"].tolist()
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting the data
fig, ax = plt.subplots(figsize=(14, 8))

bar_width = 0.1
index = range(len(df))

bars0 = ax.bar([i - bar_width for i in index], df['t_N'], bar_width, label='NesQ')
#bars1 = ax.bar([i - bar_width for i in index], df['t_Cirq'], bar_width, label='Cirq')
#bars2 = ax.bar([i - bar_width for i in index], df['t_Qiskit_basic'], bar_width, label='Qiskit (basic)')
#bars3 = ax.bar(index, df['t_Qiskit_stochastic'], bar_width, label='Qiskit (stochastic)')
#bars4 = ax.bar([i + bar_width for i in index], df['t_Qiskit_sabre'], bar_width, label='Qiskit (sabre)')
#bars5 = ax.bar([i + 2*bar_width for i in index], df['t_PyTket'], bar_width, label='PyTket')
#bars2 = ax.bar(index, df['t_M'], bar_width, label='GNN aided MCTS (pretrained)')
bars4 = ax.bar(index, df['t_NO'], bar_width, label='NesQ+')
bars3 = ax.bar([i + bar_width for i in index], df['t_M_t'], bar_width, label='Qroute')


ax.set_xlabel('Number of Gates in Circuit',fontsize=15)
ax.set_ylabel('Time taken to route (in minutes)',fontsize=15)
#ax.set_title('Time taken to route vs Number of Gates in Circuit')
ax.set_xticks(index)
ax.set_xticklabels(df['gates_count'])
ax.legend()
plt.savefig("result_time_large_circuits.png")
plt.show()
