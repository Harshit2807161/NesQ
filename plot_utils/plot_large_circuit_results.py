import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the provided CSV file
df = pd.read_csv("results_largedata_new_opt.csv")

#df = df[:-1]

# Data provided by the user
data = {
    "circuit": df["circuit_name"].tolist(),
    "cd_N": df["cd_N"].tolist(),
    "Cirq": df["Cirq"].tolist(),
    "Qiskit_basic": df["Qiskit_basic"].tolist(),
    "Qiskit_stochastic": df["Qiskit_stochastic"].tolist(),
    "Qiskit_sabre": df["Qiskit_sabre"].tolist(),
    "PyTket": df["PyTket"].tolist(),
    "cd_M": df["cd_M"].tolist(),
    "cd_M_t": df["cd_M_t"].tolist(),
    #"cd_G": df["cd_G"].tolist(),
    "cd_NO": df["cd_NO"].tolist(),
    "gates_count": df["gates_count"].tolist()
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting the data
fig, ax = plt.subplots(figsize=(14, 8))

bar_width = 0.1
index = range(len(df))

bars8 = ax.bar([i - 4*bar_width for i in index], df['cd_NO'], bar_width, label='NesQ+')
bars0 = ax.bar([i - 3*bar_width for i in index], df['cd_N'], bar_width, label='NesQ')
bars1 = ax.bar([i - 2*bar_width for i in index], df['Cirq'], bar_width, label='Cirq')
bars2 = ax.bar([i - bar_width for i in index], df['Qiskit_basic'], bar_width, label='Qiskit (basic)')
bars3 = ax.bar(index, df['Qiskit_stochastic'], bar_width, label='Qiskit (stochastic)')
bars4 = ax.bar([i + bar_width for i in index], df['Qiskit_sabre'], bar_width, label='Qiskit (sabre)')
bars5 = ax.bar([i + 2*bar_width for i in index], df['PyTket'], bar_width, label='PyTket')
#bars6 = ax.bar([i + 3*bar_width for i in index], df['cd_M'], bar_width, label='MCTS (pretrained)')
bars7 = ax.bar([i + 3*bar_width for i in index], df['cd_M_t'], bar_width, label='Qroute')




ax.set_xlabel('Number of Gates in Circuit',fontsize=15)
ax.set_ylabel('Depth of Output Circuit',fontsize=15)
#ax.set_title('Depth of Output Circuit vs Number of Gates in Circuit')
ax.set_xticks(index)
ax.set_xticklabels(df['gates_count'])
ax.legend()
plt.savefig("result_large_circuit.png")
plt.show()