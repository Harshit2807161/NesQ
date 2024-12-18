
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data for cumulative routed circuit depths
data = {
    "NesQ": 2690,		
    "Cirq": 3616,	
    "Qiskit_basic": 3008,	
    "Qiskit_stochastic": 3016,	
    "Qiskit_sabre": 2908,
    "PyTket": 2794,
    "Qroute": 2583,
    #"GNRPA": 3324,
    "NesQ+": 2447
}

# Standard deviation values (example values, replace with actual data)
std_dev = {
    "NesQ": 26.51,		
    "Cirq": 36.34,	
    "Qiskit_basic": 30.94,	
    "Qiskit_stochastic": 30.25,	
    "Qiskit_sabre": 30.90,
    "PyTket": 30.50,
    "Qroute": 26.08,
    #"GNRPA": 33.83,
    "NesQ+": 25.54
}

agent = list(data.keys())
cumm_output_circ_depth = list(data.values())
errors = list(std_dev.values())

fig = plt.figure(figsize=(15, 8))

# Creating the bar plot with error bars
plt.bar(agent, cumm_output_circ_depth, color='blue', width=0.3, yerr=errors, capsize=8)

plt.xlabel("Agent", fontsize=15)
plt.ylabel("Cummulative routed circuit depth", fontsize=15)
#plt.title("Cummulative routed circuit depths on small circuit data", fontsize=15)

# Save the figure
plt.savefig("result_small_circuit_opt_with_errors.png")

# Show the plot
plt.show()
