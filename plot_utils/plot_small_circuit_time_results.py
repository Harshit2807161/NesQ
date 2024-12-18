
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#2680		3520	3008	3048	2914	2794	2560	2537	3164

data = {"NMCS":1.833722047,			
        "MCTS (training on)":66.58165269,
        "MCTS (pretrained)":121.008243,
        "NMCS with opt passes":1.864338962
        }

agent = list(data.keys())
cumm_output_circ_depth = list(data.values())
 
fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(agent, cumm_output_circ_depth, color ='black', 
        width = 0.3)

plt.xlabel("Agent")
plt.ylabel("Cummulative runtime (in minutes)")
plt.title("Cummulative runtime on small circuit dataset")
plt.savefig("result_time_small_circuits.png")