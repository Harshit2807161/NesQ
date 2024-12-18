import os

import numpy as np
import tqdm
import torch
import wandb
import time
import cirq
from .metas import CombinerAgent
from .environment.env import step
from .environment.circuits import CircuitRepDQN, circuit_to_json
from .environment.device import DeviceTopology
from .environment.state import CircuitStateDQN
from .visualizers.solution_validator import validate_solution
import matplotlib.pyplot as plt 
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CXCancellation, CommutativeCancellation, RemoveBarriers, Depth

def cirq_to_qiskit(cirq_circuit):
    qiskit_circuit = QuantumCircuit(len(cirq_circuit.all_qubits()))
    qubit_map = {qubit: idx for idx, qubit in enumerate(sorted(cirq_circuit.all_qubits()))}

    for moment in cirq_circuit:
        for op in moment.operations:
            qubits = [qubit_map[q] for q in op.qubits]
            if isinstance(op.gate, cirq.ops.HPowGate):
                qiskit_circuit.h(qubits[0])
            elif isinstance(op.gate, cirq.ops.CNotPowGate):
                qiskit_circuit.cx(qubits[0], qubits[1])
            elif isinstance(op.gate, cirq.ops.SwapPowGate):
                qiskit_circuit.swap(qubits[0], qubits[1])
            else:
                raise ValueError(f"Unsupported gate: {op.gate}")

    return qiskit_circuit



def plot_stepwise_reward(best_rewards, agent_type, levelN, levelG, numitersG, search_depth):
    """
    Plot the frequency of unique reward values and save the plot.
    
    Parameters:
        best_rewards (list): List of rewards obtained at each step.
        agent_type (str): Type of agent used ("N" for NMCS, "M" for MCTS, "G" for GNRPA).
        levelN (int): Level for NMCS.
        levelG (int): Level for GNRPA.
        numitersG (int): Number of iterations for GNRPA.
        search_depth (int): Search depth for MCTS.
    """
    # Plot individual step rewards
    plt.figure(figsize=(14, 8))
    print(best_rewards)
    plt.plot(best_rewards, color='b', alpha=0.7)
    plt.xlabel('Solving Step no.')
    plt.ylabel('Score of step')
    if agent_type == "N":
        t = "NMCS"
        title = f'Stepwise score of NMCS - level {levelN}'
        cumm_title = f'Cummulative score of NMCS - level {levelN}'
        save_name = f'stepwise_score_plot_{t}_{levelN}.png'
        cumm_save_name = f'cummulative_score_plot_{t}_{levelN}.png'
    elif agent_type == "M":
        t = "MCTS"
        title = f'Stepwise score of MCTS - SD {search_depth}'
        cumm_title = f'Cummulative score of MCTS - SD {search_depth}'
        save_name = f'stepwise_score_plot_{t}_{search_depth}.png'
        cumm_save_name = f'cummulative_score_plot_{t}_{search_depth}.png'
    else:
        t = "GNRPA"
        title = f'Stepwise score of GNRPA - level {levelG}, iters {numitersG}'
        cumm_title = f'Cummulative score of GNRPA - level {levelG}, iters {numitersG}'
        save_name = f'stepwise_score_plot_{t}_{levelG}_{numitersG}.png'
        cumm_save_name = f'cummulative_score_plot_{t}_{levelG}_{numitersG}.png'
    plt.title(title)
    
    # Define the save directory and create it if it doesn't exist
    save_dir = "enter_your_save_directory_here"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Define the save path
    save_path = os.path.join(save_dir, save_name)
    
    # Save the plot
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Show the plot
    plt.close()
    plt.clf()

    # Plot cumulative rewards
    cumulative_rewards = np.cumsum(best_rewards)
    plt.figure(figsize=(14, 8))
    plt.plot(cumulative_rewards, color='b', alpha=0.7)
    plt.xlabel('Solving Step no.')
    plt.ylabel('Cumulative score till step')
    plt.title(cumm_title)
    
    # Define the save path for cumulative plot
    cumulative_save_path = os.path.join(save_dir, cumm_save_name)
    
    # Save the cumulative plot
    plt.savefig(cumulative_save_path)
    print(f"Cumulative plot saved to {cumulative_save_path}")
    
    # Show the cumulative plot
    plt.close()
    plt.clf()


def plot_reward(rewards,agent_type,levelN,levelG,numitersG,search_depth):
    """
    Plots the frequency of reward values with reward value on x-axis and frequency on y-axis.

    Parameters:
    rewards (list or array-like): Array of reward values to plot.
    agent_type (str): Type of agent ('N' for NMCS, 'M' for MCTS, etc.)
    """
    # Compute the frequency of each unique reward value
    unique_rewards, counts = np.unique(rewards, return_counts=True)
    
    # Determine the agent type title
    if agent_type == "N":
        t = "NMCS"
        save_name = f"allplayouts_frequency_plot_{t}_{levelN}.png"
        title_name = f'Solving circuit via {t} - level {levelN}'
    elif agent_type == "M":
        t = "MCTS"
        save_name = f"allplayouts_frequency_plot_{t}_{search_depth}.png"
        title_name = f'Solving circuit via {t} - SD {search_depth}'
    else:
        t = "GNRPA"
        save_name = f"allplayouts_frequency_plot_{t}_{levelG}_{numitersG}.png"
        title_name = f'Solving circuit via {t} - level {levelG}, iters {numitersG}'
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    plt.bar(unique_rewards, counts, color='b', alpha=0.7, width = 17)
    plt.xlabel('Playout Score Value')
    plt.ylabel(f'Frequency of all {len(rewards)} playout scores')
    plt.title(title_name)
    #plt.xscale('log')  # Use a logarithmic scale for the y-axis if needed

    # Define the save directory and create it if it doesn't exist
    save_dir = "enter_your_save_directory_here"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Define the save path
    save_path = os.path.join(save_dir, save_name)
    
    # Save the plot
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Show the plot
    plt.close()
    plt.clf()


def train_step(agent: CombinerAgent,
               device: DeviceTopology,
               circuit: CircuitRepDQN,  
               rewards,
               best_rewards,
               agent_type,
               levelN,
               levelG,
               numitersG,
               search_depth,
               training_steps=100000, episode_name="Unnamed Run",
               use_wandb=False, train_model=True):

    os.makedirs("./test/test_results", exist_ok=True)
    input_circuit = circuit
    state = CircuitStateDQN(input_circuit, device)
    solution_start, solution_moments = np.array(state.node_to_qubit), []
    progress_bar = tqdm.tqdm(total=len(list(circuit.cirq.all_operations())))
    state, total_reward, done, debugging_output = step(np.full(len(state.device.edges), False), state)
    progress_bar.update(len(debugging_output.cnots))
    solution_moments.append(debugging_output)
    if done:
        print("Episode %03d: The initial circuit is executable with no additional swaps" % episode_name)
        return
    progress_bar.set_description(episode_name)
    start_time = time.time()
    
    for times in range(2, training_steps + 1):
        if time.time()-start_time > 18000:
            print(f"\nOutput circuit depth for {agent_type}: 1000000000000")
            return 1000000000000
        if agent_type == "NQ":
            dep = agent.act(state,rewards)
            print(f"\nOutput circuit depth for {agent_type}: ",dep)
            return dep
        action = agent.act(state,rewards)
        assert not np.any(np.bitwise_and(state.locked_edges, action)), "Bad Action"
        next_state, reward, done, debugging_output = step(action, state)
        total_reward += reward
        best_rewards.append(reward)
        #print(f"Reward for step {time-1} is {reward}")
        solution_moments.append(debugging_output)
        progress_bar.update(len(debugging_output.cnots))
        state = next_state

        if train_model and (times + 1) % 1000 == 0:
            loss_v, loss_p = agent.replay()
            if use_wandb:
                wandb.log({'Value Loss': loss_v, 'Policy Loss': loss_p})
            torch.save(agent.model.state_dict(), f"{device.name}-weights.h5")
        
        progress_bar.set_postfix(total_reward=total_reward, time=times)
        if done:
            #plot_reward(rewards,agent_type,levelN,levelG,numitersG,search_depth)
            #plot_stepwise_reward(best_rewards,agent_type,levelN,levelG,numitersG,search_depth)
            result_circuit = validate_solution(input_circuit, solution_moments, solution_start, device)
            circuit_to_json(result_circuit, ("./test/test_results/%s.json" % episode_name))
            depth = len(result_circuit.moments)
            if agent_type=="NO":
                # Define the pass manager and add optimization passes
                pass_manager = PassManager()
                pass_manager.append(Optimize1qGates())
                #pass_manager.append(CXCancellation())
                pass_manager.append(CommutativeCancellation())
                #pass_manager.append(RemoveBarriers())

                # Run the optimization passes on the routed circuit
                optimized_circuit = pass_manager.run(cirq_to_qiskit(result_circuit))
                optimized_depth = optimized_circuit.depth()
                print(f"\nOutput circuit depth for {agent_type}: ",optimized_depth)
            else:
                if train_model:
                    print(f"\nOutput circuit depth for {agent_type} with Training: ",depth)
                else:
                    print(f"\nOutput circuit depth for {agent_type}: ",depth)

            progress_bar.set_postfix(circuit_depth=depth, total_reward=total_reward, time=times)
            progress_bar.close()
            # print(solution_start, "\n", input_circuit.cirq, "\n", result_circuit, "\n", flush=True)
            if train_model:
                loss_v, loss_p = agent.replay()
                if use_wandb:
                    wandb.log({'Value Loss': loss_v, 'Policy Loss': loss_p})
                torch.save(agent.model.state_dict(), f"{device.name}-weights.h5")
            if use_wandb:
                wandb.log({'Circuit Depth': depth,
                           'Circuit Name': episode_name,
                           'Steps Taken': times})
            return solution_start, solution_moments, True

    if train_model:
        loss_v, loss_p = agent.replay()
        if use_wandb:
            wandb.log({'Value Loss': loss_v, 'Policy Loss': loss_p})
        torch.save(agent.model.state_dict(), f"{device.name}-weights.h5")

    return solution_start, solution_moments, False
