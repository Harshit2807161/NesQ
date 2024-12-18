import os
import logging
import argparse

import wandb
import torch
import numpy as np
from .environment.device import IBMqx20TokyoDevice, GridComputerDevice, GoogleSycamore, Rigetti19QAcorn, IBMqx5Device
from .environment.circuits import circuit_from_qasm, CircuitRepDQN, \
    circuit_generated_randomly, circuit_generated_full_layer
from .algorithms.nmcs_agent import NMCAgent
#from .algorithms.nmcs_agent_old import NMCAgentold
from .algorithms.deepmcts import MCTSAgent
from .algorithms.GNRPA_agent import GNRPAagent
from .models.graph_dual import GraphDualModel
from .memory.list import MemorySimple
from .engine import train_step
import qiskit, qiskit.transpiler.exceptions
from .visualizers.greedy_schedulers import cirq_routing, qiskit_routing, tket_routing, qiskit_routing_stochastic
import time
import matplotlib.pyplot as plt 
logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', default="small",
                        help='Choose training and test dataset from small, large, full, random')
    parser.add_argument('--gates', default=100, type=int,
                        help='Size of circuit if not from a file dataset')
    parser.add_argument('--hardware', default="qx20",
                        help='Device to run on, eg. qx20, grid/6, grid/4, etc.')
    parser.add_argument('--iterations', default=10, type=int,
                        help='Number of iterations to train for on generated circuits.')
    parser.add_argument('--train', action='store_const', default=False, const=True,
                        help='Whether the training loop should be run or just evaluation.')
    parser.add_argument('--wandb', action='store_const', default=False, const=True,
                        help='Whether to use WandB to log the results of experiments.')
    parser.add_argument('--search', default=200, type=int,
                        help='Number of iterations to search for before making a move.')
    parser.add_argument('--levelG', default=1, type=int,
                        help='Depth of GNRPA search for making a move.')
    parser.add_argument('--levelN', default=1, type=int,
                        help='Depth of NMCS search for making a move.')
    parser.add_argument('--agent', default="N", type=str,
                        help='Which agent to use for the search.')
    parser.add_argument('--numitersG', default=100, type=int,
                        help='Number of iterations in GNRPA')
    parser.add_argument('--large_files', default="rd84_142", type=str,
                        help='Enter large circuit file name if needed')
    parser.add_argument('--small_file', default="3_17_13", type=str,
                        help='Enter small circuit file name if needed')
    args = parser.parse_args()

    # Get the right environment up
    device = None
    if args.hardware == "qx20":
        device = IBMqx20TokyoDevice()
    elif "grid" in args.hardware:
        device = GridComputerDevice(int(args.hardware.split("/")[-1]))
    elif args.hardware == "sycamore":
        device = GoogleSycamore()
    elif args.hardware == "acorn":
        device = Rigetti19QAcorn()
    elif args.hardware == "qx5":
        device = IBMqx5Device()
    else:
        raise ValueError(f"{args.hardware} is not a valid device.")

    # Get the agent up and ready
    model = GraphDualModel(device, True)
    '''
    if os.path.exists(f"results/{device.name}-weights.h5"):        
        model.load_state_dict(torch.load(f"results/{device.name}-weights.h5"))
    '''
    memory = MemorySimple(0)
    if args.agent=="N":
        agent = NMCAgent(model, device, memory,levelN=args.levelN)
    elif args.agent=="M":
        agent = MCTSAgent(model,device,memory,search_depth=args.search)
    elif args.agent=="NO":
        agent = NMCAgent(model, device, memory,levelN=args.levelN)
    else:
        agent = GNRPAagent(model,device,memory,levelG=args.levelG,numitersG = args.numitersG)
    # Other preferences
    '''
    # Run different benchmarks
    if args.dataset == "small":
        for e, file in enumerate(list(filter(lambda x: '_onlyCX' in x, 
                                             os.listdir("./test/circuit_qasm")))):
            cirq = circuit_from_qasm(
                os.path.join("./test/circuit_qasm", file))
            if len(list(cirq.all_operations())) > 100:
                continue
            rewards = []
            best_rewards = []
            circuit = CircuitRepDQN(cirq, len(device))
            train_step(agent, device, circuit, rewards, best_rewards, args.agent, args.levelN, args.levelG, args.numitersG, args.search, episode_name=file, use_wandb=args.wandb, train_model=args.train)
            #([0, 19, 28, 29, 31], [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, True, False, True, False, False, False, False, False, False, False, False, False, False, False])
            print("Layers in input circuit: ", len(cirq.moments))
            print("Cirq Routing Distance: ", cirq_routing(circuit, device))
            print("Qiskit Routing Distance: ", qiskit_routing(circuit, device))
            print("PyTket Routing Distance: ", tket_routing(circuit, device))  
    '''

    if args.dataset == "small":
        i = 0 
        file = args.small_file
        i += 1
        cirq = circuit_from_qasm(
            os.path.join("./test/circuit_qasm", file + "_onlyCX.qasm"))
        rewards = []
        best_rewards = []
        circuit = CircuitRepDQN(cirq, len(device))
        print("Gates in input circuit: ",sum(1 for _ in cirq.all_operations()))
        start_time = time.time()
        curr_time = time.time() - start_time
        times = []
        depths = []
        print("Layers in input circuit: ", len(cirq.moments))
        agentN = NMCAgent(model, device, memory,levelN=args.levelN)
        agentM = MCTSAgent(model,device,memory,search_depth=args.search)
        agentG = GNRPAagent(model,device,memory,levelG=args.levelG,numitersG = args.numitersG)
        rewards = []
        best_rewards = []
        start_time = time.time()
        train_step(agentN, device, circuit, rewards, best_rewards, 'N', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb, train_model=args.train)
        time_N = time.time()-start_time
        print("Time N: ",time_N)
        
        start_time = time.time()
        train_step(agentN, device, circuit, rewards, best_rewards, 'NO', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb, train_model=args.train)
        time_NO = time.time()-start_time
        print("Time NO: ",time_NO)
        
        
        start_time = time.time()
        train_step(agentM, device, circuit, rewards, best_rewards, 'M', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb, train_model=args.train)
        time_M = time.time()-start_time
        print("Time M: ",time_M)
        
        start_time = time.time()
        train_step(agentM, device, circuit, rewards, best_rewards, 'M', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb)
        time_M_t = time.time()-start_time
        print("Time M with Training: ",time_M_t)
       
        if len(cirq.moments)<1000:
            start_time = time.time()
            train_step(agentG, device, circuit, rewards, best_rewards, 'G', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb, train_model=args.train)
            time_G = time.time()-start_time
            print("Time G: ",time_G)
        
        start_time = time.time()
        print("Cirq Routing Distance: ", cirq_routing(circuit, device))
        time_cirq = time.time()-start_time
        print("Time Cirq: ",time_cirq)
        
        #print("Qiskit Routing Distance: ", qiskit_routing(circuit, device,start_time))
        q_circuit = qiskit.QuantumCircuit.from_qasm_str(circuit.cirq.to_qasm(header=''))
        coupling_map = list(map(list, device.edges)) + list(map(list, map(reversed, device.edges)))
        routing = {'basic': 0, 'stochastic': 0, 'sabre': 0}
        routing_time = {'t_basic': 0, 't_stochastic': 0, 't_sabre': 0}
        #routing = {'stochastic': 0}
        for rt in routing.keys():
            start_time = time.time()
            try:
                tr_circuit = qiskit.compiler.transpile(q_circuit, coupling_map=coupling_map, routing_method=rt,
                                                       optimization_level=0)
                routing.update({rt: tr_circuit.depth()})
                routing_time.update({"t_"+rt: time.time()-start_time})
            except qiskit.transpiler.exceptions.TranspilerError:
                routing.update({rt: 999999999})

        print("Qiskit Routing Distance: ", routing)
        print("Time Qiskit: ", routing_time)
        
        start_time = time.time()
        print("PyTket Routing Distance: ", tket_routing(circuit, device))
        time_pytket = time.time()-start_time
        print("Time PyTket: ",time_pytket)
      
    

    elif args.dataset == "large":
        
            large_files = ["rd84_142", "adr4_197", "radd_250", "z4_268", "sym6_145", "misex1_241",
                       "rd73_252", "cycle10_2_110", "square_root_7", "sqn_258", "rd84_253"]
            i = 0 
            file = args.large_files
            i += 1
            cirq = circuit_from_qasm(
                os.path.join("./test/circuit_qasm", file + "_onlyCX.qasm"))
            rewards = []
            best_rewards = []
            circuit = CircuitRepDQN(cirq, len(device))
            print(f"Gates in input circuit {file}: ",sum(1 for _ in cirq.all_operations()))
            start_time = time.time()
            curr_time = time.time() - start_time
            times = []
            depths = []
            print("Layers in input circuit: ", len(cirq.moments))
            agentN = NMCAgent(model, device, memory,levelN=args.levelN)
            agentM = MCTSAgent(model,device,memory,search_depth=args.search)
            agentG = GNRPAagent(model,device,memory,levelG=args.levelG,numitersG = args.numitersG)
            '''
            rewards = []
            best_rewards = []
            start_time = time.time()
            train_step(agentN, device, circuit, rewards, best_rewards, 'N', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb, train_model=args.train)
            time_N = time.time()-start_time
            print("Time N: ",time_N)
            
            
            start_time = time.time()
            train_step(agentM, device, circuit, rewards, best_rewards, 'M', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb, train_model=args.train)
            time_M = time.time()-start_time
            print("Time M: ",time_M)
            '''
            start_time = time.time()
            train_step(agentM, device, circuit, rewards, best_rewards, 'M', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb)
            time_M_t = time.time()-start_time
            print("Time M with Training: ",time_M_t)
            '''
            start_time = time.time()
            train_step(agentN, device, circuit, rewards, best_rewards, 'NO', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb, train_model=args.train)
            time_N = time.time()-start_time
            print("Time NO: ",time_N)
            
            if len(cirq.moments)<1000:
                start_time = time.time()
                train_step(agentG, device, circuit, rewards, best_rewards, 'G', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb, train_model=args.train)
                time_G = time.time()-start_time
                print("Time G: ",time_G)
            
            start_time = time.time()
            print("Cirq Routing Distance: ", cirq_routing(circuit, device))
            time_cirq = time.time()-start_time
            print("Time Cirq: ",time_cirq)
            
            #print("Qiskit Routing Distance: ", qiskit_routing(circuit, device,start_time))
            q_circuit = qiskit.QuantumCircuit.from_qasm_str(circuit.cirq.to_qasm(header=''))
            coupling_map = list(map(list, device.edges)) + list(map(list, map(reversed, device.edges)))
            routing = {'basic': 0, 'stochastic': 0, 'sabre': 0}
            routing_time = {'t_basic': 0, 't_stochastic': 0, 't_sabre': 0}
            #routing = {'stochastic': 0}
            for rt in routing.keys():
                start_time = time.time()
                try:
                    tr_circuit = qiskit.compiler.transpile(q_circuit, coupling_map=coupling_map, routing_method=rt,
                                                           optimization_level=0)
                    routing.update({rt: tr_circuit.depth()})
                    routing_time.update({"t_"+rt: time.time()-start_time})
                except qiskit.transpiler.exceptions.TranspilerError:
                    routing.update({rt: 999999999})
    
            print("Qiskit Routing Distance: ", routing)
            print("Time Qiskit: ", routing_time)
            
            start_time = time.time()
            print("PyTket Routing Distance: ", tket_routing(circuit, device))
            time_pytket = time.time()-start_time
            print("Time PyTket: ",time_pytket)
            
            while curr_time<1200:
                circuit = CircuitRepDQN(cirq, len(device))
                curr_depth = train_step(agent, device, circuit, rewards, best_rewards, args.agent, args.levelN, args.levelG, args.numitersG, args.search, episode_name=file, use_wandb=args.wandb, train_model=args.train)
                if curr_depth<best_depth:
                    depths.append(curr_depth)
                    curr_time = time.time() - start_time 
                    times.append(curr_time)
                    best_depth = curr_depth
                    print(best_depth)
                else:
                    depths.append(best_depth)
                    curr_time = time.time() - start_time 
                    times.append(curr_time)
            
            plt.figure(figsize=(10, 6))
            plt.plot(times, depths, marker='o', linestyle='-', color='b')
            
            # Add titles and labels
            plt.title('Best output Depth vs Time for QISKIT stochastic')
            plt.xlabel('Time')
            plt.ylabel('Best Output Depth')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Define the save path
            save_name = "QISKIT_best_output_vs_time_large_circuit.png"
            save_path = os.path.join(save_dir, save_name)
            # Show the plot
            plt.grid(True)
            plt.savefig(save_path)
            plt.show()
            plt.clf()
            plt.close()
            
            #print(f"Time taken:{curr_time}")
            #print("Qiskit Routing Distance: ", qiskit_routing(circuit, device))
            #print("PyTket Routing Distance: ", tket_routing(circuit, device))
            
            if i==1:
                break
            '''
    elif args.dataset == "random":
        #for e in range(args.iterations):
            cirq = circuit_generated_randomly(len(device), args.gates)
            circuit = CircuitRepDQN(cirq, len(device))
            print("Layers in input circuit: ", len(cirq.moments))
            agentN = NMCAgent(model, device, memory,levelN=args.levelN)
            agentM = MCTSAgent(model,device,memory,search_depth=args.search)
            agentG = GNRPAagent(model,device,memory,levelG=args.levelG,numitersG = args.numitersG)
            rewards = []
            best_rewards = []
            start_time = time.time()
            train_step(agentN, device, circuit, rewards, best_rewards, 'N', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb, train_model=args.train)
            time_N = time.time()-start_time
            print("Time N: ",time_N)
            
            start_time = time.time()
            train_step(agentM, device, circuit, rewards, best_rewards, 'M', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb, train_model=args.train)
            time_M = time.time()-start_time
            print("Time M: ",time_M)
            
            start_time = time.time()
            train_step(agentM, device, circuit, rewards, best_rewards, 'M', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb)
            time_M_t = time.time()-start_time
            print("Time M with Training: ",time_M_t)
            
            start_time = time.time()
            train_step(agentG, device, circuit, rewards, best_rewards, 'G', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb, train_model=args.train)
            time_G = time.time()-start_time
            print("Time G: ",time_G)
            
            start_time = time.time()
            train_step(agentN, device, circuit, rewards, best_rewards, 'NO', args.levelN, args.levelG, args.numitersG, args.search, episode_name="random_0", use_wandb=args.wandb, train_model=args.train)
            time_N = time.time()-start_time
            print("Time NO: ",time_N)
            
            start_time = time.time()
            print("Cirq Routing Distance: ", cirq_routing(circuit, device))
            time_cirq = time.time()-start_time
            print("Time Cirq: ",time_cirq)
            
            #print("Qiskit Routing Distance: ", qiskit_routing(circuit, device,start_time))
            q_circuit = qiskit.QuantumCircuit.from_qasm_str(circuit.cirq.to_qasm(header=''))
            coupling_map = list(map(list, device.edges)) + list(map(list, map(reversed, device.edges)))
            routing = {'basic': 0, 'stochastic': 0, 'sabre': 0}
            routing_time = {'t_basic': 0, 't_stochastic': 0, 't_sabre': 0}
            #routing = {'stochastic': 0}
            for rt in routing.keys():
                start_time = time.time()
                try:
                    tr_circuit = qiskit.compiler.transpile(q_circuit, coupling_map=coupling_map, routing_method=rt,
                                                           optimization_level=0)
                    routing.update({rt: tr_circuit.depth()})
                    routing_time.update({"t_"+rt: time.time()-start_time})
                except qiskit.transpiler.exceptions.TranspilerError:
                    routing.update({rt: 999999999})

            print("Qiskit Routing Distance: ", routing)
            print("Time Qiskit: ", routing_time)
            
            start_time = time.time()
            print("PyTket Routing Distance: ", tket_routing(circuit, device))
            time_pytket = time.time()-start_time
            print("Time PyTket: ",time_pytket)
            
            
    elif args.dataset == "full":
        for e in range(args.iterations):
            cirq = circuit_generated_full_layer(len(device), args.gates)
            circuit = CircuitRepDQN(cirq, len(device))
            print("Layers in input circuit: ", len(cirq.moments))
            train_step(agent, device, circuit, rewards, best_rewards, args.agent, args.levelN, args.levelG, args.numitersG, args.search, episode_name=f"full_{e}", use_wandb=args.wandb, train_model=args.train)
            print("Cirq Routing Distance: ", cirq_routing(circuit, device))
            print("Qiskit Routing Distance: ", qiskit_routing(circuit, device))
            print("PyTket Routing Distance: ", tket_routing(circuit, device))
