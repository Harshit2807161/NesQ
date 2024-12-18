

import copy
import typing as ty
import collections
import math
import qiskit
import numpy as np
import torch
import time
from ..metas import CombinerAgent
from ..environment.state import CircuitStateDQN
from ..environment.env import step, evaluate
from nesq.algorithms.deepmcts import MCTSAgent

MemoryItem = collections.namedtuple('MemoryItem', ['state', 'reward', 'action', 'next_state', 'done'])

Lreward = []

class NMCQAgent2(CombinerAgent):
    """
    
    """
    class NMCState:
        """
        State object representing the solution (boolean vector of swaps) as a MCTS node
        """

        HYPERPARAM_NOISE_ALPHA = 0.2
        HYPERPARAM_PRIOR_FRACTION = 0.25

        def __init__(self, state, model, solution=None, r_previous=0, parent_state=None, parent_action=None):
            """
            Initialize a new state
            """
            self.state: CircuitStateDQN = state
            self.model = model
            self.parent_state, self.parent_action = parent_state, parent_action
            self.r_previous = r_previous
            self.num_actions = len(self.state.device.edges)
            self.solution: np.ndarray = copy.copy(solution) if solution is not None else \
                np.full(self.num_actions, False)

            #self.rollout_reward = self.rollout() if self.parent_action is not None else 0.0
            self.action_mask = np.concatenate([state.device.swappable_edges(
                self.solution, self.state.locked_edges, self.state.target_nodes == -1),
                np.array([solution is not None or np.any(self.state.locked_edges)])])
            
            self.n_value = torch.zeros(self.num_actions + 1)
            self.q_value = torch.zeros(self.num_actions + 1)
            self.child_states: ty.List[ty.Optional[MCTSAgent.MCTSState]] = [None for _ in range(self.num_actions + 1)]

      
            model.eval()
            with torch.no_grad():
                _value, self.priors = self.model(self.state)
                self.priors = self.priors.detach().numpy()
                self.priors += np.bitwise_not(self.action_mask) * -1e8
                self.priors = torch.flatten(torch.tensor(self.priors))
            noise = np.random.dirichlet([self.HYPERPARAM_NOISE_ALPHA for _ in self.priors]) * self.action_mask
            self.priors = self.HYPERPARAM_PRIOR_FRACTION * self.priors + (1 - self.HYPERPARAM_PRIOR_FRACTION) * noise



    HYPERPARAM_DISCOUNT_FACTOR = 0.95
    HYPERPARAM_EXPLORE_C = 100
    HYPERPARAM_POLICY_TEMPERATURE = 0
    
    def __init__(self, model, device, memory,levelN = 1):
        super().__init__(model, device)
        self.model = model
        self.root: ty.Optional[MCTSAgent.MCTSState] = None
        self.memory = memory
        self.levelN = levelN
        self.j = 0
        
    def play (self,etat, move):
        next_sol = np.copy(etat.solution)
        next_sol[move] = True

        etat.child_states[move] = NMCQAgent2.NMCState(
                            etat.state, self.model, next_sol,0, etat, move)
        return etat.child_states[move], 0


    def playout(self,etat, rewards, L):
        node = copy.deepcopy(etat)
        solution = copy.deepcopy(etat.solution)
        
        L2, _ = self.qiskit_routing(node.state)
        
        for i in range(len(L2)):
            L.append(L2[i])
            solution[L2[i]] = True
        
        _, reward, _ , _ = step(solution, etat.state)
        L.append(len(solution))
        return reward, L


    def nested(self, etat, rewards, level, L):
        bestSequence = []
        bestReward = -1000000
        # At any level L is the history of moves played to reach there from root 
        # len(L) gives where you are in the bestsequence - your current bestmove
        reward = bestReward
        etat1 = copy.deepcopy(etat)
        while True: 
            mask = etat1.action_mask
            moves = np.where(mask)[0]
            if not np.any(mask):
                break
            
            move_scores = []
            L1s = []
            for i in moves:
                    L1 = copy.deepcopy(L)    
                    if i == len(etat1.solution):
                        break
                    
                    s1, _ = self.play(etat1, i)
                    
                    L1.append(i)
                    
                    if(level == 1):
                        reward, L1 = self.playout(s1, rewards, L1)
                    else: 
                        reward, L1 = self.nested(s1, rewards, level-1, L1) 
                    
                    move_scores.append(reward)
                    L1s.append(L1)
                    
            if (len(move_scores)!=0):
                if move_scores[np.argmax(move_scores)] > bestReward:
                    
                    bestReward = move_scores[np.argmax(move_scores)]
                    bestSequence = L1s[np.argmax(move_scores)]
                    # print("L1", bestSequence)
            '''
            if etat1.state.is_done():
                break  
            '''
            if etat1.state.is_done():
                break  
                        
            if len(etat1.solution) == bestSequence[len(L)]:
                break  
            

            etat1, _ = self.play(etat1,bestSequence[len(L)])

            L.append(bestSequence[len(L)])
            
        #L.append(len(state.solution))
        return bestReward, bestSequence
        
        
    def act(self,state,rewards):

        state_to_act = copy.deepcopy(state)
        
        if self.root is None or self.root.state != state:
            self.root = NMCQAgent2.NMCState(state, self.model)
        else:
            self.root.parent_state = None
            self.root.parent_action = None 
        
        best_score = -10000000
        fin_solution: np.ndarray = np.full(self.root.num_actions, False)
        

        state: NMCQAgent2.NMCState = self.root
        
        bestRew, L = self.nested(copy.deepcopy(self.root), rewards, self.levelN, [])
        
        
        if bestRew > best_score:
            fin_solution = self.root.solution
            assert not np.any(np.bitwise_and(state_to_act.locked_edges, self.root.solution)), "Bad Action"
            for m in L:
                if m==len(fin_solution):
                    break
                fin_solution[m] = True
        
        
        return fin_solution

    def replay(self):
        self.model.train()
        value_losses = []
        policy_losses = []
        for state, v, p in self.memory:
            loss_v, loss_p = self.model.fit(state, v, p)
            value_losses.append(loss_v)
            policy_losses.append(loss_p)
        self.memory.clear()
        return np.mean(value_losses), np.mean(policy_losses)

    def qiskit_routing(self, state):
        circuit = state.circuit
        q_circuit = qiskit.QuantumCircuit.from_qasm_str(circuit.cirq.to_qasm(header=''))
        device = state.device
        coupling_map = list(map(list, device.edges)) + list(map(list, map(reversed, device.edges)))
        # pass the updated circuit 
        # Plan B: no changes here, but omit the illegal SWAPs in second half of function
        routing = 'stochastic'
        print(q_circuit)
        print(coupling_map)
        tr_circuit = qiskit.compiler.transpile(q_circuit, coupling_map=coupling_map, routing_method=routing,
                                                       optimization_level=0)
        print(tr_circuit)
        print(tr_circuit.data)
        swaps = []
        for gate, qargs, cargs in tr_circuit.data:
            if gate.name == 'swap':
                qubit1 = qargs[0]._index
                qubit2 = qargs[1]._index
                swaps.append((qubit1, qubit2))
        print(swaps)
        # Initialize the list of edges where swaps are made
        edges_involved_in_swaps = [False] * len(device.edges)
        
        # Check which edges were involved in swaps
        for swap in swaps:
            for idx, edge in enumerate(device.edges):
                if (swap[0], swap[1]) == edge or (swap[1], swap[0]) == edge:
                    edges_involved_in_swaps[idx] = True
    
        swap_indices = [idx for idx, involved in enumerate(edges_involved_in_swaps) if involved]
    
        return swap_indices,-tr_circuit.depth()

