
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

class NMCAgent(CombinerAgent):
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
            self.count = [0]*(self.num_actions+1)
            self.nb = [0]*(self.num_actions+1)
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

        def code(self, p):
            return p

            


    HYPERPARAM_DISCOUNT_FACTOR = 0.95
    HYPERPARAM_EXPLORE_C = 100
    HYPERPARAM_POLICY_TEMPERATURE = 0
      
    def __init__(self, model, device, memory,levelN = 1):
        super().__init__(model, device)
        self.model = model
        self.root: ty.Optional[MCTSAgent.MCTSState] = None
        self.memory = memory
        self.levelN = levelN
        
    def play (self,etat, move):
        _reward = 0 
        if len(etat.solution) == move:
            next_state, _reward, _done, _debug = step(etat.solution, etat.state) 
            etat.child_states[move] = NMCAgent.NMCState(
                            next_state, self.model,
                            r_previous=0, parent_state=etat, parent_action=move)
        else:
            next_sol = np.copy(etat.solution)
            next_sol[move] = True
            etat.child_states[move] = NMCAgent.NMCState(
                                etat.state, self.model, next_sol,0, etat, move)
        return etat.child_states[move], _reward
    
    def replay(self,etat, sequence):
        etat1 = copy.deepcopy(etat)
        for b in sequence:
            etat1.count[etat.code(b)] += 1 
            mask = np.concatenate([etat1.state.device.swappable_edges(etat1.solution, etat1.state.locked_edges, etat1.state.target_nodes == -1),
                                    np.array([True])]) 
            moves = np.where(mask)[0]
            for m in moves:
                etat1.nb[etat1.code(m)] += 1
            
            etat1,_ = self.play(etat1,b)
            

    def playout(self,etat, rewards, L):

        solution = np.copy(etat.solution) # changes in Copy does not change original 
        i = 0
        while True:
            i+=1
            mask = np.concatenate([etat.state.device.swappable_edges(solution, etat.state.locked_edges, etat.state.target_nodes == -1),
                                    np.array([True])]) 

            if not np.any(mask):
                break
            
            swap = np.random.choice(np.where(mask)[0])
            
            if swap == len(solution):
                L.append(swap)
                break  # This only evaluates one step deep
            
            solution[swap] = True
            L.append(swap)
            
        _, reward, _ , _ = step(solution, etat.state)
        rewards.append(reward)
        return reward, L

    def prior_playout(self,etat,bias,rewards,L):

        solution = np.copy(etat.solution) # changes in Copy does not change original 
        i = 0
        while True:
            i+=1
            mask = np.concatenate([etat.state.device.swappable_edges(solution, etat.state.locked_edges, etat.state.target_nodes == -1),
                                    np.array([True])]) 

            if not np.any(mask):
                break
            
            possible_move_indices = np.where(mask)[0]
            swap = np.random.choice(np.where(mask)[0],p=bias)
            
            if swap == len(solution):
                L.append(swap)
                break  # This only evaluates one step deep
            
            solution[swap] = True
            L.append(swap)
            
        _, reward, _ , _ = step(solution, etat.state)
        rewards.append(reward)
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
                    
                    '''
                    if i == len(etat.solution):
                        break #is it okay?
                    '''
                    s1, _ = self.play(etat1, i) 
                    
                    L1.append(i)
                    
                    if i == len(etat1.solution):
                        reward = _
                        move_scores.append(reward)
                        L1s.append(L1)
                        continue
                        
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

            if len(etat1.solution) == bestSequence[len(L)]:
                break  
            
            etat1, _ = self.play(etat1,bestSequence[len(L)])
            L.append(bestSequence[len(L)])

        #L.append(len(state.solution))
        return bestReward, bestSequence
        
        
    def act(self,state,rewards):
        state_to_act = copy.deepcopy(state)
        
        if self.root is None or self.root.state != state:
            self.root = NMCAgent.NMCState(state, self.model)
        else:
            self.root.parent_state = None
            self.root.parent_action = None 
        
        best_score = -10000000
        fin_solution: np.ndarray = np.full(self.root.num_actions, False)
        

        state: NMCAgent.NMCState = self.root
        
        _, L = self.nested(copy.deepcopy(self.root), rewards, self.levelN, [])
        if _ > best_score:
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

