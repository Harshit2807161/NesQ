"""
Monte Carlo Tree Search for asymmetric trees
CREDITS : Thomas Moerland, Delft University of Technology
"""

import copy
import typing as ty
import collections
import math

import numpy as np
import torch

from ..metas import CombinerAgent
from ..environment.state import CircuitStateDQN
from ..environment.env import step, evaluate
from nesq.algorithms.deepmcts import MCTSAgent

MemoryItem = collections.namedtuple('MemoryItem', ['state', 'reward', 'action', 'next_state', 'done'])

Lreward = []

class Policy():
    def __init__ (self):
        self.dict = {}

    def get(self, code):
        w = 0
        if code in self.dict:
            w = self.dict[code]
        return w

    def put(self, code, w):
        self.dict[code] = w 



class GNRPAagent(CombinerAgent):
    """
    
    """
    class GNRPAState:
        """
        State object representing the solution (boolean vector of swaps) as a MCTS node
        """

        HYPERPARAM_NOISE_ALPHA = 0.2
        HYPERPARAM_PRIOR_FRACTION = 0.25

        def __init__(self, state, model, n=1, solution=None, r_previous=0, parent_state=None, parent_action=None):
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
            self.n = 1 

            model.eval()
            with torch.no_grad():
                _value, self.priors = self.model(self.state)
                self.priors = self.priors.detach().numpy()
                self.priors += np.bitwise_not(self.action_mask) * -1e8
                self.priors = torch.flatten(torch.tensor(self.priors))
            noise = np.random.dirichlet([self.HYPERPARAM_NOISE_ALPHA for _ in self.priors]) * self.action_mask
            self.priors = self.HYPERPARAM_PRIOR_FRACTION * self.priors + (1 - self.HYPERPARAM_PRIOR_FRACTION) * noise
            self.N = 10
        
        def code(self, p):
            return p
            

    HYPERPARAM_DISCOUNT_FACTOR = 0.95
    HYPERPARAM_EXPLORE_C = 100
    HYPERPARAM_POLICY_TEMPERATURE = 0
      
    def __init__(self, model, device, memory, levelG = 1, numitersG = 100):
        super().__init__(model, device)
        self.model = model
        self.root: ty.Optional[MCTSAgent.MCTSState] = None
        self.memory = memory
        self.levelG = levelG
        self.numitersG = numitersG
    def play(self,etat, move):
        next_sol = np.copy(etat.solution)
        next_sol[move] = True

        
        reward = evaluate(next_sol, etat.state) - \
                                 evaluate(etat.solution, etat.state)
 
        etat.child_states[move] = GNRPAagent.GNRPAState(
                            etat.state, self.model, etat.n+1, next_sol,reward, etat, move)
        return etat.child_states[move], reward    
    
    def gnr_playout(self,policy,etat,rewards, t = 1):
        seq = []
        solution = copy.copy(etat.solution)
        # solution = etat.solution
        while True:
            mask = np.concatenate([etat.state.device.swappable_edges(solution, etat.state.locked_edges, etat.state.target_nodes == -1),
                                        np.array([True])]) 
            moves = np.where(mask)[0]
            o = np.zeros(len(moves))

            if not np.any(mask):
                break

            z = 0

            for i in range(len(moves)):
                o[i] = math.exp((policy.get(etat.code(moves[i]))/t) + math.log(math.exp(policy.get(etat.code(moves[i]))))) #To check - log(policy)
                z += o[i]
            
            m = np.random.choice(moves, p = o/z)
            seq.append(m)
            if(m == len(etat.solution)):
                break
            solution[m] = True
        
        _, reward, _, _ = step(solution, etat.state)
        
        rewards.append(reward)
        
        return reward, seq
    
    def adapt(self, policy, seq, alpha = 1,t = 1):
        polp = copy.deepcopy(policy)
        etat = copy.deepcopy(self.root)
        for b in seq:

            mask = etat.action_mask
            moves = np.where(mask)[0]
            o = np.zeros(len(moves))
  
            if not np.any(mask):
                break
            
            z = 0
            
            for i in range(len(moves)):
                o[i] = math.exp((policy.get(etat.code(moves[i]))/t) + math.log(math.exp(policy.get(etat.code(moves[i])))))
                
                z += o[i]

            for i in range(len(moves)):
                if(moves[i] == b):
                    delta = 1
                else:
                    delta = 0

                polp.put(etat.code(moves[i]), polp.get(etat.code(moves[i])) - (alpha/t)*((o[i]/z) - delta))
                
            if b != len(etat.solution):
                etat, _ = self.play(etat,b)
        
        return polp
    
    def GNRPA(self, policy, root, rewards, level, n_iter):
        state = copy.deepcopy(root)
        if level == 0:
            return self.gnr_playout(policy, state, rewards)  # Should be passing the root
        else:
            bestScore = -100000
            pol = copy.deepcopy(policy)
            for i in range(n_iter):
                result, new = self.GNRPA(pol, root, rewards, level-1, n_iter)
                if result >= bestScore:
                    bestScore = result
                    L = new

                pol = self.adapt(pol, L)
            
            return bestScore, L


    def act(self,state,rewards):
        
        state_to_act = copy.deepcopy(state)
        
        if self.root is None or self.root.state != state:
            self.root = GNRPAagent.GNRPAState(state, self.model)
        else:
            self.root.parent_state = None
            self.root.parent_action = None 
        
        fin_solution: np.ndarray = np.full(self.root.num_actions, False)

        state: GNRPAagent.GNRPAState = self.root
        policy = Policy()
        best_score, best_set_of_swaps = self.GNRPA(policy,state,rewards,self.levelG,self.numitersG)
        
        fin_solution = self.root.solution
        
        assert not np.any(np.bitwise_and(state_to_act.locked_edges, self.root.solution)), "Bad Action"
        
        for m in best_set_of_swaps:
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

        
'''        
    def GNRPA_final(root, level, policy):
       state = copy.deepcopy(root)
       if level == 0:
           playout(state, policy)
           return state
       pol = copy.deepcopy(policy) # So that after any NRPA run when we get ws, pol is not changed, it only changes by adapt at end of iteration
       for i in range(100):
           term_state = NRPA(root ,level - 1, pol)
           if term_state.score() >= state.score():
               state = term_state
           pol = adapt(state.sequence, pol)
       return state

    def playout_final(state, policy):
       while not state.terminal():
           l = state.legalMoves()
           z = 0
           for i in range(len(l)):
               z = z + math.exp(policy.get(state.code(l[i])) + state.beta(l[i]))
           stop = random.random()*z
           move = 0
           z = 0
           while True:
               z = z + math.exp(policy.get(state.code(l[move])) + state.beta(l[move]))
               if z >= stop:
                   break
               move = move + 1
           state.play(l[move])

    def adapt_final(sequence, policy):
       polp = copy.deepcopy(policy)
       s = WS ()
       while not s.terminal():
           l = s.legalMoves()
           z = 0
           for i in range(len(l)):
               z = z + math.exp(policy.get(s.code(l[i])) + s.beta(l[i]))
           move = sequence[len(s.sequence)]
           polp.put(s.code(move), polp.get(s.code(move)) + 1)
           for i in range(len(l)):
               proba = math.exp(policy.get(s.code(l[i])) + s.beta(l[i])) / z
               polp.put(s.code(l[i]), polp.get(s.code(l[i])) - proba)
           s.play(move)
       return polp
'''   