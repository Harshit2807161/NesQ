"""
Monte Carlo Tree Search for asymmetric trees
CREDITS : Thomas Moerland, Delft University of Technology
"""

import copy
import typing as ty
import collections
import time
import numpy as np
import torch

from ..metas import CombinerAgent
from ..environment.state import CircuitStateDQN
from ..environment.env import step, evaluate

MemoryItem = collections.namedtuple('MemoryItem', ['state', 'reward', 'action', 'next_state', 'done'])


class MCTSAgent(CombinerAgent):

    class MCTSState:
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
            self.r_previous = r_previous  # Useful to get reward when expanding to a child via performing SWAP 
            self.num_actions = len(self.state.device.edges)
            self.solution: np.ndarray = copy.copy(solution) if solution is not None else \
                np.full(self.num_actions, False)  # Tracks edges which have been executed
            self.rollout_reward = self.rollout() if self.parent_action is not None else 0.0  #IF root, rollout reward is 0 else perform rollout
            self.action_mask = np.concatenate([state.device.swappable_edges(
                self.solution, self.state.locked_edges, self.state.target_nodes == -1),
                np.array([solution is not None or np.any(self.state.locked_edges)])])
            
            self.n_value = torch.zeros(self.num_actions + 1) # Last one is commit action child
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

        def update_q(self, reward, index):
            """
            Updates the q-value for the state
            :param reward: The obtained total reward from this state
            :param index: the index of the action chosen for which the reward was provided

            n_value is the number of times a node visited
            q_value is the q function

            n += 1, w += reward, q = w / n -> this is being implicitly computed using the weighted average
            """
            self.q_value[index] = (self.q_value[index] * self.n_value[index] + reward) / (self.n_value[index] + 1)
            self.n_value[index] += 1

        def select(self, c=1000) -> int:
            """
            Select one of the child actions based on UCT rule
            """
            n_visits = torch.sum(self.n_value).item()
            uct = self.q_value + (self.priors * c * np.sqrt(n_visits + 0.001) / (self.n_value + 0.001))
            best_val = torch.max(uct)
            best_move_indices: torch.Tensor = torch.where(torch.eq(best_val, uct))[0]
            winner: int = np.random.choice(best_move_indices.numpy())
            return winner

        def choose(self) -> int:
            """
            Select one of the child actions based on the best q-value which is allowed
            """
            q_real = self.q_value + np.bitwise_not(self.action_mask) * -1e8  
            best_val = torch.max(q_real)
            best_move_indices: torch.Tensor = torch.where(torch.eq(best_val, q_real))[0]
            winner: int = np.random.choice(best_move_indices.numpy())
            return winner

        def rollout(self, num_rollouts=None):  # TODO: Benchmark this on 100 rollouts
            """
            performs R random rollout, the total reward in each rollout is computed.
            returns: mean across the R random rollouts.
            """
            if num_rollouts is None:
                assert not np.any(np.bitwise_and(self.state.locked_edges, self.solution)), "Bad Action" # A locked edge is swapped by solution (When will this be called?)
                next_state, _, _, _ = step(self.solution, self.state) # Commits the current swaps to the state
                with torch.no_grad():
                    self.model.eval()
                    self.rollout_reward, _priors = self.model(next_state) # Evaluate model through NN
                return self.rollout_reward.item()
            else:
                total_reward = 0
                for i in range(num_rollouts):
                    solution = np.copy(self.solution) # Reset the solution to the solution at the current state being rolled out 
                    # How is self.solution reset?
                    while True:
                        mask = np.concatenate([self.state.device.swappable_edges(solution, self.state.locked_edges),
                                               np.array([True])]) # Last element for no swap option
                        if not np.any(mask):
                            break
                        swap = np.random.choice(np.where(mask)[0]) # Choose an edge to swap randomly
                        if swap == len(solution):
                            break  # Commit option 
                        solution[swap] = True
                    _, reward, _, _ = step(solution, self.state) # Commit action to get reward and update state (not reset it)
                    total_reward += reward
                    
                return total_reward / num_rollouts

    """
    Monte Carlo Tree Search combiner object for evaluating the combination of moves
    that will form one step of the simulation.
    This at the moment does not look into the future steps, just calls an evaluator
    """

    HYPERPARAM_DISCOUNT_FACTOR = 0.95
    HYPERPARAM_EXPLORE_C = 100
    HYPERPARAM_POLICY_TEMPERATURE = 0

    def __init__(self, model, device, memory, search_depth=100):
        super().__init__(model, device)
        self.model = model
        self.root: ty.Optional[MCTSAgent.MCTSState] = None
        self.memory = memory
        self.search_depth = search_depth

    def search(self, n_mcts,rewards):
        """Perform the MCTS search from the root"""
        max_depth, mean_depth = 0, 0
        #print("Starting search for 200 iterations\n")
        for _ in range(n_mcts):
            mcts_state: MCTSAgent.MCTSState = self.root  # reset to root for new trace
            # input(str(self.root.n_value) + " " + str(self.root.q_value))  # To Debug the tree
            depth = 0

            while True:
                depth += 1
                action_index: int = mcts_state.select()
                
                # If no expansion to be done - all children explored
                if mcts_state.child_states[action_index] is not None:
                    # MCTS Algorithm: SELECT STAGE
                    mcts_state = mcts_state.child_states[action_index]
                    continue
                
                # Terminal state
                elif mcts_state.state.is_done():
                    break
                
                # Unexplored child
                else:
                    # MCTS Algorithm: EXPAND STAGE
                    if action_index == len(mcts_state.solution):  # This is a commit action
                        next_state, _reward, _done, _debug = step(mcts_state.solution, mcts_state.state) 
                        mcts_state.child_states[action_index] = MCTSAgent.MCTSState(
                            next_state, self.model,
                            r_previous=0, parent_state=mcts_state, parent_action=action_index)
                    else:  # This is a swap action
                        next_solution = np.copy(mcts_state.solution)
                        next_solution[action_index] = True
                        reward = evaluate(next_solution, mcts_state.state) - \
                                 evaluate(mcts_state.solution, mcts_state.state)
                        mcts_state.child_states[action_index] = MCTSAgent.MCTSState(
                            mcts_state.state, self.model, next_solution, reward, mcts_state, action_index)
                    mcts_state = mcts_state.child_states[action_index]
                    break

            # SIMULATION- why NONE case is done? 
            total_reward = mcts_state.rollout_reward
            rewards.append(total_reward)
            #print(f"Rollout reward for search iteration {_} is {total_reward} and depth of search was {depth}")

            # MCTS Algorithm: BACKUP STAGE - BACKPROPOGATION
            while mcts_state.parent_action is not None:
                total_reward = mcts_state.r_previous + self.HYPERPARAM_DISCOUNT_FACTOR * total_reward
                mcts_state.parent_state.update_q(total_reward, mcts_state.parent_action)
                mcts_state = mcts_state.parent_state


            max_depth = max(max_depth, depth)
            mean_depth += depth / n_mcts

        return max_depth, mean_depth

    @staticmethod
    def _stable_normalizer(x, temp=1.5):
        x = (x / torch.max(x)) ** temp
        return torch.abs(x / torch.sum(x))
    
    
    # Final part of MCTS- Do the best commit possible (remember? action at a point is a set of SWAPs? exactly. that was the aim all along. move != action(output from MCTS))
    def act(self, state, rewards):
        """Process the output at the root node"""
        self.j = 0 
        if self.root is None or self.root.state != state:
            self.root = MCTSAgent.MCTSState(state, self.model)
        else:
            self.root.parent_state = None
            self.root.parent_action = None
        s = 0 
        while True:
            # print(f"Number of swaps done: {s}")
            # Keep swapping until the best commit is done
            self.search(self.search_depth,rewards)
            self.memory.store(state,
                              torch.sum((self.root.n_value / torch.sum(self.root.n_value)) * self.root.q_value),
                              self._stable_normalizer(self.root.n_value)) # For updating the NN 
            
            # Use updated Q-values to get the best child 
            pos = self.root.choose()
            
            # If best thing to do is to commit the swaps
            if pos == len(self.root.solution) or self.root.child_states[pos] is None:
                # print("Commiting and returning the best sequence of swaps")
                assert not np.any(np.bitwise_and(state.locked_edges, self.root.solution)), "Bad Action"
                step_solution = self.root.solution
                
                # Do the swaps in the circuit, set whole solution to False (reset), and no reward on selecting the child (r_prev = 0)
                self.root = self.root.child_states[pos]
                
                # Return the set of SWAPs done in the best commit
                return step_solution
            
            
            # If best thing to do is to SWAP 
            else:
                self.root = self.root.child_states[pos]
                s+=1 
        

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