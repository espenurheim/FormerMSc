"""
MCTS implementation.
"""
import numpy as np
from Former.Cpp_code.former_class_cpp import FormerGame
import random
import Former.daily_board as db
import time
from models.heuristics import find_point_minimizing_groups_static, find_point_minimizing_groups_2
from models.PPO_classes import get_policy_PPO

# MCTS
class MCTSNode:
    """
    One node in a MCTS tree. The "expected_best_solution" is the actual best solution obtained - it is called expected
    since the implementation of MCTS allows for using value networks to predict the value instead of full simulations,
    but when using full simulations, it is the exact best solution found.
    """
    def __init__(self, state, prior, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action_made = action
        self.children = {}
        self.visit_count = 0
        self.prior = prior
        self.depth = parent.depth + 1 if parent is not None else 0
        self.expected_best_solution = state.shape[0]*state.shape[1] # Initially, expect the best solution of a node to be the worst possible
        self.move_sum = 0 # The total sum of number of moves used from this state
        
class MCTS:
    def __init__(self, root_state, net1, net2=None, params = {"num_simulations": 1000,
                                                              "c_puct": 10.0,
                                                              "min_exploration_limit": 10,
                                                              "N": 9,
                                                              "M": 7,
                                                              "S": 4,
                                                              "rollout": False,
                                                              "Q_type": "max",
                                                              "use_prior": True}, anytime = False, best_sol = None, policy_type = 'network', t = 0):
        self.root_node = MCTSNode(root_state, 1)
        
        self.params = params
        self.net1 = net1 # Value network (we do not really use this anymore, but it is an option if rollout is False)
        self.net2 = net2 # Policy network
            
        self.num_simulations = params["num_simulations"]
        self.c_puct = params["c_puct"]
        self.N, self.M, self.S = params["N"], params["M"], params["S"]
        self.min_exploration_limit = params["min_exploration_limit"]
        self.rollout = params["rollout"]
        if net1 is None and net2 is None:
            self.rollout = True # If no networks are given, we have to perform rollouts with heuristic
        
        self.Q_type = params["Q_type"]
        self.use_prior = params["use_prior"]
        self.start_time = time.time()
        self.anytime = anytime
        
        # Always keep track of the best found solution
        self.best_found_solution = float('inf') 
        self.best_action_sequence = None
        self.action_list_from_previous_node = []
        self.best_sol = best_sol
        
        self.policy_type = policy_type
        self.t = t # For heuristic
    
        
    def _puct(self, node):
        """
        Calculate the PUCT value for a node.
        """
        parent_visit_count = node.parent.visit_count if node.parent else 1
        if self.Q_type == 'max':
            Q = -node.expected_best_solution # Value is set to be the (negative) best solution, i.e. the fewer moves, the higher the Q.
        elif self.Q_type == 'avg':
            Q = -node.move_sum / node.visit_count # Negative average, to encourage less moves
        U = self.c_puct * node.prior * np.sqrt(parent_visit_count) / (node.visit_count + 1)
        return Q + U
        
    def select(self, node):
        """
        Select a leaf node. 
        """
        while node.children:
            max_puct = float('-inf')
            best_child = None
            for action, child in node.children.items():
                if child.visit_count == 0:     # If child has not been visited, choose it immediately 
                    return child
                if node.parent is None:       # At root node, force a minimum amount of visits
                    if child.visit_count < self.min_exploration_limit:
                        return child
                puct = self._puct(child)
                if puct > max_puct:
                    max_puct = puct
                    best_child = child
            node = best_child
        return node
    
    def expand(self, node, use_prior = True):
        """ 
        Expand the node by generating its children. Returns the last child.
        """
        legal_actions = FormerGame.get_valid_turns_static(node.state)
        
        # Either use heuristic...
        if self.policy_type == 'heuristic' and self.use_prior:
            if self.t == 0: # Random (uniform)
                priors_list = np.ones(len(legal_actions)) / len(legal_actions)
                priors = {action: prob for action, prob in zip(legal_actions, priors_list)}
            elif self.t > 0: # 1 or 2 look-ahead heuristic
                best_act = find_point_minimizing_groups_2(node.state, T = self.t)
                priors = {action: 0 for action in legal_actions}
                priors[best_act] = 1
        
        # ... or policy network...
        elif self.use_prior:
            if self.policy_type == 'ppo':
                priors = get_policy_PPO(node.state, self.net2)
            else:
                priors = self.net2.evaluate_state(node.state)
                
        # ... or uniform distribution.
        else:
            priors = {action: 1/len(legal_actions) for action in legal_actions}
        
        # Add all children
        for action in legal_actions:
            prior = priors[action]
            new_state = np.array(FormerGame.apply_turn_static(node.state, action))
            child_node = MCTSNode(new_state, prior=prior, parent=node, action=action)
            node.children[action] = child_node
            
    def simulate(self, node):
        """
        Simulate a random game from the current node. Return the expected number of moves from the current node.
        If self.rollout = params['rollout'] is set to False, we replace rollout by an estimation from the value net. If it is
        set to True, we perform a full rollout using the policy network as heuristic.
        """
        state = node.state.copy()
        if FormerGame.is_game_over_static(state):
            return 0
        
        # Rollout is false -> use value net instead
        if not self.rollout:
            if self.net2 is None: # Dual-head structure
                return self.net1.evaluate_state(state)[0]
            return self.net1.evaluate_state(state)
        
        # Rollout is true -> perform rollout with policy net / heuristic as policy function
        action_count = 0
        action_list = []
        while not FormerGame.is_game_over_static(state):
            if self.policy_type == 'heuristic' and self.use_prior:
                legal_actions = FormerGame.get_valid_turns_static(state)
                if self.t == 0: # Random (uniform)
                    priors_list = np.ones(len(legal_actions)) / len(legal_actions)
                    policy = {action: prob for action, prob in zip(legal_actions, priors_list)}
                elif self.t > 0: # 1 or 2 look-ahead heuristic
                    best_act = find_point_minimizing_groups_2(state, T = self.t)
                    policy = {action: 0 for action in legal_actions}
                    policy[best_act] = 1
            elif self.policy_type == 'ppo':
                policy = get_policy_PPO(state, self.net2) # Get policy from PPO model
            elif self.net2 is None:
                policy = self.net1.evaluate_state(state)[1] # Dual net (not really used)
            else:
                policy = self.net2.evaluate_state(state) # Policy net
            
            actions = list(policy.keys())
            probs = list(policy.values())
            chosen_action = random.choices(actions, weights=probs, k=1)[0] # Sample from policy

            state = np.array(FormerGame.apply_turn_static(state, chosen_action))
            action_list = action_list + [chosen_action]
            action_count += 1
        
        return action_count, action_list
        
    def backpropagate(self, node, value):
        """
        Backtrack the tree and update the expected best solution and visit counts of all nodes.
        """
        estimated_solution = node.depth + value # The estimated solution found is the value of the leaf node + the depth of the leaf node
        action_list = []
        while node.parent is not None:
            action_list = [node.action_made] + action_list
            if estimated_solution < node.expected_best_solution: # Update the expected best solution of the node if the estimation is lower
                node.expected_best_solution = estimated_solution
            node.move_sum += estimated_solution
            node.visit_count += 1
            node = node.parent
        
        # Update for root node as well
        if estimated_solution < node.expected_best_solution:
            node.expected_best_solution = estimated_solution
        node.move_sum += estimated_solution
        node.visit_count += 1
        return action_list
        
    def search(self):
        """
        Perform a full iteration of MCTS, limited by num_simulations simulations. 
        """
        root = self.root_node
        for _ in range(self.num_simulations):
            if (_ + 1) % self.meta_switch_sims == 0:
                best_action = self.get_best_action()
                root.children.pop(best_action)
                print(f"Blocked action: {best_action}")
            selected_node = self.select(root) # Select leaf node. Can be the root itself.
            if FormerGame.is_game_over_static(selected_node.state): # If leaf node represent a finished game state, backpropagate
                self.backpropagate(selected_node, 0)
                continue
            self.expand(selected_node) # Expand child nodes of the leaf node
            value, simulation_actions = self.simulate(selected_node)
            initial_actions = self.backpropagate(selected_node, value)
            
            sim_value = value + selected_node.depth
            if self.anytime and sim_value < self.best_found_solution: # Print the solution if anytime
                self.best_found_solution = sim_value
                self.best_action_sequence = self.action_list_from_previous_node + initial_actions + simulation_actions
                #print(f"New best: {sim_value}: {self.best_action_sequence} Time passed: {round(time.time()-self.start_time,2)}.")
                if self.best_found_solution == self.best_sol:
                    return len(self.best_action_sequence), self.best_action_sequence
                
        
        visit_counts = {}
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count / (root.visit_count - 1)
        
        return len(self.best_action_sequence), self.best_action_sequence
    
    def search_with_time_limit(self, time_limit, swap_freq = float('inf')):
        """
        Perform a full iteration of MCTS, based on a maximum time limit.
        """
        root = self.root_node
        i = 0
        solution_moves = []
        solution_times = []
        start_time = time.time()
        last_meta = time.time()
        while time.time() - start_time < time_limit:
            if time.time() > last_meta + swap_freq: # Reset
                best_action = self.get_best_action()
                self.root_node.children.pop(best_action)
                print("swap")
                last_meta = time.time()
            
            selected_node = self.select(root) # Select leaf node. Can be the root itself.
            if FormerGame.is_game_over_static(selected_node.state): # If leaf node represent a finished game state, backpropagate
                self.backpropagate(selected_node, 0)
                continue
            self.expand(selected_node) # Expand child nodes of the leaf node
            value, simulation_actions = self.simulate(selected_node)
            initial_actions = self.backpropagate(selected_node, value)
            
            sim_value = value + selected_node.depth
            if self.anytime and sim_value < self.best_found_solution: # Print the solution if anytime
                solution_times.append(time.time()-start_time)
                solution_moves.append(sim_value)
                self.best_found_solution = sim_value
                self.best_action_sequence = self.action_list_from_previous_node + initial_actions + simulation_actions
                #print(f"New best: {sim_value}: {self.best_action_sequence} Time passed: {round(time.time()-self.start_time,2)}.")
                if self.best_found_solution == self.best_sol:
                    return solution_moves, solution_times, self.best_action_sequence, i
            i += 1
                
        
        visit_counts = {}
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count / (root.visit_count - 1)
        
        return solution_moves, solution_times, self.best_action_sequence, i
    
    def get_best_action(self):
        """
        Return the best action to take from the root node, based on the visit counts of its children.
        """
        best_action = None
        max_visit = 0
        for action, child_node in self.root_node.children.items():
            if child_node.visit_count > max_visit:
                best_action = action
                max_visit = child_node.visit_count
                
        return best_action
    
    def move_root_to_child(self, action):
        """
        Moves the root node to the child node. We do not really use this anymore.
        """
        old_root = self.root_node
        self.action_list_from_previous_node += [action]
        child = old_root.children[action]
        child.parent = None

        # Remove all pointers (clear from memory)
        old_root.children.clear()
        old_root.parent = None

        self.root_node = child
        
