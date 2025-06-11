"""
MCTS implementation.
"""
from pathlib import Path
import sys
import numpy as np
import random
import time
from Former.Cpp_code.former_class_cpp import FormerGame
from models.heuristics import find_point_minimizing_groups_2
from models.PPO_classes import get_policy_PPO

# ensure project root on sys.path for imports
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MCTSNode:
    """A node in the MCTS search tree."""
    def __init__(self, state, prior, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.prior = prior
        self.depth = parent.depth + 1 if parent else 0
        self.best = state.shape[0] * state.shape[1]
        self.move_sum = 0


class MCTS:
    """Monte Carlo Tree Search with fixed or time-limited iterations."""
    def __init__(self, root_state, net1, net2=None, params=None,
                 anytime=False, best_sol=None, policy_type='network', t=0):
        self.root = MCTSNode(root_state, prior=1)
        defaults = {
            'num_simulations': 1000,
            'c_puct': 10.0,
            'min_exploration_limit': 10,
            'rollout': False,
            'Q_type': 'max',
            'use_prior': True
        }
        self.params = params or defaults
        self.net1 = net1
        self.net2 = net2
        self.num_simulations = self.params['num_simulations']
        self.c_puct = self.params['c_puct']
        self.min_explore = self.params['min_exploration_limit']
        self.rollout = self.params['rollout'] or (net1 is None and net2 is None)
        self.Q_type = self.params['Q_type']
        self.use_prior = self.params['use_prior']
        self.policy_type = policy_type
        self.t = t
        self.anytime = anytime
        self.best_sol = best_sol
        self.best_found = float('inf')
        self.best_sequence = []
        self.start_time = time.time()

    def _puct(self, node):
        """Compute PUCT score combining value and exploration."""
        parent_visits = node.parent.visit_count if node.parent else 1
        if self.Q_type == 'max':
            Q = -node.best
        else:
            Q = -node.move_sum / node.visit_count
        U = self.c_puct * node.prior * np.sqrt(parent_visits) / (node.visit_count + 1)
        return Q + U

    def select(self, node):
        """Traverse tree to a leaf for expansion or simulation."""
        while node.children:
            best, best_score = None, -np.inf
            for child in node.children.values():
                if child.visit_count == 0 or (node.parent is None and child.visit_count < self.min_explore):
                    return child
                score = self._puct(child)
                if score > best_score:
                    best, best_score = child, score
            node = best
        return node

    def _get_priors(self, state):
        """Compute prior probabilities for available actions."""
        actions = FormerGame.get_valid_turns_static(state)
        if self.use_prior:
            if self.policy_type == 'heuristic':
                if self.t > 0:
                    best = find_point_minimizing_groups_2(state, T=self.t)
                    return {a: 1 if a == best else 0 for a in actions}
                p = 1 / len(actions)
                return {a: p for a in actions}
            if self.policy_type == 'ppo':
                return get_policy_PPO(state, self.net2)
            return self.net2.evaluate_state(state)
        p = 1 / len(actions)
        return {a: p for a in actions}

    def expand(self, node):
        """Add child nodes to a leaf based on priors."""
        priors = self._get_priors(node.state)
        for act, prob in priors.items():
            new_state = np.array(FormerGame.apply_turn_static(node.state, act))
            node.children[act] = MCTSNode(new_state, prior=prob, parent=node, action=act)

    def simulate(self, node):
        """Evaluate or rollout to estimate moves and sequence."""
        state = node.state.copy()
        if FormerGame.is_game_over_static(state):
            return 0, []
        if not self.rollout:
            out = self.net1.evaluate_state(state)
            return (out[0] if isinstance(out, tuple) else out), []
        count, seq = 0, []
        while not FormerGame.is_game_over_static(state):
            policy = self._get_priors(state)
            action = random.choices(list(policy), weights=policy.values(), k=1)[0]
            state = np.array(FormerGame.apply_turn_static(state, action))
            seq.append(action)
            count += 1
        return count, seq

    def backpropagate(self, node, value, rollout_seq=None):
        """Propagate simulation results up the tree."""
        total = value + node.depth
        while node:
            node.visit_count += 1
            node.move_sum += total
            if total < node.best:
                node.best = total
            node = node.parent
        return rollout_seq or []

    def search(self):
        """Run a fixed number of MCTS simulations and return best action."""
        for _ in range(self.num_simulations):
            leaf = self.select(self.root)
            if FormerGame.is_game_over_static(leaf.state):
                self.backpropagate(leaf, 0)
                continue
            self.expand(leaf)
            val, seq = self.simulate(leaf)
            self.backpropagate(leaf, val, seq)
        return self.get_best_action()

    def search_with_time_limit(self, time_limit, swap_freq=float('inf')):
        """Run MCTS until the time limit (s) is reached, returning moves, times, sequence, and sims."""
        start = time.time()
        last_swap = start
        moves, times, sims = [], [], 0
        while time.time() - start < time_limit:
            sims += 1
            if time.time() - last_swap > swap_freq:
                best = self.get_best_action()
                self.root.children.pop(best, None)
                last_swap = time.time()
            leaf = self.select(self.root)
            if FormerGame.is_game_over_static(leaf.state):
                self.backpropagate(leaf, 0)
                continue
            self.expand(leaf)
            val, seq = self.simulate(leaf)
            self.backpropagate(leaf, val, seq)
            total = val + leaf.depth
            # record anytime improvements
            if self.anytime and total < self.best_found:
                self.best_found = total
                self.best_sequence = seq
                moves.append(total)
                times.append(time.time() - start)
        return moves, times, self.best_sequence, sims

    def get_best_action(self):
        """Return the action with highest visit count from root."""
        return max(self.root.children,
                   key=lambda a: self.root.children[a].visit_count)
