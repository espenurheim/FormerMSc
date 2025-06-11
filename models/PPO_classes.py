# Move root one step out
import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from Former.Cpp_code.former_class_cpp import FormerGame
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

# Extract features, input into PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Freeze classes
from stable_baselines3.common.callbacks import BaseCallback


class FormerEnv(gym.Env):
    """
    Gymnasium environment for the puzzle "Former".
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        N_ROWS: int = 9,
        N_COLUMNS: int = 7,
        N_SHAPES: int = 4,
        render_mode=None,
        custom_board=None,
    ):
        super().__init__()
        self.N_ROWS = N_ROWS
        self.N_COLUMNS = N_COLUMNS
        self.N_SHAPES = N_SHAPES

        self.game = FormerGame(N_ROWS, N_COLUMNS, N_SHAPES, custom_board)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.N_ROWS * self.N_COLUMNS)

        # observation: one‑hot grid with shapes 0–3 = channels 0–3, empty = channel 4
        self.observation_space = spaces.Box(
            0.0, 1.0,
            shape=(self.N_SHAPES+1, self.N_ROWS, self.N_COLUMNS),
            dtype=np.float32,
        )

        self.board = self.game.get_board()
        self.action_count = 0
    
    def get_action_mask(self) -> np.ndarray:
        """
        Mask out only the truly illegal actions (empty cells == -1).
        Every non-empty cell is allowed.
        """
        flat = self.board.flatten()
        mask = (flat != -1)
        return mask


    def reset(self, *, seed=None, options=None):
        """
        Restart the environment and return the initial observation and action mask.
        """
        super().reset(seed=seed)
        self.np_random, _ = seeding.np_random(seed)
        self.game = FormerGame(self.N_ROWS, self.N_COLUMNS, self.N_SHAPES)
        self.board = np.array(self.game.get_board())
        self.action_count = 0

        # Observation is the board, build mask based on it
        obs = self.board
        one_hot = one_hot_encode(obs, self.N_SHAPES)
        mask = self.get_action_mask()

        return one_hot, {"action_mask": mask}

    def step(self, action):
        self.action_count += 1
        pt = divmod(action, self.N_COLUMNS)
        if self.board[pt[0],pt[1]] == -1:
            reward = -2
        else:
            group = FormerGame.find_group_static(self.board, pt)
            self.game.make_turn(pt)
            self.board = self.game.get_board()

            # Reward is -1 for each move
            terminated = bool(self.game.is_game_over())
            truncated = (self.action_count >= 100) # This cannot happen, but truncated must be returned
            reward = -1
            
        obs = self.board
        one_hot = one_hot_encode(obs, self.N_SHAPES)
        mask = self.get_action_mask()
        info = {
            "group_size": len(group),
            "actions_taken": self.action_count,
            "invalid_move": (len(group) == 0),
            "action_mask": mask,
        }

        return one_hot, reward, terminated, truncated, info

    def render(self, mode="human"):
        if mode == "human":
            print(f"Moves: {self.action_count}")
            print(self.board)

    def close(self):
        pass
    
def one_hot_encode(board: np.ndarray, S:int) -> np.ndarray:
    """
    One-hot-encode board (like psi in the thesis).
    """
    H, W = board.shape
    onehot = np.zeros((S+1, H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            v = board[i,j]
            if v < 0:
                onehot[S, i, j] = 1.0   # empty in last channel
            else:
                onehot[int(v), i, j] = 1.0
    return onehot
    
class CustomExtractorBig(BaseFeaturesExtractor):
    """
    This is the network architecture used in the final PPO model.
    """
    def __init__(self, obs_space, features_dim=256):
        super().__init__(obs_space, features_dim)
        in_ch = obs_space.shape[0]
        # your 15 conv layers @256 filters would go here; this is a 5-layer example at 128
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 256, kernel_size=7, padding=3), nn.BatchNorm2d(256), nn.ReLU(),
            *[nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ) for _ in range(4)]
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, in_ch, *obs_space.shape[1:])).shape[1]

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.linear(x)
    
    
    # To make MCTS work:
    # To make PPO work here:

N_ROWS, N_COLUMNS, N_SHAPES = 9, 7, 4

def one_hot_encode(board: np.ndarray, S: int) -> np.ndarray:
    H, W = board.shape
    onehot = np.zeros((S+1, H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            v = board[i, j]
            if v < 0:
                onehot[S, i, j] = 1.0   # empty channel
            else:
                onehot[int(v), i, j] = 1.0
    return onehot

def get_mask_from_board(board: np.ndarray) -> np.ndarray:
    """
    Exactly your FormerEnv mask: legal iff board[r,c] != -1
    Returns a boolean array of length N_ROWS*N_COLUMNS.
    """
    flat = board.flatten()
    return (flat != -1)

# Only use one PPO model, so we load it here to "hard-code" the functions below. I think this is more efficient than checking every time.
device = "cpu"
#actor_large = torch.jit.load("models/reinforcement/scripted_actor.pt", map_location=device)
#actor_large.eval()

def get_action_probs_from_board(board, model) -> np.ndarray:
    """
    Use PPO model to get action probabilities for the current board state.
    t = 1 for small model, t = 2 for large model.
    """
    obs = one_hot_encode(board, N_SHAPES)              # shape (5,9,7)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    obs_tensor = obs_tensor.unsqueeze(0)               # (1,5,9,7)

    with torch.no_grad():
        logits = model(obs_tensor).squeeze(0)          # (63,)
    
    # Print size of logits for debugging

    # Mask where there are no shapes
    mask = get_mask_from_board(board)                  # (63,) bool
    mask_tensor = torch.as_tensor(mask, device=device)#.unsqueeze(0)  # (1,63)
    logits[~mask_tensor] = float("-inf")

    # Get probabilities
    probs = F.softmax(logits, dim=-1)                  # (1,63)

    # Return probabilities over actions, to use with CPU
    probs_np = probs.cpu().numpy().reshape(N_ROWS, N_COLUMNS)
    return probs_np

def get_policy_PPO(board, model):
    """
    Get the policy for the current board state using the PPO model.
    """
    full_policy = get_action_probs_from_board(board, model)  # (9,7) float array

    reps = FormerGame.get_valid_turns_static(board)  # list of (r,c)

    # Strict masking over all but valid moves
    prob_dict: dict[tuple[int,int], float] = {}
    for rep in reps:
        cells = FormerGame.find_group_static(board, rep)
        p = sum(full_policy[r, c] for (r, c) in cells)
        prob_dict[rep] = p

    # Renormalize so they sum to 1
    total = sum(prob_dict.values())
    if total > 0:
        for rep in prob_dict:
            prob_dict[rep] /= total
    else:
        # Uniform if no legal moves are assigned probability
        uniform = 1.0 / len(prob_dict)
        for rep in prob_dict:
            prob_dict[rep] = uniform

    # Return probability dictionary
    return prob_dict

def load_ppo_models():
    """
    Load the PPO models for use in the game.
    """
    device = "cpu"
    actor_large = torch.jit.load("models/trained_models/reinforcement/scripted_actor.pt", map_location=device)
    actor_large.eval()
    
    critic_large = torch.jit.load("models/trained_models/reinforcement/scripted_critic.pt", map_location=device)
    critic_large.eval()
    
    actor_small = torch.jit.load("models/trained_models/reinforcement/scripted_actor_small.pt", map_location=device)
    actor_small.eval()
    
    critic_small = torch.jit.load("models/trained_models/reinforcement/scripted_critic_small.pt", map_location=device)
    critic_small.eval()
    
    ppo_nets = {
        "actor_large": actor_large,
        "critic_large": critic_large,
        "actor_small": actor_small,
        "critic_small": critic_small,
    }
    
    return ppo_nets

def get_recommended_action_ppo(board, model, type='actor'):
    """
    Get the recommended action from the PPO model for the current board state.
    """
    if type == 'actor':
        policy = get_policy_PPO(board, model)
        # Get the action with the highest probability
        best_action = max(policy.items(), key=lambda x: x[1])[0]
        return best_action, policy[best_action]
    elif type == 'critic':
        # For the critic, we evaluate each next state, and return the best action
        valid_turns = FormerGame.get_valid_turns_static(board)
        best_action = None
        best_value = float('inf')
        for action in valid_turns:
            next_board = np.array(FormerGame.apply_turn_static(board, action))
            value = evaluate_state_critic(next_board, model)
            if value < best_value:
                best_value = value
                best_action = action
        return best_action, best_value
    
    
# Helper functions for critic
# I just load the PPO models here.
def evaluate_state_critic(board, model) -> float:
    """
    Evaluates a board state with a PPO critic.
    """
    obs = one_hot_encode(board, N_SHAPES)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    obs_tensor = obs_tensor.unsqueeze(0)  # (1, C, H, W)
    with torch.no_grad():
        value = model(obs_tensor)        # → tensor of shape (1,)
    return -float(value.item()) # Negative since reward is negative number of moves remaining