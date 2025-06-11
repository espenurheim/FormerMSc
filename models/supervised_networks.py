# Move root one step out
import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import search.beam_search as bs
import time
import numpy as np
import Former.daily_board as db
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from Former.Cpp_code.former_class_cpp import FormerGame
import os
from tqdm import tqdm
import csv
from pathlib import Path

class NetworkDataset(Dataset):
    """
    Class to handle datasets.
    """
    def __init__(self, data, mode="dual", S=4):
        """
        data: list of (board_state, action, value) tuples
        mode: one of "value", "policy", or "dual"
        S: number of shapes
        """
        assert mode in ("value", "policy")
        self.data      = data
        self.mode      = mode
        self.S         = S
        self.empty_val = -1
    def __len__(self):
        return len(self.data)

    def _one_hot(self, board):
        rows, cols = board.shape
        flat       = board.flatten()
        channels   = np.where(flat == self.empty_val, self.S, flat).astype(np.int64)
        oh_flat    = np.eye(self.S+1, dtype=np.float32)[channels]
        return torch.from_numpy(oh_flat.T.reshape(self.S+1, rows, cols))

    def __getitem__(self, idx):
        board, action, value = self.data[idx]
        onehot = self._one_hot(np.asarray(board, dtype=np.int64))

        if self.mode == "value":
            # (state, value)
            return onehot, torch.tensor([value], dtype=torch.float32)

        elif self.mode == "policy":
            if isinstance(action, (int, float)):
                return onehot, torch.tensor([action], dtype=torch.float32)
            return onehot, torch.tensor(action, dtype=torch.int64)

# VALUE NET
class ValueNet(nn.Module):
    """
    Class for value networks.
    """
    def __init__(self, input_channels, board_shape, d=5, w=64):
        super().__init__()
        self.board_shape = board_shape
        self.d = d
        self.w = w

        # First convolutional block (ReLU done in forward)
        self.conv1 = nn.Conv2d(input_channels, w, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm2d(w)

        # Convolutional blocks
        for i in range(2, d+1):
            setattr(self, f'conv{i}', nn.Conv2d(w, w, kernel_size=3, padding=1))
            setattr(self, f'bn{i}',   nn.BatchNorm2d(w))

        # GAP + FC
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(w, 1)

    def forward(self, x):
        # Initial conv -> BN -> ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # d-1 middle blocks: conv -> BN -> ReLU
        for i in range(2, self.d+1):
            conv = getattr(self, f'conv{i}')
            bn   = getattr(self, f'bn{i}')
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        # Global pooling -> linear regression head
        x = self.gap(x)            # (B, w, 1, 1)
        x = x.view(x.size(0), -1)  # (B, w)
        x = self.fc1(x)            # (B, 1)
        return x
    
    def evaluate_state(self, board):
        """
        Evaluates a given board state using the value network.
        
        Parameters:
            board (np.array): A numpy array representing the board with shape (rows, cols).
                              It should contain integer values, with -1 representing an empty cell.
                              
        Returns:
            value (float): The network's estimated minimal number of moves remaining.
        """
        rows, cols = board.shape
        S = 4
        onehot = np.zeros((S+1, rows, cols), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                val = board[i, j]
                if val == -1:
                    onehot[S, i, j] = 1.0
                else:
                    onehot[int(val), i, j] = 1.0

        input_tensor = torch.tensor(onehot).unsqueeze(0)
        input_tensor = input_tensor.to(next(self.parameters()).device)
        
        self.eval()
        with torch.no_grad():
            output = self(input_tensor)
        return output.item()
    
    def get_recommended_action(self, board, valid_turns = None):
        """
        A helper function to get the recommended action based on the value network, through
        look-ahead evaluations of each state after applying each valid action.
        
        Returns:
            best_action (tuple): The recommended action as a (row, col) tuple.
        """
        if valid_turns is None:
            valid_turns = FormerGame.get_valid_turns_static(board)
        
        # Evaluate each next state, and return the one with the lowest value
        best_value = float('inf')
        best_action = None
        for turn in valid_turns:
            next_board = np.array(FormerGame.apply_turn_static(board, turn))
            value = self.evaluate_state(next_board)
            if value < best_value:
                best_value = value
                best_action = turn
        return best_action, best_value

def train_value_net(
    model,
    dataset,
    test_set=None,
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=0.0,
    csv_path=None,
    checkpoint_folder=None,
):
    """
    Function for training a value network. Parameters used are given in the notebook.
    """
    device = next(model.parameters()).device
    print(f"Using device {device}")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False) if test_set is not None else None

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    mse_loss  = nn.MSELoss()
    mae_loss  = nn.L1Loss()

    # CSV header
    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch','train_loss','val_mse','val_mae','epoch_time_s'])

    train_losses, val_mse_list, val_mae_list = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        running = 0.0
        for inputs, targets in tqdm(train_loader,
                                    desc=f"Train {epoch}/{epochs}",
                                    leave=False,
                                    position=1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * inputs.size(0)

        train_loss = running / len(dataset)
        train_losses.append(train_loss)

        val_mse, val_mae = None, None
        if test_loader:
            model.eval()
            msum, asum = 0.0, 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    out = model(inputs)
                    msum += mse_loss(out, targets).item() * inputs.size(0)
                    asum += mae_loss(out, targets).item() * inputs.size(0)
            val_mse = msum / len(test_set)
            val_mae = asum / len(test_set)
            val_mse_list.append(val_mse)
            val_mae_list.append(val_mae)
        elapsed = time.time() - start

        print(f"Epoch {epoch}/{epochs} — "
              f"Train L={train_loss:.4f} "
              f"Val MSE={val_mse:.4f} MAE={val_mae:.4f} "
              f"({elapsed:.1f}s)")

        if checkpoint_folder:
            os.makedirs(checkpoint_folder, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_folder, f"epoch_{epoch}.pth"))

        if csv_path:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch,f"{train_loss:.6f}",f"{val_mse:.6f}",f"{val_mae:.6f}",f"{elapsed:.3f}"])

    return train_losses, val_mse_list, val_mae_list

# POLICY NET
def accumulate_probabilities(board, full_p, valid_turns):
    """
    Helper function to mask probabilities so that only valid turns are given a probability.
    Args:
        board:  2D array-like of shape (R, C), with -1 for empty cells and non-negative ints for shapes.
        full_p: 1D torch.Tensor of length R*C containing the network’s softmax output
                (before any masking). full_p[i] is the prob for flat index i = r*C + c.

    Returns:
        A dict mapping each valid turn (r, c) to the total probability mass
        assigned to its entire shape-group, normalized so the values sum to 1.
    """
    valid_set   = set(valid_turns)

    R, C = board.shape
    probs = {}

    # Iterate over valid turns and accumulate probabilities from each group
    for turn in valid_turns:
        group_pts = FormerGame.find_group_static(board, turn)
        idxs = [
            rr * C + cc
            for (rr, cc) in group_pts
            if (rr, cc) in valid_set
        ]
        if idxs:
            mass = full_p[idxs].sum().item()
        else:
            mass = 0.0
        probs[turn] = mass

    # Normalize sum to 1
    total = sum(probs.values())
    if total > 0:
        for turn in probs:
            probs[turn] /= total
    else:
        # If no valid turns are assigned probabilities, assign uniform distribution
        uni = 1.0 / len(valid_turns)
        for turn in probs:
            probs[turn] = uni

    return probs

class PolicyNet(nn.Module):
    """
    Class for policy networks.
    """
    def __init__(self, input_channels, board_shape, d=5, w=64):
        super().__init__()
        self.board_shape = board_shape
        self.d = d
        self.w = w

        # First convolution + batch norm
        self.conv1 = nn.Conv2d(input_channels, w, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(w)

        # Middle convolutions + batch norms
        for i in range(2, d+1):
            setattr(self, f'conv{i}', nn.Conv2d(w, w, kernel_size=3, padding=1))
            setattr(self, f'bn{i}', nn.BatchNorm2d(w))

        # 1x1 convolution head for per-cell policy logits
        self.policy_head = nn.Conv2d(w, 1, kernel_size=1)

    def forward(self, x):
        # Initial conv -> BN -> ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # d-1 middle blocks: conv -> BN -> ReLU
        for i in range(2, self.d+1):
            conv = getattr(self, f'conv{i}')
            bn = getattr(self, f'bn{i}')
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        # 1x1 conv head -> flatten (SoftMax done in evaluate_state)
        p = self.policy_head(x)            # (B, 1, R, C)
        p = p.view(p.size(0), -1)          # (B, R*C)
        return p
    
    def evaluate_state(self, board):
        """
        Single-board inference: returns
          policy (dict mapping legal (r,c)->prob)
        """
        R, C = board.shape
        S = 4
        # one‐hot encode channels = S+1
        onehot = np.zeros((S+1, R, C), dtype=np.float32)
        for r in range(R):
            for c in range(C):
                v = int(board[r,c])
                onehot[S if v<0 else v, r, c] = 1.0

        # Evaluate the board state. Apply softmax to get probabilities.
        x = torch.from_numpy(onehot)[None].to(next(self.parameters()).device)
        self.eval()
        with torch.no_grad():
            full_p = F.softmax(self(x), dim=1)
        full_p = full_p.squeeze(0)
        valid = FormerGame.get_valid_turns_static(board)
        policy = accumulate_probabilities(board, full_p, valid)
        return policy
    
    def get_recommended_action(self, board):
        """
        A helper function to get the recommended action based on the policy network.
        """
        
        policy = self.evaluate_state(board)
        best_action, prob = max(policy.items(), key=lambda item: item[1])
        return best_action, prob

def train_policy_net(model,
                     dataset,
                     test_set=None,
                     epochs=10,
                     batch_size=32,
                     learning_rate=1e-3):
    """
    Function to train policy networks. Parameters used are given in the notebook.
    """
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False) if test_set is not None else None

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Cross-entropy is log_softmax + NLLLoss (negative log-likelihood) - this implementation is (apparently) more efficient
    policy_crit = nn.NLLLoss() 

    train_policy_losses, val_policy_losses = [], []

    for epoch in range(1, epochs+1):
        # Train
        model.train()
        total_pol, n_samples = 0.0, 0
        t0 = time.time()

        for onehot, best_act in tqdm(loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            B, C_in, R, C = onehot.shape

            acts = best_act.squeeze(1)
            rows, cols = acts[:,0], acts[:,1]
            tgt_idx = (rows * C + cols).to(device).long()

            optimizer.zero_grad()
            p_logits = model(onehot.to(device))   # Evaluate network on batch
            logp = F.log_softmax(p_logits, dim=1) # Apply softmax to get log probabilities
            loss_p = policy_crit(logp, tgt_idx)   # Calculate loss

            loss_p.backward()
            optimizer.step()

            total_pol += loss_p.item() * B
            n_samples += B

        train_loss = total_pol / n_samples
        train_policy_losses.append(train_loss)
        t_train = time.time() - t0
        torch.save(model.state_dict(), f"checkpoint_{epoch}.pth")

        # Test on validation data
        if test_loader is not None:
            model.eval()
            total_vpol, n_val = 0.0, 0
            with torch.no_grad():
                for onehot, best_act in tqdm(test_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                    B, C_in, R, C = onehot.shape
                    acts = best_act.squeeze(1)
                    rows, cols = acts[:,0], acts[:,1]
                    tgt_idx = (rows * C + cols).to(device).long()

                    p_logits = model(onehot.to(device))
                    logp = F.log_softmax(p_logits, dim=1)
                    loss_p = policy_crit(logp, tgt_idx)

                    total_vpol += loss_p.item() * B
                    n_val += B

            val_loss = total_vpol / n_val
            val_policy_losses.append(val_loss)
            print(f"Epoch {epoch}/{epochs} | Train P: {train_loss:.4f} | Val P: {val_loss:.4f} ({t_train:.1f}s)")
        else:
            print(f"Epoch {epoch}/{epochs} | Train P: {train_loss:.4f} ({t_train:.1f}s)")

    return train_policy_losses, val_policy_losses


# Compute project root based on this file's location
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

# Base directory where pretrained weights are stored
WEIGHTS_DIR = PROJECT_ROOT / "models" / "trained_models" / "supervised"

# Ensure WEIGHTS_DIR exists for sanity
if not WEIGHTS_DIR.exists():
    raise FileNotFoundError(f"Weights directory not found: {WEIGHTS_DIR}")


def load_networks():
    """
    Load all value and policy networks and return them in a dict with custom keys:
      - 'v_<name>' for value networks
      - 'pi_<name>' for policy networks
    """
    nets = {}

    # Specifications: (folder_name, depth, width)
    value_specs = [
        ("w32d5", 5, 32),
        ("w32d10", 10, 32),
        ("w64d5", 5, 64),
        ("w64d10", 10, 64),
    ]
    # Load value networks
    for name, depth, width in value_specs:
        net = ValueNet(5, (9,7), depth, width)
        path = WEIGHTS_DIR / "value" / name / f"{name}.pth"
        state = torch.load(path, map_location=torch.device("cpu"))
        net.load_state_dict(state)
        net.eval()
        nets[f"v_{name}"] = net

    # Specifications for policy networks
    policy_specs = [
        ("w32d5", 5, 32),
        ("w32d10", 10, 32),
        ("w64d5", 5, 64),
        ("w64d10", 10, 64),
    ]
    # Load policy networks
    for name, depth, width in policy_specs:
        net = PolicyNet(5, (9,7), depth, width)
        path = WEIGHTS_DIR / "policy" / name / f"{name}.pth"
        state = torch.load(path, map_location=torch.device("cpu"))
        net.load_state_dict(state)
        net.eval()
        nets[f"pi_{name}"] = net

    return nets

