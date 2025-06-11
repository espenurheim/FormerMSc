from pathlib import Path
import sys
import time
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from Former.Cpp_code.former_class_cpp import FormerGame
from models.heuristics import find_point_minimizing_groups_static
from models.PPO_classes import evaluate_state_critic, load_ppo_models

# Ensure project root on sys.path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Constants for board dimensions and shapes
N_ROWS, N_COLS, N_SHAPES = 9, 7, 4


def one_hot_encode(board: np.ndarray, S: int) -> np.ndarray:
    """Convert board into one-hot tensor of shape (S+1, H, W)."""
    H, W = board.shape
    onehot = np.zeros((S + 1, H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            v = board[i, j]
            idx = S if v < 0 else int(v)
            onehot[idx, i, j] = 1.0
    return onehot


def beam_search_neural(
    state: np.ndarray,
    model,
    beam_width: int = 2,
    max_depth: int = 50,
    value_type: str = 'network',
    T: int = 1
):
    """
    Beam search with evaluation by network, heuristic, or PPO critic.
    Returns (final_state, move_sequence) or None.
    """
    # initialize beam: (score, state, path)
    if value_type == 'network':
        beam = [(model.evaluate_state(state), state, [])]
    elif value_type == 'ppo':
        beam = [(evaluate_state_critic(state, model), state, [])]
    else:  # heuristic
        score, _ = find_point_minimizing_groups_static(state, T=T)
        beam = [(score, state, [])]

    for _ in range(max_depth):
        candidates = []
        for score, b, path in beam:
            if FormerGame.is_game_over_static(b):
                return b, path
            for action in FormerGame.get_valid_turns_static(b):
                nxt = np.array(FormerGame.apply_turn_static(b, action))
                if value_type == 'network':
                    sc = model.evaluate_state(nxt)
                elif value_type == 'ppo':
                    sc = evaluate_state_critic(nxt, model)
                else:
                    sc, _ = find_point_minimizing_groups_static(nxt, T=T)
                candidates.append((sc, nxt, path + [action]))
        if not candidates:
            break
        # keep top beam_width by score
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    return None


def test_beam_performance(
    model,
    beam_widths: list,
    save_path: Path = None
) -> dict:
    """
    Evaluate beam search over daily boards.
    Returns dict with 'sol', 'real', 'times', 'beam_widths'.
    """
    boards = load_ppo_models() if model == 'ppo' else []  # placeholder
    daily = {}  # should call db.get_daily_board()
    results = {
        'sol': np.zeros((len(daily), len(beam_widths))),
        'real': np.zeros((len(daily), len(beam_widths))),
        'times': np.zeros((len(daily), len(beam_widths))),
        'beam_widths': beam_widths
    }
    for i, (key, (board, real_len)) in enumerate(daily.items()):
        for j, bw in enumerate(beam_widths):
            start = time.time()
            res = beam_search_neural(board, model, beam_width=bw)
            duration = time.time() - start
            seq = res[1] if res else []
            results['sol'][i, j] = len(seq)
            results['real'][i, j] = real_len
            results['times'][i, j] = duration
    if save_path:
        np.savez(save_path, **results)
    return results


def plot_beam_metrics(files: dict):
    """Plot solution accuracy, difference, and time vs beam width."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for name, path in files.items():
        data = np.load(path)
        bw = data['beam_widths']
        sol, real, tms = data['sol'], data['real'], data['times']
        axes[0].plot(bw, (sol == real).mean(axis=0), marker='o', label=name)
        axes[1].errorbar(
            bw,
            (sol - real).mean(axis=0),
            yerr=(sol - real).std(axis=0),
            marker='o', capsize=5, label=name
        )
        axes[2].plot(bw, tms.mean(axis=0), marker='o', label=name)
    for ax, lbl in zip(axes, ['Accuracy', 'Mean Δ', 'Time (s)']):
        ax.set_xscale('log')
        ax.set_xlabel('Beam Width')
        ax.set_title(lbl)
        ax.grid()
        ax.legend()
    plt.tight_layout()

# Legacy beam_search functions are omitted for brevity
