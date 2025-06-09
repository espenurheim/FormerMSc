import os
import pickle
import time
import torch
import optuna
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
import threading
import re
from supervised_networks import ValueNet, NetworkDataset, train_value_net

"""
This file contains the code used to perform hyperparameter tuning with the Optuna library,
on one of the networks. It was run on Idun, thus it may not run on other machines without modification.
"""

def load_data():
    loaded = []
    n_batches = 89
    for i in range(70, n_batches + 1):
        num = f"{i:03d}"
        path = f"/cluster/home/espenbur/masters_thesis/beam_search_data/training_data/batches/training_batch_{num}.pkl"
        with open(path, 'rb') as f:
            loaded.extend(pickle.load(f))
    split = int(0.9 * len(loaded))
    return NetworkDataset(loaded[:split], mode="value"), NetworkDataset(loaded[split:], mode="value")

# Load once
train_ds, val_ds = load_data()

# OPTUNA OBJECTIVE
def objective(trial):
    num_gpus = torch.cuda.device_count()

    # Try to get a per-worker index from the process name
    proc_name = mp.current_process().name  # e.g. "ForkPoolWorker-2" or "MainProcess"
    if proc_name.startswith("ForkPoolWorker"):
        worker_idx = int(proc_name.split('-')[-1]) - 1
    else:
        # Fallback: parse the current thread name
        thread_name = threading.current_thread().name  # e.g. "ThreadPoolExecutor-0_1"
        m = re.search(r'_(\d+)$', thread_name)
        worker_idx = int(m.group(1)) if m else 0

    gpu_id = worker_idx % num_gpus
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    d = 5
    w = 64
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    bs = trial.suggest_categorical("bs", [32, 64, 128, 256])
    wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)

    print(f"[Trial {trial.number}] d={d}, w={w}, lr={lr:.2e}, bs={bs}, wd={wd:.2e}")
    
    # Model + train
    model = ValueNet(5, (9,7), d, w).to(device)
    _, val_mse, _ = train_value_net(
        model,
        train_ds,
        test_set=val_ds,
        epochs=5,
        batch_size=bs,
        learning_rate=lr,
        weight_decay=wd,
        csv_path=None,
        checkpoint_folder=None
    )

    # minimize MSE
    return val_mse[-1] if val_mse else float('inf')

# Save trials callback
def save_trials(study, trial):
    df = study.trials_dataframe()
    df.to_csv("/cluster/home/espenbur/masters_thesis/optuna_value_w64d5.csv", index=False)

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    study = optuna.create_study(
        study_name="value_hpo",
        storage="sqlite:////cluster/home/espenbur/masters_thesis/optuna_value_w64d5.db",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    study.optimize(
        objective,
        n_trials=50,
        n_jobs=num_gpus,
        callbacks=[save_trials]
    )

    best = study.best_trial
    print("Best hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")