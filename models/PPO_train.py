# Move root one step out
import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from Former.Cpp_code.former_class_cpp import FormerGame
from tqdm import tqdm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from PPO_classes import FormerEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

"""
This script is what was used to train the PPO agents - model sizes are changed by changing
the CustomExtractorBig class. This was run on Idun with 1 GPU and 17 CPUs per task,
so some computational power is required for it to run in a reasonable time.
"""


# It is very infeasible to use anything else but GPU for this
if torch.cuda.is_available():
    DEVICE = "cuda"
    cudnn.benchmark = True
else:
    DEVICE = "cpu"

# Set 1 CPU per environment worker, plus 1 for learner
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Custom extractor (network)
class CustomExtractorBig(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=256):
        super().__init__(obs_space, features_dim)
        in_ch = obs_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=7, padding=3), nn.BatchNorm2d(32), nn.ReLU(),
            *[nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ) for _ in range(4)]
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, *obs_space.shape[1:])
            out = self.cnn(dummy)
            n_flatten = int(torch.prod(torch.tensor(out.shape[1:])))
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.linear(x)

# Directories (set via SLURM on Idun)
TENSORBOARD_LOGDIR = os.getenv(
    "TENSORBOARD_LOGDIR",
    "ppo_w32d5_final"
)
CHECKPOINT_DIR = os.getenv(
    "CHECKPOINT_DIR",
    "ppo_w32d5_final"
)
os.makedirs(TENSORBOARD_LOGDIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Determine number of parallel environments:
# reserve 1 core for learner, rest for env workers
total_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", "4"))
num_envs = max(1, total_cpus - 1)
# -----------------------------------------------------------------------------

def mask_fn(env) -> np.ndarray:
    """Retrieve the action mask from the (possibly wrapped) env."""
    base = env
    if hasattr(env, "envs"):
        base = env.envs[0]
    while hasattr(base, "env"):
        base = base.env
    return base.get_action_mask()

def make_env():
    """Factory for a single masked FormerEnv wrapped in Monitor."""
    env = FormerEnv(9, 7, 4)
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)
    return env

def linear_lr_schedule(progress_remaining: float) -> float:
    """
    Linearly anneal learning rate from 1e-3 to 0.
    """
    initial_lr = 3e-4
    final_lr   = 0
    return final_lr + progress_remaining * (initial_lr - final_lr)

class TBExtraCallback(BaseCallback):
    def _on_step(self) -> bool:
        lr = self.model.lr_schedule(self.num_timesteps)
        self.logger.record("custom/learning_rate_exact", lr)
        self.logger.record("custom/ent_coef", self.model.ent_coef)
        return True

def main():
    # 1) Create SubprocVecEnv for parallel CPU rollouts
    env_fns = [make_env for _ in range(num_envs)]
    train_env = SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        features_extractor_class=CustomExtractorBig,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
    )

    model = MaskablePPO(
        policy="CnnPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=1048,
        batch_size=256,
        device=DEVICE,
        tensorboard_log=TENSORBOARD_LOGDIR,
        learning_rate=linear_lr_schedule,
        ent_coef=0.0,
        target_kl=0.1,
    )

    # Checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_w32d5_new",
    )

    # Start training
    total_timesteps = 2_000_000_000
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="ppo_w32d5_new",
        callback=checkpoint_callback,
        reset_num_timesteps=False,
    )

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "ppo_w32d5_new")
    model.save(final_path)

if __name__ == "__main__":
    main()