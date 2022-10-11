import torch
import numpy as np
import sys

from RL.ppo import PPO
from RL.train_ppo import parse_args
from utils.utils import config_wandb


def train_with_measures():
    cfg = parse_args()

    if cfg.use_wandb:
        config_wandb(batch_size=cfg.batch_size, total_steps=cfg.total_timesteps, run_name=cfg.wandb_run_name)

    cfg.measure_coeffs = [0.1, 0.1, 0.1, 0.1]  # ant that moves really fast
    ppo = PPO(seed=cfg.seed, cfg=cfg)
    num_updates = cfg.total_timesteps // cfg.batch_size
    ppo.train(num_updates, rollout_length=cfg.rollout_length)


if __name__ == '__main__':
    train_with_measures()
    sys.exit(0)
