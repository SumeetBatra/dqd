import gym
import numpy as np
import pickle
import pandas

from attrdict import AttrDict
from models.actor_critic import Actor

if __name__ == '__main__':
    cfg = {'env_name': 'ant', 'env_batch_size': None, 'normalize_obs': False, 'normalize_rewards': True,
           'num_dims': 4, 'envs_per_model': 1, 'seed': 0}
    cfg = AttrDict(cfg)
    archive_path = '/home/sbatra/QDPPO/logs/qdppo_ant_brax/cma_mega/trial_0/archive_00000250.pkl'
    with open(archive_path, 'rb') as f:
        archive_df = pickle.load(f)
    elites = archive_df.query("objective > 6000").sort_values("objective", ascending=False)
    agent_params = elites.query('0').to_numpy()[5:]
    agent = Actor(cfg, obs_shape=(87,), action_shape=(8,)).deserialize(agent_params)
    pass
