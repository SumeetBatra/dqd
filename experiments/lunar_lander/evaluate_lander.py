import gym
import numpy as np
import pickle
import pandas

from envs.qd_lunar_lander import QDLunarLanderEnv


def simulate(agent):
    env = gym.make('QDLunarLanderContinuous-v2')
    env.seed(0)
    obs = env.reset()
    all_y_vels = []
    impact_x_pos = None
    impact_y_vel = None
    done = False
    for step in range(1000):
        if done:
            obs = env.reset()
        act = obs @ agent
        obs, _, done, _ = env.step(act)
        x_pos = obs[0]
        y_vel = obs[3]
        leg0_touch = bool(obs[6])
        leg1_touch = bool(obs[7])

        if impact_x_pos is None and (leg0_touch or leg1_touch):
            impact_x_pos = x_pos
            impact_y_vel = y_vel
        print(f'{impact_x_pos=}, {impact_y_vel=}')
        print(f'{step=}')
        env.render()


if __name__ == '__main__':
    archive_path = './logs/qdppo_lander_emitters-5_num-steps-250_global-critic/cma_mega/trial_0/archive_00001000.pkl'
    with open(archive_path, 'rb') as f:
        archive_df = pickle.load(f)
    elites = archive_df.query("objective > 200").sort_values("objective", ascending=False)
    agent = elites.query('165').to_numpy()[5:].reshape((8, 2))
    simulate(agent)

