import os
import sys
import time

import gym
import csv
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

from dqd.ribs.archives import GridArchive
from dqd.ribs.emitters import GradientImprovementEmitter, GradientEmitter
from dqd.ribs.optimizers import Optimizer
from dqd.ribs.visualize import grid_archive_heatmap
from pathlib import Path

from RL.ppo import PPO
from RL.train_ppo import parse_args
from utils.utils import log, config_wandb
from models.actor_critic import Actor
from models.vectorized import VectorizedActor
from envs.cpu.vec_env import make_vec_env
from envs.brax_custom.gpu_env import make_vec_env_brax
from utils.normalize_obs import NormalizeReward, NormalizeObservation


def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0.0, vmax=100.0)
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())


def create_optimizer(cfg, algorithm, seed, num_emitters):
    """Creates an optimizer based on the algorithm name.

    Args:
        algorithm (str): Name of the algorithm passed into sphere_main.
        dim (int): Dimensionality of the sphere function.
        seed (int): Main seed or the various components.
    Returns:
        Optimizer: A ribs Optimizer for running the algorithm.
    """
    # fraction contact time with ground for each of the 4 legs
    bounds = [
        (0., 1.0),
        (0., 1.0),
        (0., 1.0),
        (0., 1.0),
    ]
    obs_shape, action_shape = (28,), (8,)  # TODO: fix this
    action_dim, obs_dim = np.prod(action_shape), np.prod(obs_shape)
    batch_size = cfg.mega_lambda

    initial_agent = Actor(cfg, obs_shape, action_shape)
    initial_sol = initial_agent.serialize()

    # Create archive.
    if algorithm in [
        "map_elites", "map_elites_line", "cma_me_imp",
        "og_map_elites", "og_map_elites_line",
        "og_map_elites_ind", "og_map_elites_line_ind",
        "omg_mega", "cma_mega", "cma_mega_adam",
    ]:
        archive = GridArchive(
            dims=[6, 6, 6, 6],
            ranges=bounds,
            seed=seed
        )
    else:
        raise ValueError(f"{algorithm=} is not recognized")

    # Create emitters. Each emitter needs a different seed, so that they do not
    # all do the same thing.
    emitter_seeds = [None] * num_emitters if seed is None else list(
        range(seed, seed + num_emitters))
    # emitters for cma-mega
    if algorithm in ['cma_mega']:
        emitters = [
            GradientImprovementEmitter(archive,
                                       initial_sol,
                                       sigma_g=0.05,
                                       stepsize=0.001,
                                       gradient_optimizer="adam",
                                       normalize_gradients=True,
                                       selection_rule="mu",
                                       restart_rule='basic',
                                       bounds=None,
                                       batch_size=batch_size,
                                       seed=s) for s in emitter_seeds
        ]
    elif algorithm in ['omg_mega']:
        emitters = [
            GradientEmitter(archive,
                            initial_sol,
                            sigma_g=0.05,
                            normalize_gradients=True,
                            bounds=None,
                            batch_size=batch_size,
                            seed=s) for s in emitter_seeds
        ]

    return Optimizer(archive, emitters)


def run_experiment(cfg,
                   algorithm,
                   ppo,
                   trial_id,
                   dim=1000,
                   init_pop=100,
                   itrs=10000,
                   outdir="./logs",
                   log_freq=1,
                   log_arch_freq=1000,
                   seed=None,
                   use_wandb=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create a directory for this specific trial.
    s_logdir = os.path.join(outdir, f"{algorithm}", f"trial_{trial_id}")
    logdir = Path(s_logdir)
    if not logdir.is_dir():
        logdir.mkdir()

    # create a new summary file
    summary_filename = os.path.join(s_logdir, f'summary.csv')
    if os.path.exists(summary_filename):
        os.remove(summary_filename)
    with open(summary_filename, 'w') as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(['Iteration', 'QD-Score', 'Coverage', 'Maximum', 'Average'])

    # cma_mega - specific params
    is_init_pop = False
    is_dqd = True

    optimizer = create_optimizer(cfg, algorithm, seed, cfg.num_emitters)
    archive = optimizer.archive

    best = 0.0
    non_logging_time = 0.0

    obs_shape = ppo.vec_env.single_observation_space.shape
    action_shape = ppo.vec_env.single_action_space.shape

    reward_normalizer = None

    for itr in range(1, itrs + 1):
        itr_start = time.time()

        if is_dqd:
            # returns a single sol per emitter
            sols = optimizer.ask(grad_estimate=True)
            agents = [Actor(cfg, obs_shape, action_shape).deserialize(sol).to(device) for sol
                      in sols]

            if reward_normalizer is None:
                reward_normalizer = agents[0].reward_normalizer
            else:
                agents[0].reward_normalizer = reward_normalizer

            vec_agent = VectorizedActor(cfg, agents, Actor, obs_shape=obs_shape,
                                        action_shape=action_shape).to(device)
            ppo.agents = agents
            ppo.vec_inference = vec_agent
            objs, obj_grads, measures, measure_grads = ppo.train(num_updates=10, rollout_length=ppo.cfg.rollout_length)

            reward_normalizer = agents[0].reward_normalizer

            best = max(best, max(objs))
            obj_grads = np.expand_dims(obj_grads, axis=1)
            jacobian = np.concatenate((obj_grads, measure_grads), axis=1)
            optimizer.tell(objs, measures, jacobian=jacobian)

        # gets sols around the current solution point by varying coeffs of grad_f and grad_m's
        # i.e. these are nn-agents
        sols = optimizer.ask()
        agents = [Actor(cfg, obs_shape, action_shape).deserialize(sol).to(device) for sol in
                  sols]

        vec_agent = VectorizedActor(cfg, agents, Actor, obs_shape=obs_shape,
                                    action_shape=action_shape, measure_dims=cfg.num_dims).to(device)
        objs, measures = ppo.evaluate(vec_agent, ppo.multi_eval_env)
        best = max(best, max(objs))
        optimizer.tell(objs, measures)
        non_logging_time += time.time() - itr_start
        log.debug(f'{itr=}, {itrs=}, Progress: {(100.0 * (itr / itrs)):.2f}%')

        # Save the archive at the given frequency.
        # Always save on the final iteration.
        final_itr = itr == itrs
        if (itr > 0 and itr % log_arch_freq == 0) or final_itr:
            # Save a full archive for analysis.
            df = archive.as_pandas(include_solutions=final_itr)
            df.to_pickle(os.path.join(s_logdir, f"archive_{itr:08d}.pkl"))

            # Save a heatmap image to observe how the trial is doing.
            save_heatmap(archive, os.path.join(s_logdir, f"heatmap_{itr:08d}.png"))

            # Update the summary statistics for the archive
            if (itr > 0 and itr % log_freq == 0) or final_itr:
                with open(summary_filename, 'a') as summary_file:
                    writer = csv.writer(summary_file)

                    sum_obj = 0
                    num_filled = 0
                    num_bins = archive.bins
                    for sol, obj, beh, idx, meta in zip(*archive.data()):
                        num_filled += 1
                        sum_obj += obj
                    qd_score = sum_obj / num_bins
                    average = sum_obj / num_filled
                    coverage = 100.0 * num_filled / num_bins
                    data = [itr, qd_score, coverage, best, average]
                    writer.writerow(data)

        if use_wandb:
            sum_obj = 0
            num_filled = 0
            num_bins = archive.bins
            for sol, obj, beh, idx, meta in zip(*archive.data()):
                num_filled += 1
                sum_obj += obj
            qd_score = sum_obj / num_bins
            average = sum_obj / num_filled
            coverage = 100.0 * num_filled / num_bins
            wandb.log({
                "QD/QD Score": qd_score,
                "QD/average performance": average,
                "QD/coverage (%)": coverage,
                "QD/best score": best,
            })


def ant_main(cfg,
             algorithm,
             ppo,
             trials=20,
             dim=1000,
             init_pop=100,
             itrs=10000,
             outdir="logs",
             log_freq=1,
             log_arch_freq=1000,
             seed=None,
             use_wandb=False):
    """Experimental tool for the planar robotic arm experiments.

    Args:
        algorithm (str): Name of the algorithm.
        trials (int): Number of experimental trials to run.
        dim (int): Dimensionality of solutions.
        init_pop (int): Initial population size for MAP-Elites (ignored for CMA variants).
        itrs (int): Iterations to run.
        outdir (str): Directory to save output.
        log_freq (int): Number of iterations between computing QD metrics and updating logs.
        log_arch_freq (int): Number of iterations between saving an archive and generating heatmaps.
        seed (int): Seed for the algorithm. By default, there is no seed.
    """
    # Create a shared logging directory for the experiments for this algorithm.
    s_logdir = os.path.join(outdir, f"{algorithm}")
    logdir = Path(s_logdir)
    outdir = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()
    if not logdir.is_dir():
        logdir.mkdir()

    for i in range(trials):
        run_experiment(
            cfg,
            algorithm,
            ppo,
            i,
            dim,
            init_pop,
            itrs,
            outdir,
            log_freq,
            log_arch_freq,
            seed,
            use_wandb
        )


if __name__ == '__main__':
    cfg = parse_args()
    cfg.num_emitters = 1

    if cfg.env_type == 'cpu':
        vec_env = make_vec_env(cfg)
        cfg.batch_size = int(cfg.num_workers * cfg.envs_per_worker * cfg.rollout_length)
        cfg.num_envs = int(cfg.num_workers * cfg.envs_per_worker)
        cfg.envs_per_model = cfg.num_envs // cfg.num_emitters
    elif cfg.env_type == 'brax':
        vec_env = make_vec_env_brax(cfg)
        cfg.batch_size = int(cfg.env_batch_size * cfg.rollout_length)
        cfg.num_envs = int(cfg.env_batch_size)
        cfg.envs_per_model = cfg.num_envs // cfg.num_emitters
    else:
        raise NotImplementedError(f'{cfg.env_type} is undefined for "env_type"')

    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.obs_shape = vec_env.single_observation_space.shape
    cfg.action_dim = vec_env.single_action_space.shape

    ppo = PPO(seed=cfg.seed, cfg=cfg, vec_env=vec_env)
    if cfg.use_wandb:
        config_wandb(batch_size=cfg.batch_size, total_steps=cfg.total_timesteps, run_name=cfg.wandb_run_name)
    outdir = './logs/xyz/'
    assert not os.path.exists(outdir), "Warning: this dir exists. Danger of overwriting previous run"
    os.mkdir(outdir)

    obs_shape = ppo.vec_env.single_observation_space.shape
    action_shape = ppo.vec_env.single_action_space.shape
    dummy_agent_params = Actor(cfg, obs_shape, action_shape).serialize()
    dims = len(dummy_agent_params)

    ant_main(
        cfg,
        algorithm='cma_mega',
        ppo=ppo,
        trials=1,
        dim=dims,
        init_pop=1,
        itrs=1000,
        outdir=outdir,
        seed=0,
        use_wandb=cfg.use_wandb
    )
    sys.exit(0)
