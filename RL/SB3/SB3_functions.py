import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib import rc, cm
import pandas as pd
import math
from math import pi
import collections
import re
import os, sys
from os.path import exists
import csv
from contextlib import contextmanager
from scipy.signal import decimate
import torch
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
from random import randint
from IPython.display import HTML

## for running RL agents
import gym
from gym import spaces, Env
from gym.spaces import Dict, Box
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.linalg import vector_norm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.data.dataset import Dataset, random_split
from PIL import Image, ImageDraw, ImageOps
from IPython.display import Image as Image2
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from typing import Any
from typing import Dict




def test_agent_for_plotting(obs, model, n_steps = 10000):
    # Test the trained agent
    obs = env.reset()
    mx = []
    my = []
    mheading = []  # in radians
    ffxy_all = []
    ffxy_visible = []
    ffxy2_all = []
    time_rl = []
    captured_ff = []
    num_targets = []
    env_obs = []
    visible_ff_indices_all = []
    memory_ff_indices_all = []
    monkey_speed = []
    obs_ff_indices_all = []
    obs_ff_overall_indices_all = []
    memory_all = []
    ff_angles2 = []
    ff_distances2 = []
    all_captured_ff_x = []
    all_captured_ff_y = []
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cum_rewards += reward
        previous_ffxy = env.ffxy
        prev_ff_information = env.ff_information.copy()
        obs, reward, done, info = env.step(action)
        reward_log.append(reward)
        num_targets.append(env.num_targets)
        memory_all.append(env.ff_memory_all)
        if env.num_targets > 0:
            captured_ff.append(env.captured_ff_index)
            all_captured_ff_x = all_captured_ff_x + previous_ffxy[env.captured_ff_index][:, 0].tolist()
            all_captured_ff_y = all_captured_ff_y + previous_ffxy[env.captured_ff_index][:, 1].tolist()
        else:
            captured_ff.append(0)
        mx.append(env.agentx.item())
        my.append(env.agenty.item())
        monkey_t.append(env.time)
        monkey_speed.append(env.dv.item())
        mheading.append(env.agentheading.item())
        ffxy_all.append(env.ffxy.clone())
        ffxy2_all.append(env.ffxy2.clone())
        ffxy_visible.append(env.ffxy[env.visible_ff_indices].clone())
        env_obs.append(obs)
        visible_ff_indices_all.append(env.visible_ff_indices)
        memory_ff_indices_all.append(env.ff_in_memory_indices)
        obs_ff_indices_all.append(env.topk_indices)
        real_indices = []
        for index in env.topk_indices:
            real_indices.append(
                int(prev_ff_information[:, 0][np.where(prev_ff_information[:, 7] == index.item())[0][-1]].copy()))
        obs_ff_overall_indices_all.append(real_indices)
        if len(env.topk_indices) > 0:
            ff_angles2.append(env.ff_angle_topk_2)
            ff_distances2.append(env.ff_distance_topk)
        else:
            ff_angles2.append(torch.tensor([]))
            ff_distances2.append(torch.tensor([]))
        if done:
            obs = env.reset()
        return mx, my, mheading, ffxy_all, ffxy_visible, ffxy2_all, time_rl, captured_ff, num_targets, env_obs, \
               visible_ff_indices_all, memory_ff_indices_all, monkey_speed, obs_ff_indices_all, \
               obs_ff_overall_indices_all, memory_all, ff_angles2, ff_distances2




def test_agent(obs, model, n_steps = 10000):
    # Test the trained agent
    obs = env.reset()
    cum_rewards = 0
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cum_rewards += reward
        if done:
            obs = env.reset()
        # print(step, ffxy_visible[-1])
    return cum_rewards



class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(" ")
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)
                  self.model.save_replay_buffer(log_dir+"buffer") # I added this

        return True