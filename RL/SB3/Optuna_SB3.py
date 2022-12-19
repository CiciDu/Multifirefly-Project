from env import MultiFF

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
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


env = MultiFF()
env.reset()


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """
    def __init__(self, eval_env, trial, n_eval_episodes=5,
                 eval_freq=10000, deterministic=True, verbose=0):

        super(TrialEvalCallback, self).__init__(eval_env=eval_env, n_eval_episodes=n_eval_episodes,
                                                eval_freq=eval_freq,
                                                deterministic=deterministic,
                                                verbose=verbose)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(TrialEvalCallback, self)._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elasped time ?
            self.trial.report(-1 * self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

def sample_sac_params(trial):
    """
    Sampler for SAC hyperparams.
    :param trial: (optuna.trial)
    :return: (dict)
    """


    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    tau = trial.suggest_float("tau", 1e-6, 1, log=True)
    #batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512, 1024])
    target_update_interval = trial.suggest_categorical('target_update_interval', [5, 10, 20, 40, 60, 100, 200])
    #buffer_size = trial.suggest_categorical('buffer_size', [int(1e5), int(1e6)]) # This actually doesn't matter much here because of limited timesteps
    learning_starts = trial.suggest_categorical('learning_starts', [5000, 10000, 15000])
    train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
    # gradient_steps takes too much time
    # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    gradient_steps = train_freq
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    net_arch = trial.suggest_categorical('net_arch', ["small", "medium", "big"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh"])

    net_arch = {
        'small': [100, 100],
        'medium': [128, 128],
        'big': [200, 200],
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    target_entropy = 'auto'
    if ent_coef == 'auto':
        target_entropy = trial.suggest_categorical('target_entropy', ['auto', -1, -10, -20, -50, -100])



    ## Display true values
    # trial.set_user_attr("gamma_", gamma)
    # trial.set_user_attr("n_steps", n_steps)


    return {
        'gamma': gamma,
        'learning_rate': learning_rate,
        'tau': tau,
        #'batch_size': batch_size,
        'target_update_interval': target_update_interval,
        #'buffer_size': buffer_size,
        'learning_starts': learning_starts,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
        'ent_coef': ent_coef,
        'target_entropy': target_entropy,
        'policy_kwargs': {
            "net_arch": net_arch,
            "activation_fn": activation_fn
        }
    }



def objective(trial: optuna.Trial) -> float:
  kwargs = DEFAULT_HYPERPARAMS.copy()
  # Sample hyperparameters
  kwargs.update(sample_sac_params(trial))
  # Create the RL model
  model = SAC(**kwargs)
  # Create env used for evaluation
  eval_env = env
  # Create the callback that will periodically evaluate
  # and report the performance
  eval_callback = TrialEvalCallback(
      eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
  )

  nan_encountered = False
  try:
      model.learn(N_TIMESTEPS, callback=eval_callback)
  except AssertionError as e:
      # Sometimes, random hyperparams can generate NaN
      print(e)
      nan_encountered = True
  finally:
      # Free memory
      model.env.close()
      eval_env.close()

  # Tell the optimizer that the trial failed
  if nan_encountered:
      return float("nan")

  if eval_callback.is_pruned:
      raise optuna.exceptions.TrialPruned()

  return eval_callback.last_mean_reward






## run


N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = 2000
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 1



DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": env,
}

# Set pytorch num threads to 1 for faster training
torch.set_num_threads(1)

sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
# Do not prune before 1/3 of the max budget is used
pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
try:
    study.optimize(objective, n_trials=N_TRIALS)
except KeyboardInterrupt:
    pass

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

print("  User attrs:")
for key, value in trial.user_attrs.items():
    print("    {}: {}".format(key, value))