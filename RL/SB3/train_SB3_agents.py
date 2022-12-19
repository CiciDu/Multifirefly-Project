from SB3_functions import SaveOnBestTrainingRewardCallback
from env import MultiFF
import random
import gym
from gym import spaces, Env
from gym.spaces import Dict, Box
import torch
import numpy as np
from numpy import pi
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import math
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.linalg import vector_norm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.data.dataset import Dataset, random_split
# For animation
from matplotlib import rc
rc('animation', html='jshtml')
matplotlib.rcParams['animation.embed_limit'] = 2**128
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)
from PIL import Image, ImageDraw, ImageOps
from IPython.display import Image as Image2
import os
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from typing import Any
from typing import Dict
from matplotlib import rc, cm


log_dir = "SB3_data/SB3_Aug_11/"
os.makedirs(log_dir, exist_ok=True)
env = MultiFF()
env = Monitor(env, log_dir)
env.reset()

sac_model = SAC("MlpPolicy",
            env,
            buffer_size=int(1e6),
            batch_size=1024,
            device='auto',
            verbose=False,
            train_freq=100,
            learning_starts = int(10),
            target_update_interval=20,
            learning_rate=1e-2,
            gamma=0.9999,
            policy_kwargs=dict(activation_fn=nn.ReLU, net_arch=[64, 64])
                )

'''
to retrieve model:
retrieve_dir = "SB3_data/SB3_Aug_11/"
path = os.path.join(retrieve_dir, 'best_model.zip')
path2 = os.path.join(retrieve_dir, 'buffer.pkl')
sac_model = sac_model.load(path,env=env) 
sac_model.load_replay_buffer(path2)
'''




# Train the agent
timesteps = 50000000
sac_model.learn(total_timesteps=int(timesteps), callback=callback)
plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Multiff")
plt.show()




