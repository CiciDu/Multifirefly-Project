import sys
from RL.SB3.SB3_collect_data import data_from_SB3
from functions.process_raw_data import*
from functions.find_patterns import*
from data.manufactured_data.pattern_data import stats_dict2, stats_dict_median2, stats_dict3, stats_dict_median3, stats_dict_m, stats_dict_median_m
#from data_processing import*
from functions.plotting_func import*


import ffmpeg
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
plt.rcParams["animation.html"] = "html5"
print("done")
trial_total_num = 3




# c16.2

log_dir = "RL/SB3/SB3_data/Aug_10_3"
retrieve_dir = "RL/SB3/SB3_data/Aug_10_3/"

# obs = [angle, distance, memory, angle2, distance2, memory2, ...]
# before = [angle, angle2 ..., distance, distance2, ... memory, memory2]

class MultiFF(Env):
    def __init__(self):
        super(MultiFF, self).__init__()
        self.num_ff = 200
        self.arena_radius = 1000
        self.episode_len = 16000
        self.dt = 0.25
        # self.current_episode = 0
        # self.action_space = spaces.Box(low=-1., high=1., shape=(2,),dtype=np.float32)
        self.action_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.obs_ff = 2
        self.observation_space = spaces.Box(low=-1., high=1., shape=(self.obs_ff * 4,), dtype=np.float32)
        self.terminal_vel = 0.01
        # self.pro_noise_std = 0.005
        self.vgain = 200
        self.wgain = pi / 2
        self.reward_per_ff = 100
        # self.time_cost = 0.005
        # self.total_time = 0
        # self.zero_action = False
        # self.target_update_counter_num = 20
        # self.reward_per_episode = []
        # self.update_slots = True
        # self.closest_ff_distance = 200
        self.pro_noise_std = 0.005
        self.epi_num = -1
        self.has_sped_up_before = False
        self.full_memory = 3
        self.internal_noise_factor = 3
        self.ff_memory_all = torch.ones([self.num_ff, ]) * self.full_memory
        self.invisible_distance = 400
        self.invisible_angle = 2 * pi / 9
        self.reward_boundary = 25

    def reset(self):
        self.epi_num += 1
        print("\n episode: ", self.epi_num)
        self.num_targets = 0
        # self.past_speeds = []
        self.time = 0
        # self.counter = 0
        # self.current_target_index = torch.tensor([999], dtype=torch.int32)
        # self.previous_target_index = self.current_target_index
        self.ff_flash = []
        self.has_sped_up_before = False

        for i in range(self.num_ff):
            num_intervals = 1500
            first_flash = torch.rand(1)
            intervals = torch.poisson(torch.ones(num_intervals - 1) * 3)
            t0 = torch.cat((first_flash, first_flash + torch.cumsum(intervals, dim=0) + torch.cumsum(
                torch.ones(num_intervals - 1) * 0.3, dim=0)))
            t1 = t0 + torch.ones(num_intervals) * 0.3
            self.ff_flash.append(torch.stack((t0, t1), dim=1))

        self.ffr = torch.sqrt(
            torch.rand(self.num_ff)) * self.arena_radius  # The radius of the arena changed from 1000 cm/s to 200 cm/s
        self.fftheta = torch.rand(self.num_ff) * 2 * pi
        # self.ffrt = torch.stack((self.ffr, self.fftheta), dim=1)
        self.ffx = torch.cos(self.fftheta) * self.ffr
        self.ffy = torch.sin(self.fftheta) * self.ffr
        self.ffxy = torch.stack((self.ffx, self.ffy), dim=1)
        self.ffx2 = self.ffx.clone()
        self.ffy2 = self.ffy.clone()
        self.ffxy2 = torch.stack((self.ffx2, self.ffy2), dim=1)
        # self.ff_info={}
        # self.update_slots = True
        self.agentx = torch.tensor([0])
        self.agenty = torch.tensor([0])
        self.agentr = torch.zeros(1)
        self.agentxy = torch.tensor([0, 0])
        self.agentheading = torch.zeros(1).uniform_(0, 2 * pi)
        self.dv = torch.zeros(1).uniform_(-0.05, 0.05)
        self.dw = torch.zeros(1)
        self.end_episode = False
        self.obs = self.beliefs().numpy()
        # self.chunk_50s = 1
        self.episode_reward = 0
        # self.stop_rewarding_speed = False
        # self.current_obs_steps = 0
        self.ff_memory_all = torch.ones([self.num_ff, ])
        return self.obs

    def calculate_reward(self):
        # action_cost=((self.previous_action[1]-self.action[1])**2+(self.previous_action[0]-self.action[0])**2)*self.mag_cost
        # To incorporate action_cost, we need to incorporate previous_action into decision_info
        # self.total_time += 1
        # In addition to rewarding the monkey for capturing the firefly, we also use different phases of rewards to teach monkey specific behaviours
        # reward = -self.time_cost
        reward = 0
        # Reward shaping
        # Phase I: reward the agent for learning to stop
        # Always: reward the agent for capturing fireflies
        self.num_targets = 0
        if abs(self.sys_vel[1]) <= self.terminal_vel:
            captured_ff_index = (self.ff_distance_all <= self.reward_boundary).nonzero().reshape(-1).tolist()
            self.captured_ff_index = captured_ff_index
            num_targets = len(captured_ff_index)
            self.num_targets = num_targets
            if num_targets > 0:  # If the monkey hs captured at least 1 ff
                # Calculate reward
                reward = reward + self.reward_per_ff * num_targets
                # Replace the captured ffs with ffs of new locations
                self.ffr[captured_ff_index] = torch.sqrt(torch.rand(num_targets)) * self.arena_radius
                self.fftheta[captured_ff_index] = torch.rand(num_targets) * 2 * pi
                # self.ffrt = torch.stack((self.ffr, self.fftheta), dim=1)
                self.ffx[captured_ff_index] = torch.cos(self.fftheta[captured_ff_index]) * self.ffr[captured_ff_index]
                self.ffy[captured_ff_index] = torch.sin(self.fftheta[captured_ff_index]) * self.ffr[captured_ff_index]
                self.ffxy = torch.stack((self.ffx, self.ffy), dim=1)
                self.ffx2[captured_ff_index] = self.ffx[captured_ff_index].clone()
                self.ffy2[captured_ff_index] = self.ffy[captured_ff_index].clone()
                self.ffxy2 = torch.stack((self.ffx2, self.ffy2), dim=1)
                # Delete the information from self.ff_info
                # [self.ff_info.pop(key) for key in captured_ff_index if (key in self.ff_info)]
                # self.current_target_index = torch.tensor([999], dtype=torch.int32)
                # self.previous_target_index = self.current_target_index
                # Reward the firefly based on the average speed before capturing the firefly
                ##reward = reward + sum(self.past_speeds)/len(self.past_speeds)
                # print(round(self.time, 2), "sys_vel: ", [round(i, 4) for i in self.sys_vel.tolist()], "obs: ", list(np.round(self.obs, decimals = 2)), "n_targets: ",  num_targets)
                print(round(self.time, 2), "sys_vel: ", [round(i, 4) for i in self.sys_vel.tolist()], "n_targets: ",
                      num_targets)
                # self.update_slots = True
                # self.stop_rewarding_speed = True
            # elif self.has_sped_up_before == True:
            # based on Ruiyi's formula, using the distance of the closest ff in obs
            # reward += math.exp(-((self.ff_current[1, 0]**2)*(25/1.5)**2)/2)
            # self.has_sped_up_before = False
        return reward

    def step(self, action):
        self.time += self.dt
        action = torch.tensor(action)
        action[1] = action[1] / 2 + 0.5
        self.sys_vel = action.clone()
        self.state_step(action)
        self.obs = self.beliefs().numpy()
        reward = self.calculate_reward()
        self.episode_reward += reward

        if self.time >= self.episode_len * self.dt:
            self.end_episode = True
            # self.current_episode += 1
            print("Reward for the episode: ", self.episode_reward)
        # print("action: ", torch.round(action, decimals = 3), "obs: ", np.round(self.obs, decimals = 3), "Reward: ", round(reward, 3))
        return self.obs, reward, self.end_episode, {}

    def state_step(self, action):
        vnoise = torch.distributions.Normal(0, torch.ones([1, 1])).sample() * self.pro_noise_std
        wnoise = torch.distributions.Normal(0, torch.ones([1, 1])).sample() * self.pro_noise_std
        self.dw_normed = (action[0] + wnoise)
        self.dv_normed = (action[1] + vnoise)
        self.dw = (action[0] + wnoise) * self.wgain * self.dt
        self.agentheading = self.agentheading + self.dw.item()
        self.dv = (action[1] + vnoise) * self.vgain * self.dt
        self.dx = torch.cos(self.agentheading) * self.dv
        self.dy = torch.sin(self.agentheading) * self.dv
        self.agentx = self.agentx + self.dx.item()
        self.agenty = self.agenty + self.dy.item()
        self.agentxy = torch.cat((self.agentx, self.agenty))
        self.agentr = vector_norm(self.agentxy)
        self.agenttheta = torch.atan2(self.agenty, self.agentx)

        if self.agentr >= self.arena_radius:
            self.agentr = 2 * self.arena_radius - self.agentr
            self.agenttheta = self.agenttheta + pi
            self.agentx = (self.agentr * torch.cos(self.agenttheta)).reshape(1, )
            self.agenty = (self.agentr * torch.sin(self.agenttheta)).reshape(1, )
            self.agentxy = torch.cat((self.agentx, self.agenty))
            self.agentheading = self.agenttheta - pi
        while self.agentheading >= 2 * pi:
            self.agentheading = self.agentheading - 2 * pi
        while self.agentheading < 0:
            self.agentheading = self.agentheading + 2 * pi

    def beliefs(self):

        # Make a tensor containing the relative distance of all fireflies to the agent
        self.ff_distance_all = vector_norm(self.ffxy - self.agentxy, dim=1)
        # Make a tensor containing the relative (real) angle of all fireflies to the agent
        ffradians = torch.atan2(self.ffy - self.agenty, self.ffx - self.agentx)
        angle0 = ffradians - self.agentheading
        angle0[angle0 > pi] = angle0[angle0 > pi] - 2 * pi
        angle0[angle0 < -pi] = angle0[angle0 < -pi] + 2 * pi
        # Adjust the angle based on reward boundary
        angle1 = torch.abs(angle0) - torch.abs(torch.arcsin(torch.div(self.reward_boundary,
                                                                      torch.clip(self.ff_distance_all,
                                                                                 self.reward_boundary,
                                                                                 400))))  # use torch clip to get valid arcsin input
        angle2 = torch.clip(angle1, 0, pi)
        ff_angle_all = torch.sign(angle0) * angle2
        # Update the tensor containing the uncertainties of all fireflies to the agent
        visible_ff = torch.logical_and(self.ff_distance_all < self.invisible_distance,
                                       torch.abs(ff_angle_all) < self.invisible_angle)
        self.visible_ff_indices0 = visible_ff.nonzero().reshape(-1)
        for index in self.visible_ff_indices0:
            ff = self.ff_flash[index]
            if not torch.any(torch.logical_and(ff[:, 0] <= self.time, ff[:, 1] >= self.time)):
                visible_ff[index] = False
        self.visible_ff_indices = visible_ff.nonzero().reshape(-1)
        # Update memory
        self.ff_memory_all -= 1
        self.ff_memory_all[self.visible_ff_indices] = self.full_memory
        self.ff_memory_all = torch.clamp(self.ff_memory_all, 0, self.full_memory)
        # Calculate the uncertainties that will be added to relative distance and angle based on memory
        ff_uncertainty_all = (self.full_memory - self.ff_memory_all) * self.internal_noise_factor
        self.ffx2 = self.ffx2 + torch.normal(torch.zeros([self.num_ff, ]), ff_uncertainty_all)
        self.ffy2 = self.ffy2 + torch.normal(torch.zeros([self.num_ff, ]), ff_uncertainty_all)
        self.ffx2[self.visible_ff_indices] = self.ffx[self.visible_ff_indices].clone()
        self.ffy2[self.visible_ff_indices] = self.ffy[self.visible_ff_indices].clone()
        self.ffxy2 = torch.stack((self.ffx2, self.ffy2), dim=1)
        # find ffs that are in memory
        self.ff_in_memory_indices = (self.ff_memory_all > 0).nonzero().reshape(-1)
        # Consider the case where there are fewer than self.obs_ff fireflies that are in memory

        if torch.numel(self.ff_in_memory_indices) >= self.obs_ff:
            # Rank the ff whose "memory" is creater than 0 based on distance
            sorted_indices = torch.topk(-self.ff_distance_all[self.ff_in_memory_indices], self.obs_ff).indices
            self.topk_indices = self.ff_in_memory_indices[sorted_indices]
            self.ffxy2_topk = self.ffxy2[self.topk_indices]
            self.ff_distance_topk = vector_norm(self.ffxy2_topk - self.agentxy, dim=1)
            # Calculate relative angles
            ffradians = torch.atan2(self.ffxy2_topk[:, 1] - self.agenty, self.ffxy2_topk[:, 0] - self.agentx)
            angle0 = ffradians - self.agentheading
            angle0[angle0 > pi] = angle0[angle0 > pi] - 2 * pi
            angle0[angle0 < -pi] = angle0[angle0 < -pi] + 2 * pi
            self.ff_angle_topk_2 = angle0.clone()
            # Calculate relative angles of all ffs based on reward boundaries
            # Adjust the angle based on reward boundary
            angle1 = torch.abs(angle0) - torch.abs(torch.arcsin(torch.div(self.reward_boundary,
                                                                          torch.clip(self.ff_distance_topk,
                                                                                     self.reward_boundary,
                                                                                     400))))  # use torch clip to get valid arcsin input
            angle2 = torch.clip(angle1, 0, pi)
            ff_angle_topk_3 = torch.sign(angle0) * angle2
            # Concatenate distance, angle, and memory
            ff_array = torch.stack(
                (self.ff_angle_topk_2, ff_angle_topk_3, self.ff_distance_topk, self.ff_memory_all[self.topk_indices]),
                dim=0)


        elif torch.numel(self.ff_in_memory_indices) == 0:
            ff_array = torch.tensor([[0], [0], [self.invisible_distance], [0]]).repeat([1, self.obs_ff])
            self.topk_indices = torch.tensor([])

        else:
            sorted_distance, sorted_indices = torch.sort(-self.ff_distance_all[self.ff_in_memory_indices])
            self.topk_indices = self.ff_in_memory_indices[sorted_indices]
            self.ffxy2_topk = self.ffxy2[self.topk_indices]
            self.ff_distance_topk = vector_norm(self.ffxy2_topk - self.agentxy, dim=1)
            # Calculate relative angles
            ffradians = torch.atan2(self.ffxy2_topk[:, 1] - self.agenty, self.ffxy2_topk[:, 0] - self.agentx)
            angle0 = ffradians - self.agentheading
            angle0[angle0 > pi] = angle0[angle0 > pi] - 2 * pi
            angle0[angle0 < -pi] = angle0[angle0 < -pi] + 2 * pi
            self.ff_angle_topk_2 = angle0.clone()
            # Calculate relative angles of all ffs based on reward boundaries
            # Adjust the angle based on reward boundary
            angle1 = torch.abs(angle0) - torch.abs(torch.arcsin(torch.div(self.reward_boundary,
                                                                          torch.clip(self.ff_distance_topk,
                                                                                     self.reward_boundary,
                                                                                     400))))  # use torch clip to get valid arcsin input
            angle2 = torch.clip(angle1, 0, pi)
            ff_angle_topk_3 = torch.sign(angle0) * angle2
            # Concatenate distance, angle, and memory
            ff_array0 = torch.stack(
                (self.ff_angle_topk_2, ff_angle_topk_3, self.ff_distance_topk, self.ff_memory_all[self.topk_indices]),
                dim=0)
            needed_ff = self.obs_ff - torch.numel(self.ff_in_memory_indices)
            ff_array = torch.stack([ff_array0.reshape([4, -1]),
                                    torch.tensor([[0], [0], [self.invisible_distance], [0]]).repeat([1, needed_ff])],
                                   dim=1)

        # ff_array[0:2,:] = ff_array[0:2,:]/pi
        # ff_array[2,:] = (ff_array[2,:]/self.invisible_distance-0.5)*2
        # ff_array[3,:] = (ff_array[3,:]/20-0.5)*2
        self.ff_array = ff_array.clone()
        return torch.flatten(ff_array.transpose(0, 1))
print("done")




class CollectInformation(MultiFF):  # Note when using this wrapper, the number of steps cannot exceed one episode
  def __init__(self):
      super().__init__()
      self.ff_information = np.ones([self.num_ff, 8])*(-9999)   #[index, x, y, time_start, time_captured, mx(when_captured), my(when_captured), index_in_flash]

  def reset(self):
      self.obs = super().reset()
      self.ff_information[:,0] = np.arange(self.num_ff)
      self.ff_information[:,7] = np.arange(self.num_ff)
      self.ff_information[:,1] = self.ffx.numpy()
      self.ff_information[:,2] = self.ffy.numpy()
      self.ff_information[:,3] = 0
      return self.obs

  def calculate_reward(self):
      reward = super().calculate_reward()
      if self.num_targets > 0:
        for index in self.captured_ff_index:
          overall_index = int(self.ff_information[:,0][np.where(self.ff_information[:,-1]==index)[0][-1]])
          self.ff_information[overall_index, 4] = self.time
          self.ff_information[overall_index, 5] = self.agentx.item()
          self.ff_information[overall_index, 6] = self.agenty.item()
        self.new_ff_info = np.ones([self.num_targets, 8])*(-9999)
        self.new_ff_info[:,0] = np.arange(len(self.ff_information), len(self.ff_information)+self.num_targets)
        self.new_ff_info[:,7] = np.array(self.captured_ff_index)
        self.new_ff_info[:,1] = self.ffx[self.captured_ff_index].numpy()
        self.new_ff_info[:,2] = self.ffy[self.captured_ff_index].numpy()
        self.new_ff_info[:,3] = self.time
        self.ff_information = np.concatenate([self.ff_information, self.new_ff_info], axis = 0)
      return(reward)


env = CollectInformation()
env = Monitor(env, log_dir)

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


path = os.path.join(retrieve_dir, 'best_model.zip')
sac_model = sac_model.load(path,env=env)
# path2 = os.path.join(retrieve_dir, 'buffer.pkl')
# sac_model.load_replay_buffer(path2)


data_folder_name = retrieve_dir

monkey_information, ff_flash_sorted, ff_catched_T_sorted, ff_believed_position_sorted, ff_real_position_sorted, ff_life_sorted, ff_flash_end_sorted, catched_ff_num, total_ff_num \
    =data_from_SB3(env, sac_model, retrieve_dir, n_steps = 1000, retrieve_buffer = False)

filepath = data_folder_name + '/ff_dataframe.csv'
if exists(filepath):
    ff_dataframe = pd.read_csv(filepath)
    ff_dataframe = ff_dataframe.drop(["Unnamed: 0"], axis=1)
    ff_dataframe.memory = ff_dataframe.memory.astype('int')
    min_point_index = np.min(np.array(ff_dataframe['point_index']))
    max_point_index = np.max(np.array(ff_dataframe['point_index']))
else:
    ff_dataframe = MakeFFDataframe(monkey_information, ff_catched_T_sorted, ff_flash_sorted,  ff_real_position_sorted, ff_life_sorted, \
        player = "monkey", max_distance = 400, data_folder_name = data_folder_name, num_missed_index = -1, truncate = True)
    min_point_index = np.min(np.array(ff_dataframe['point_index']))
    max_point_index = np.max(np.array(ff_dataframe['point_index']))



filepath = data_folder_name + '/point_vs_cluster.csv'
if exists(filepath):
    filepath = data_folder_name + '/point_vs_cluster.csv'
    point_vs_cluster = np.array(np.loadtxt(open(filepath), delimiter=","))
    point_vs_cluster = point_vs_cluster.astype('int')
else:
    point_vs_cluster = make_point_vs_cluster(data_folder_name, monkey_information, ff_dataframe, min_point_index, max_point_index,
                          max_cluster_distance=100, max_ff_distance_from_monkey=250, max_time_past=1)

    # max_ff_distance_from_monkey = 250
    # max_cluster_distance = 100
    # max_time_past = 1  # second
    # max_points_past = math.floor(
    #     max_time_past / (monkey_information['monkey_t'][100] - monkey_information['monkey_t'][99]))
    # min_memory_of_ff = 100 - max_points_past
    # point_vs_cluster = []
    # # new structure: [[point_index, ff_index, cluster_label], [point_index, ff_index, cluster_label], ...]
    # # for i in range(7893, 7893+10):  #for testing purpose
    # for i in range(min_point_index, max_point_index + 1):
    #     selected_ff = ff_dataframe[
    #         (ff_dataframe['point_index'] == i) & (ff_dataframe['memory'] > (min_memory_of_ff)) & (
    #                     ff_dataframe['ff_distance'] < max_ff_distance_from_monkey)][['ff_x', 'ff_y', 'ff_index']]
    #     ffxy_array = selected_ff[['ff_x', 'ff_y']].to_numpy()
    #     if len(ffxy_array) > 1:
    #         ff_indices = selected_ff[['ff_index']].to_numpy()
    #         linked = linkage(ffxy_array, method='single')
    #         num_clusters = sum(linked[:, 2] > max_cluster_distance) + 1  # This is a formula I developed
    #         if num_clusters < len(ff_indices):
    #             cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='single')
    #             labels = cluster.fit_predict(ffxy_array)
    #             u, c = np.unique(labels, return_counts=True)
    #             dup = u[c > 1]
    #             # cluster_info = np.concatenate([np.repeat([i], len(ff_indices)).reshape(-1,1), ff_indices, labels.reshape([-1,1])], axis=1)
    #             for index in np.isin(labels, dup).nonzero()[0]:
    #                 point_vs_cluster.append([i, ff_indices[index].item(), labels[index]])
    #     if i % 1000 == 0:
    #         print(i, " out of ", max_point_index, " for point_vs_cluster")
    # point_vs_cluster = np.array(point_vs_cluster)
    # filepath = data_folder_name + '/point_vs_cluster.csv'
    # os.makedirs(data_folder_name, exist_ok=True)
    # np.savetxt(filepath, point_vs_cluster, delimiter=',')




n_ff_in_a_row = n_ff_in_a_row_func(catched_ff_num, ff_believed_position_sorted, distance_between_ff = 50)
two_in_a_row= np.where(n_ff_in_a_row==2)[0]
dif_time = ff_catched_T_sorted[two_in_a_row] - ff_catched_T_sorted[two_in_a_row-1]
two_in_a_row_simul = two_in_a_row[np.where(dif_time <= 0.1)[0]]
two_in_a_row_non_simul = two_in_a_row[np.where(dif_time > 0.1)[0]]

on_before_last_one_trials = on_before_last_one_func(ff_flash_end_sorted, ff_catched_T_sorted, catched_ff_num)
dif_time2 = ff_catched_T_sorted[on_before_last_one_trials] - ff_catched_T_sorted[on_before_last_one_trials-1]
on_before_last_one_simul = on_before_last_one_trials[np.where(dif_time2 <= 0.1)[0]]
on_before_last_one_non_simul = on_before_last_one_trials[np.where(dif_time2 > 0.1)[0]]

visible_before_last_one_trials = visible_before_last_one_func(ff_dataframe)

disappear_latest_trials = disappear_latest_func(ff_dataframe)

cluster_exist_trials, cluster_dataframe_point, cluster_dataframe_trial = clusters_of_ffs_func(point_vs_cluster, monkey_information, ff_catched_T_sorted)

waste_cluster_trials = np.intersect1d(cluster_exist_trials+1, np.where(n_ff_in_a_row == 1)[0])

ffs_around_target_trials, ffs_around_target_positions = ffs_around_target_func(ff_dataframe, catched_ff_num, ff_catched_T_sorted, ff_real_position_sorted, max_time_apart = 1.25)

waste_cluster_last_target_trials = np.intersect1d(ffs_around_target_trials+1, np.where(n_ff_in_a_row == 1)[0])

sudden_flash_ignore_trials, sudden_flash_ignore_trials_non_unique, sudden_flash_ignore_indices, sudden_flash_ignore_points = sudden_flash_ignore_func(ff_dataframe, ff_real_position_sorted)
sudden_flash_ignore_time = monkey_information['monkey_t'][sudden_flash_ignore_indices]

try_a_few_times_trials, try_a_few_times_indices = try_a_few_times_func(catched_ff_num, ff_catched_T_sorted, monkey_information, ff_believed_position_sorted, PLAYER, max_point_index)

give_up_after_trying_trials, give_up_after_trying_indices = give_up_after_trying_func(catched_ff_num, ff_catched_T_sorted, monkey_information, ff_believed_position_sorted, PLAYER)

