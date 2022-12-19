log_dir = "RL/SB3/SB3_data/SB3_Aug_11/"
retrieve_dir = "RL/SB3/SB3_data/SB3_Aug_11/"
retrieve_buffer = False
n_steps = 1000

import os

from RL.SB3.SB3_data_categorization import n_ff_in_a_row, visible_before_last_one_trials, disappear_latest_trials, \
                            sudden_flash_ignore_trials, try_a_few_times_trials, give_up_after_trying_trials, env, sac_model

from config import *
from RL.SB3.env import MultiFF
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from matplotlib import animation
plt.rcParams["animation.html"] = "html5"


print(n_ff_in_a_row)

# # Test the trained agent
# obs = env.reset()
# cum_rewards = 0
# mx = []
# my = []
# mheading = []  # in radians
# ffxy_all = []
# ffxy_visible = []
# ffxy2_all = []
# time_rl = []
# mx_rewarded = []
# my_rewarded = []
# reward_log = []
# captured_ff = []
# num_targets = []
# env_obs = []
# visible_ff_indices_all = []
# memory_ff_indices_all = []
# monkey_t = []
# monkey_speed = []
# obs_ff_indices_all = []
# obs_ff_overall_indices_all = []
# memory_all = []
# ff_angles2 = []
# ff_distances2 = []
# all_captured_ff_x = []
# all_captured_ff_y = []
# for step in range(n_steps):
#     action, _ = sac_model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     cum_rewards += reward
#     previous_ffxy = env.ffxy
#     prev_ff_information = env.ff_information.copy()
#     obs, reward, done, info = env.step(action)
#     reward_log.append(reward)
#     num_targets.append(env.num_targets)
#     memory_all.append(env.ff_memory_all)
#     if env.num_targets > 0:
#         captured_ff.append(env.captured_ff_index)
#         all_captured_ff_x = all_captured_ff_x + previous_ffxy[env.captured_ff_index][:, 0].tolist()
#         all_captured_ff_y = all_captured_ff_y + previous_ffxy[env.captured_ff_index][:, 1].tolist()
#     else:
#         captured_ff.append(0)
#     mx.append(env.agentx.item())
#     my.append(env.agenty.item())
#     monkey_t.append(env.time)
#     monkey_speed.append(env.dv.item())
#     mheading.append(env.agentheading.item())
#     time_rl.append(env.time)
#     ffxy_all.append(env.ffxy.clone())
#     ffxy2_all.append(env.ffxy2.clone())
#     ffxy_visible.append(env.ffxy[env.visible_ff_indices].clone())
#     env_obs.append(obs)
#     visible_ff_indices_all.append(env.visible_ff_indices)
#     memory_ff_indices_all.append(env.ff_in_memory_indices)
#     obs_ff_indices_all.append(env.topk_indices)
#     real_indices = []
#     for index in env.topk_indices:
#         real_indices.append(
#             int(prev_ff_information[:, 0][np.where(prev_ff_information[:, 7] == index.item())[0][-1]].copy()))
#     obs_ff_overall_indices_all.append(real_indices)
#     if len(env.topk_indices) > 0:
#         ff_angles2.append(env.ff_angle_topk_2)
#         ff_distances2.append(env.ff_distance_topk)
#     else:
#         ff_angles2.append(torch.tensor([]))
#         ff_distances2.append(torch.tensor([]))
#     if done:
#         obs = env.reset()







# ## Animation

# # base_number = 100
# # series = base_number + 10
# # filename = f"Demo{series + 1}"

# start = 100
# num_frame = 200
# arena_radius = 1000
# invisible_distance = 250
# global fig;
# fig = plt.figure(dpi=100)
# plt.rcParams['figure.figsize'] = (10, 10)
# plt.rcParams['font.size'] = 15
# colors_YlGn = plt.get_cmap("YlGn")(np.linspace(0, 1, 101))
# # colors_RdPu = plt.get_cmap("RdPu")(np.linspace(0,1,101))
# circle_theta = np.arange(0, 2 * pi, 0.01)
# circle_x = np.cos(circle_theta) * arena_radius
# circle_y = np.sin(circle_theta) * arena_radius
# fig, ax = plt.subplots()
# cum_mx, cum_my = mx[start:start + num_frame], my[start:start + num_frame]
# xmin, xmax = np.min(cum_mx), np.max(cum_mx)
# ymin, ymax = np.min(cum_my), np.max(cum_my)
# ax.set_xlim((xmin - invisible_distance, xmax + invisible_distance))
# ax.set_ylim((ymin - invisible_distance, ymax + invisible_distance))
# ax.set_aspect('equal')




# def animate_annotated(j):
#     ax.cla()
#     ax.axis('off')
#     ax.plot(circle_x, circle_y)
#     i = j + start
#     ax.scatter(mx[start:i + 1], my[start:i + 1], s=20)
#     ax.scatter(ffxy_all[i].T[0], ffxy_all[i].T[1], alpha=0.9, c="gray", s=30)
#     # Plot the reward boundaries of the circles

#     for k in obs_ff_indices_all[i]:
#         circle = plt.Circle((ffxy_all[i][k][0], ffxy_all[i][k][1]), 25, facecolor='yellow', edgecolor='brown',
#                             alpha=0.3, zorder=1)
#         ax.add_patch(circle)
#         if memory_all[i][k] < 20:
#             circle = plt.Circle((ffxy2_all[i][k][0], ffxy2_all[i][k][1]), 25, facecolor='grey', edgecolor='orange',
#                                 alpha=0.2, zorder=1)
#             ax.add_patch(circle)

#     ax.plot(np.array([mx[i], mx[i] + 30 * np.cos(mheading[i] + 2 * pi / 9)]),
#             np.array([my[i], my[i] + 30 * np.sin(mheading[i] + 2 * pi / 9)]), linewidth=2)
#     ax.plot(np.array([mx[i], mx[i] + 30 * np.cos(mheading[i] - 2 * pi / 9)]),
#             np.array([my[i], my[i] + 30 * np.sin(mheading[i] - 2 * pi / 9)]), linewidth=2)


#     if torch.numel(ffxy_visible[i]) > 0:
#         ax.scatter(ffxy_visible[i].T[0], ffxy_visible[i].T[1], alpha=0.9, c="red", s=30)

#     # Plot position of ffs with uncertainties
#     ax.scatter(ffxy2_all[i][memory_ff_indices_all[i]].T[0], ffxy2_all[i][memory_ff_indices_all[i]].T[1], s=60,
#                alpha=0.5, color="green")

#     # Plot ff positions recovered from relative angle and distance
#     ##if torch.numel(ffx_recovered) > 0:
#     ##  ffy_recovered = torch.sin(ff_angles2[i]+mheading[i]) * ff_distances2[i] + my[i]
#     ##  ax.scatter(ffx_recovered.clone(), ffy_recovered.clone(), alpha=0.9, c="black", s=30)

#     # Plot captured fireflies
#     if num_targets[i] > 0:
#         # ax.scatter(ffxy_all[i-1][captured_ff[i]].T[0], ffxy_all[i-1][captured_ff[i]].T[1], s=70, alpha=0.7, color="purple")
#         # captured_ff_cum_x = captured_ff_cum_x+ffxy_all[i-1][captured_ff[i]].T[0].tolist()
#         # captured_ff_cum_y = captured_ff_cum_y+ffxy_all[i-1][captured_ff[i]].T[1].tolist()
#         # if len(captured_ff_cum_x) >0:
#         ax.scatter(ffxy_all[i - 1][captured_ff[i]].T[0], ffxy_all[i - 1][captured_ff[i]].T[1], s=70, alpha=0.7,
#                    color="purple")
#     ax.set_xlim((xmin - invisible_distance, xmax + invisible_distance))
#     ax.set_ylim((ymin - invisible_distance, ymax + invisible_distance))
#     ax.set_aspect('equal')

#     # If the monkey has captured more than one 1 ff in a cluster
#     if n_ff_in_a_row[trial_num] > 1:
#       annotation = annotation + f"Captured {n_ff_in_a_row[trial_num]} ffs in a cluster\n"
#     # If the target stops being on before the monkey captures the previous firefly
#     if visible_before_last_one2[trial_num] == 1:
#       annotation = annotation + "Target visible before last captre\n"
#     # If the target disappears the latest among visible ffs
#     if disappear_latest2[trial_num] == 1:   
#       annotation = annotation + "Target disappears latest\n"
#     # If the monkey ignored a closeby ff that suddenly became visible
#     if sudden_flash_ignore_dummy[index] > 0:
#       annotation = annotation + "Ignored sudden flash\n"
#     # If the monkey uses a few tries to capture a firefly
#     if try_a_few_times_dummy[index] > 0:
#       annotation = annotation + "Try a few times to catch ff\n"
#     # If during the trial, the monkey fails to capture a firefly with a few tries and moves on to capture another one 
#     if give_up_after_trying_dummy[index] > 0:
#       annotation = annotation + "Give up after trying\n"
#     ax.text(0.5, 1.04, annotation, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=12, color="black", bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# anim_annotated = animation.FuncAnimation(fig, animate_annotated,
#                                frames=num_frame, interval=200, repeat=True)



# #
# # # gif_dir = ''
# # # os.makedirs(gif_dir, exist_ok=True)
# # # anim.save(f"{gif_dir}/{filename}.gif", writer='pillow', fps=60)
# # # writervideo = animation.FFMpegWriter(fps=4)  # original = 10
# # # anim.save(f"{gif_dir}/{filename}.mp4", writer=writervideo)
# #

# # HTML(anim.to_html5_video())
# print("done")
