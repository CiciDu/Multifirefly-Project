log_dir = "RL/SB3/SB3_data/SB3_Aug_11/"
retrieve_dir = "SB3_data/SB3_Aug_11/"
from RL.SB3.env import*

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir('/Users/dusiyi/Multifirefly-Project/')
import numpy as np
import matplotlib
from matplotlib import rc
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.monitor import Monitor
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
rc('animation', html='jshtml')
matplotlib.rcParams['animation.embed_limit'] = 2**128
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# # for agent
# PLAYER = "agent"
# NEW_DATASET = True
# MONKEY_DATA = False
# NO_PLOT_NEEDED = True
# data_folder_name = "RL/LSTM_July_29"
# data_num = 721
# trial_total_num = 30


# np.random.seed(7777)
# #rng = np.random.default_rng(2021)
# NEW_DATASET = True
# MONKEY_DATA = True
# SHOW_INTERESTING = False
# trial_total_num = 20
# list_of_colors = ["navy", "magenta", "white", "gray", "brown", "black"] # For plotting ff clusters
# point_index_array = np.arange(1,100,10)




def data_from_SB3(env, sac_model, retrieve_dir, n_steps = 1000, retrieve_buffer = False):
    path = os.path.join(retrieve_dir, 'best_model.zip')
    sac_model = sac_model.load(path, env=env, print_system_info=False)

    env.reset()
    env.flash_on_interval = 0.3
    env.distance2center_cost = 0


    # Test the trained agent
    obs = env.reset()
    cum_rewards = 0
    mx = []
    my = []
    mheading = []  # in radians
    ffxy_all = []
    ffxy_visible = []
    ffxy2_all = []
    time_rl = []
    mx_rewarded = []
    my_rewarded = []
    reward_log = []
    captured_ff = []
    num_targets = []
    env_obs = []
    visible_ff_indices_all = []
    memory_ff_indices_all = []
    monkey_t = []
    monkey_speed = []
    obs_ff_indices_all = []
    obs_ff_overall_indices_all = []
    memory_all = []
    ff_angles2 = []
    ff_distances2 = []
    all_captured_ff_x = []
    all_captured_ff_y = []
    for step in range(n_steps):
        action, _ = sac_model.predict(obs, deterministic=True)
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
        time_rl.append(env.time)
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

    monkey_speed = np.array(monkey_speed)
    mheading = np.array(mheading)

    monkey_information = {
        'monkey_x': np.array(mx),
        'monkey_y': np.array(my),
        'monkey_t': np.array(monkey_t),
        'monkey_speed': monkey_speed,
        'monkey_angle': mheading,
    }


    monkey_speeddummy = (monkey_speed > 200 * 0.01 * env.dt).astype(int)
    monkey_information['monkey_speeddummy'] = monkey_speeddummy
    delta_time = np.delete(monkey_information['monkey_t'], 0) - np.delete(monkey_information['monkey_t'], -1)
    monkey_dw = np.diff(mheading, prepend=mheading[0]) / env.dt
    monkey_information['monkey_dw'] = monkey_dw

    # ff_information:
    # [index, x, y, time_start, time_captured, mx(when_captured), my(when_captured), index_in_flash]


    # sort the time of capture for ff (if they have been captured)
    ff_information = env.ff_information.copy()
    ff_time_captured_all = ff_information[:, 4]
    captured_ff_indices = np.where(ff_time_captured_all != -9999)[0]
    not_captured_ff_indices = np.where(ff_time_captured_all == -9999)[0]
    num_captured_ff = len(captured_ff_indices)
    sorted_indices_captured = captured_ff_indices[np.argsort(ff_time_captured_all[captured_ff_indices])]
    sort_indices_all = np.concatenate([sorted_indices_captured, not_captured_ff_indices])
    ff_information_sorted = ff_information[sort_indices_all]

    # make ff_flash_sorted
    ff_flash_sorted = []
    env_end_time = env.time
    for ff in ff_information[sorted_indices_captured]:
        flash = env.ff_flash[int(ff[7])].numpy()
        replace_start = False
        replace_end = False

        # first flatten flash, and then find the elements that are before ff start time
        # if the number of elements is even, then the flashing_start_time is the start time of the next interval
        before_start = np.where(flash.flatten() <= ff[3])[0]
        if len(before_start) > 0:
            if len(before_start) % 2 == 0:
                start_flash_index = int(len(before_start) / 2)
            else:
                start_flash_index = int((len(before_start) - 1) / 2)
                replace_start = True
        else:
            start_flash_index = 0

        after_finish = np.where(flash.flatten() >= ff[4])[0]
        if len(after_finish) > 0:
            num_indices_before_finish = after_finish[0]
            if num_indices_before_finish % 2 == 0:
                end_flash_index = int(num_indices_before_finish / 2) - 1
            else:
                end_flash_index = int((num_indices_before_finish + 1) / 2) - 1
                replace_end = True
        else:
            end_flash_index = len(flash) - 1

        ff_flash = flash[start_flash_index:(end_flash_index + 1)]

        if len(ff_flash) < 2:
            ff_flash = flash[end_flash_index - 1: end_flash_index + 1]
            if len(ff_flash) < 2:
                ff_flash = np.array([[-1, -1]])

        if replace_start == True:
            ff_flash[0, 0] = ff[3]
        if replace_end == True:
            ff_flash[-1, 1] = ff[4]

        if len(ff_flash) == 0:
            ff_flash = np.array([[-1, -1]])

        ff_flash_sorted.append(ff_flash)

    # For the ffs that have never been captured, the end_flash_time is evaluated not in relation to the
    # time of captrue, but to the time that the env ends
    for ff in ff_information[not_captured_ff_indices]:
        flash = env.ff_flash[int(ff[7])].numpy()
        replace_start = False

        # first flatten flash, and then find the elements that are before ff start time
        # if the number of elements is even, then the flashing_start_time is the start time of the next interval
        before_start = np.where(flash.flatten() <= ff[3])[0]
        if len(before_start) % 2 == 0:
            start_flash_index = int(len(before_start) / 2)
        else:
            start_flash_index = int((len(before_start) - 1) / 2)
            replace_start = True

        # # if we only want the part before the testing ends
        #   after_finish = np.where(flash.flatten() >= env_end_time)[0] # differing from captured ffs
        #   num_indices_before_finish = after_finish[0]
        #   if num_indices_before_finish%2 == 0:
        #     end_flash_index = int(num_indices_before_finish/2) - 1
        #   else:
        #     end_flash_index = int((num_indices_before_finish+1)/2) - 1
        #     replace_end = True
        #   ff_flash = flash[start_flash_index:(end_flash_index+1)]

        ff_flash = flash[start_flash_index:]

        if len(ff_flash) < 2:
            ff_flash = flash[end_flash_index - 1: end_flash_index + 1]
            if len(ff_flash) < 2:
                ff_flash = np.array([[-1, -1]])

        if replace_start == True:
            ff_flash[0, 0] = ff[3]

        # # if we only want the part before the testing ends
        # if replace_end == True:
        #   ff_flash[-1, 1] = ff[4]
        if len(ff_flash) == 0:
            ff_flash = np.array([[-1, -1]])

        ff_flash_sorted.append(ff_flash)

    ff_catched_T_sorted = ff_time_captured_all[
        sorted_indices_captured]  # Note that these two will be shorter than the other arrays
    ff_believed_position_sorted = ff_information[:, 5:7][sorted_indices_captured]

    ff_real_position_sorted = ff_information[:, 1:3][sort_indices_all]
    ff_life_sorted = ff_information[:, 3:5][sort_indices_all]
    ff_life_sorted[:, 1][np.where(ff_life_sorted[:, 1] == -9999)[0]] = env.time
    ff_flash_end_sorted = [flash[-1, 1] if len(flash) > 0 else env.time for flash in ff_flash_sorted]
    ff_flash_end_sorted = np.array(ff_flash_end_sorted)

    catched_ff_num = len(ff_catched_T_sorted)
    total_ff_num = len(ff_life_sorted)

    return monkey_information, ff_flash_sorted, ff_catched_T_sorted, ff_believed_position_sorted, \
           ff_real_position_sorted, ff_life_sorted, ff_flash_end_sorted, catched_ff_num, total_ff_num