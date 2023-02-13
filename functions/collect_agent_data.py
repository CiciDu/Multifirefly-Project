from model_storing_path import*
from RL.env import*
from functions.basic_func import find_intersection
from functions.LSTM_functions import *
from functions.find_patterns import *
import os
import numpy as np
import matplotlib
import pandas as pd
import torch
from matplotlib import rc
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK']='True'
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
rc('animation', html='jshtml')
matplotlib.rcParams['animation.embed_limit'] = 2**128
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)




def collect_agent_data_func(env, sac_model, n_steps = 15000, LSTM = False, hidden_dim= 128, device = "cpu", deterministic = True, first_obs = None):
    """
    Extract data points from monkey's information by increasing the interval between the points


    Parameters
    ----------
    env: obj
        the RL environment
    sac_model: obj
        the RL agent
    n_steps: num
        the number of steps that the agent will go through in the environment
    LSTM: bool
        whether LSTM is used
    hidden_dim: num
        the hidden dimension for the LSTM network
    device: str
        the device for torch
    deterministic: bool
        whether the action network will be deterministic or not
    first_obs: array, optional
        the first observation that the agent will start with 


    Returns
    -------
    monkey_information: dict
        containing the information such as the speed, angle, and location of the monkey at various points of time
    ff_flash_sorted: list
        containing the flashing-on durations of each firefly 
    ff_catched_T_sorted: np.array
        containing the time when each captured firefly gets captured
    ff_believed_position_sorted: np.array
        containing the locations of the monkey (or agent) when each captured firefly was captured 
    ff_real_position_sorted: np.array
        containing the real locations of the fireflies
    ff_life_sorted: np.array
        containing the time that each firefly comes into being and gets captured 
        (if the firefly is never captured, then capture time is replaced by the last point of time in data)
    ff_flash_end_sorted: np.array
        containing the flashing-on durations of each firefly
    catched_ff_num: num
        number of caught fireflies
    total_ff_num: num
        number of total fireflies that have appeared in the environment
    obs_ff_unique_identifiers: list
        contains the unique identifies for ffs in obs for each time point
    sorted_indices_all: np.array
        contains the sorted unique identifiers of all fireflies in the env based on time of capture 

    """

    if LSTM:
        if first_obs is None:
            state =  env.reset()
        else:
            state = first_obs
        last_action = env.action_space.sample()
        hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device), torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device))  
    else:
        if first_obs is None:
            obs = env.reset()
        else:
            obs = first_obs

    monkey_x = []
    monkey_y = []
    monkey_speed = []
    monkey_angle = []  # in radians
    monkey_t = []


    obs_ff_unique_identifiers = []
    # visible_ff_indices_all = []
    # memory_ff_indices_all = []
    # ffxy_all = []
    # ffxy_noisy_all = []
    # ffxy_visible = []

    for step in range(n_steps):
        prev_ff_information = env.ff_information.copy()
        if LSTM:
            hidden_in = hidden_out
            action, hidden_out = sac_model.policy_net.get_action(state, last_action, hidden_in, deterministic = deterministic)
            next_state, reward, done, _ = env.step(action)  
            last_action = action
            state = next_state
        else: 
            action, _ = sac_model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # memory_ff_indices_all.append(env.ff_in_memory_indices)
        
        monkey_x.append(env.agentx.item())
        monkey_y.append(env.agenty.item())
        monkey_speed.append(env.dv.item())
        monkey_angle.append(env.agentheading.item())
        monkey_t.append(env.time)
        # visible_ff_indices_all.append(env.visible_ff_indices)
        # ffxy_all.append(env.ffxy.clone())
        # ffxy_noisy_all.append(env.ffxy_noisy.clone())
        # ffxy_visible.append(env.ffxy[env.visible_ff_indices].clone())

        indexes_in_ff_information = []
        for index in env.topk_indices:
            # Find and append the row index of the last firefly with the corresponding unique identify
            last_corresponding_ff_index = np.where(prev_ff_information.loc[:, "index_in_ff_flash"] == index.item())[0][-1]
            indexes_in_ff_information.append(last_corresponding_ff_index)
        # Append the unique identifiers of ffs in obs for this time point
        obs_ff_unique_identifiers.append(indexes_in_ff_information)

        if done:
            break

    # collect all monkey's data into a dictionary
    monkey_information = pack_monkey_information(monkey_t, monkey_x, monkey_y, monkey_speed, monkey_angle, env.dt)
    
    # get information about fireflies from env.ff_information and env.ff_lash
    ff_catched_T_sorted, ff_believed_position_sorted, ff_real_position_sorted, ff_life_sorted, ff_flash_sorted, ff_flash_end_sorted, sorted_indices_all = unpack_ff_information_of_agent(env.ff_information, env.ff_flash, env.time)
    
    catched_ff_num = len(ff_catched_T_sorted)
    total_ff_num = len(ff_life_sorted)

    # Find the indices of ffs in obs for each time point, keeping the indices that will be used by ff_dataframe
    reversed_sorting = reverse_value_and_position(sorted_indices_all)
    obs_ff_indices_in_ff_dataframe = [reversed_sorting[indices] for indices in obs_ff_unique_identifiers]
    
    return monkey_information, ff_flash_sorted, ff_catched_T_sorted, ff_believed_position_sorted, \
           ff_real_position_sorted, ff_life_sorted, ff_flash_end_sorted, catched_ff_num, total_ff_num, \
           obs_ff_indices_in_ff_dataframe, sorted_indices_all





def pack_monkey_information(monkey_t, monkey_x, monkey_y, monkey_speed, monkey_angle, dt):
    """
    Organize the information of the monkey/agent into a dictionary


    Parameters
    ----------
    monkey_t: list
        containing a series of time points
    monkey_x: list
        containing a series of x-positions of the monkey/agent
    monkey_y: list
        containing a series of y-positions of the monkey/agent  
    monkey_speed: list
        containing a series of linear speeds of the monkey/agent  
    monkey_angle: list    
        containing a series of angles of the monkey/agent  
    dt: num
        the time interval

    Returns
    -------
    monkey_information: dict
        containing the information such as the speed, angle, and location of the monkey at various points of time

    """
    monkey_t = np.array(monkey_t)
    monkey_x = np.array(monkey_x)
    monkey_y = np.array(monkey_y)
    monkey_speed = np.array(monkey_speed)
    monkey_angle = np.array(monkey_angle)
    monkey_angle = np.remainder(monkey_angle, 2*pi)

    monkey_information = {
        'monkey_t': monkey_t,
        'monkey_x': monkey_x,
        'monkey_y': monkey_y,
        'monkey_speed': monkey_speed,
        'monkey_angle': monkey_angle,
    }

    # determine whether the speed of the monkey is above a threshold at each time point
    monkey_speeddummy = (monkey_speed > 200 * 0.01 * dt).astype(int)
    monkey_information['monkey_speeddummy'] = monkey_speeddummy

    # Make sure that the absolute values of the angles are smaller than 2*pi
    delta_angle = np.remainder(np.diff(monkey_angle), 2*pi)
    delta_time = np.diff(monkey_t)

    monkey_dw = np.divide(delta_angle, delta_time)
    # Since we cannot calculate the dw for the first time point, we'll just repeat the firs value
    monkey_dw = np.append(monkey_dw[0], monkey_dw)
    monkey_information['monkey_dw'] = monkey_dw

    return monkey_information





def find_flash_time_for_one_ff(ff_flash, lifetime):
    """
    Select the flashing durations that overlap with the ff's lifetime


    Parameters
    ----------
    ff_flash: np.array
        containing the intervals that the firefly flashes on
    lifetime: np.array (2,)
        contains when the ff starts to be alive and when it stops being alive (either captured or time is over)

    
    Returns
    -------
    ff_flash_valid: array
        containing the intervals that the firefly flashes on within the ff's life time.

    """

    # Find indices of overlapped intervals between ff_flash and duration
    indices_of_overlapped_intervals = find_intersection(ff_flash, lifetime)
    if len(indices_of_overlapped_intervals) > 0:
        ff_flash_valid = ff_flash[indices_of_overlapped_intervals]
        # Make sure that the first interval does not start until the ff starts to be alive
        ff_flash_valid[0][0] = max(ff_flash_valid[0][0], lifetime[0])
        # Also make sure that the last interval ends before the firefly is captured
        ff_flash_valid[-1][1] = min(ff_flash_valid[-1][1], lifetime[1])
    else:
        ff_flash_valid = np.array([[-1, -1]])
    return ff_flash_valid





def make_ff_flash_sorted(env_ff_flash, ff_information, sorted_indices_all, env_end_time):
    """
    make ff_flash_sorted by using data collected from the agent


    Parameters
    ----------
    env_ff_flash: torch.tensor
        containing the intervals that each firefly flashes on in the environment
    ff_information: df
        contains information about each firefly, with the following columns:
        [unique_identifier, ffx, ffy, time_start_to_be_alive, time_captured, 
        mx_when_catching_ff, my_when_catching_ff, index_in_ff_flash]
    sorted_indices_all: array
        contains the sorted unique identifiers of all fireflies in the env based on time of capture 
    env_end_time: num
        the last time point of the env

    Returns
    -------
    ff_flash_sorted: array
        containing the sorted intervals that each firefly flashes on 

    """

    ff_flash_sorted = []
    for index, ff in ff_information.iloc[sorted_indices_all].iterrows():
        lifetime = [ff["time_start_to_be_alive"], ff["time_captured"]]
        if ff["time_captured"] == -9999:
            lifetime[1] = env_end_time
        ff_flash = env_ff_flash[int(ff["index_in_ff_flash"])].numpy()
        ff_flash_valid = find_flash_time_for_one_ff(ff_flash, lifetime)
        ff_flash_sorted.append(ff_flash_valid)
    return ff_flash_sorted





def make_env_ff_flash_from_real_data(ff_flash_sorted_of_monkey, alive_ffs, ff_flash_duration):
    """
    make ff_flash for the env by using real monkey's data


    Parameters
    ----------
    ff_flash_sorted_of_monkey: np.array
        containing the time that each firefly flashes on and off, from the monkey data
    alive_ffs: np.array
        containing indices of fireflies that have been alive during the relevant interval
    ff_flash_duration: list (2,)
        the relevant interval for which alive ffs will be selected

    Returns
    -------
    env_ff_flash: torch.tensor
        containing the intervals that each firefly flashes on in the environment

    """

    env_ff_flash = []
    # find the starting time of the section of monkey data we are interested in
    start_time = ff_flash_duration[0]
    for index in alive_ffs:
        # take out the flashing durations for each individual alive ff
        ff_flash = ff_flash_sorted_of_monkey[index]
        ff_flash_valid = find_flash_time_for_one_ff(ff_flash, ff_flash_duration)
        if ff_flash_valid[-1, -1] != -1:
            # because time starts from 0 for the environment, we need to subtract the  
            # starting time from the monkey data
            ff_flash_valid = ff_flash_valid-start_time
        env_ff_flash.append(torch.tensor(ff_flash_valid))
    return env_ff_flash





def increase_dt_for_monkey_information(monkey_t, monkey_x, monkey_y, new_dt, old_dt = 0.0166):
    """
    Extract data points from monkey's information by increasing the interval between the points


    Parameters
    ----------
    monkey_t: np.array
        containing an array of time
    monkey_x: np.array
        containing an array of x-positions of the monkey
    monkey_y: np.array
        containing an array of y-positions of the monkey
    new_dt: num
        the new time interval between every two points
    old_dt: num
        the old time interval between every two points


    Returns
    -------
    monkey_t: np.array
        containing an array of time
    monkey_x: np.array
        containing an array of x-positions of the monkey
    monkey_y: np.array
        containing an array of y-positions of the monkey
    monkey_speed: np.array
        containing an array of speeds of the monkey
    monkey_angle: np.array
        containing an array of the angle (heading) of the monkey
    monkey_dw: np.array
        containing an array of the angular speed of the monkey

    """


    ratio = new_dt/old_dt
    agent_indices = np.arange(0, len(monkey_t)-1, ratio)
    # used len(monkey_t)-1 for fear that after rounding, the last number will exceed the limit
    agent_indices = np.round(agent_indices).astype('int') 
    
    monkey_t = monkey_t[agent_indices]
    monkey_x = monkey_x[agent_indices]
    monkey_y = monkey_y[agent_indices]

    delta_time = np.diff(monkey_t)
    delta_x = np.diff(monkey_x)
    delta_y = np.diff(monkey_y)
    delta_position = np.sqrt(np.square(delta_x) + np.square(delta_y))
    monkey_speed = np.divide(delta_position, delta_time)
    monkey_speed = np.append(monkey_speed[0], monkey_speed)

    # If the monkey's speed at one point exceeds 200 
    # (this can happen when the monkey reaches the boundary and comes out at another place)
    # we replace it with the previous speed.
    while np.where(monkey_speed >= 200)[0].size > 0:
      index = np.where(monkey_speed >= 200)[0]
      monkey_speed1 = np.append(monkey_speed[0], monkey_speed)
      monkey_speed[index] = monkey_speed1[index]

    # find monkey_angle
    monkey_angle = np.arctan2(delta_y, delta_x)
    monkey_angle = np.append(monkey_angle[0], monkey_angle)

    # Find dw
    delta_angle = np.remainder(np.diff(monkey_angle), 2 * pi)
    monkey_dw = np.divide(delta_angle, delta_time)
    monkey_dw = np.append(monkey_dw[0], monkey_dw)

    return monkey_t, monkey_x, monkey_y, monkey_speed, monkey_angle, monkey_dw






def unpack_ff_information_of_agent(ff_information, env_ff_flash, env_end_time):
    """
    Extract important information from ff_information from the agent


    Parameters
    ----------
    ff_information: df
        contains information about each firefly, with the following columns:
        [unique_identifier, ffx, ffy, time_start_to_be_alive, time_captured, 
        mx_when_catching_ff, my_when_catching_ff, index_in_ff_flash]
    env_ff_flash: torch.tensor
        containing the intervals that each firefly flashes on in the environment
    env_end_time: num
        the last time point traversed by the agent


    Returns
    -------
    ff_catched_T_sorted: np.array
        containing the time when each captured firefly gets captured
    ff_believed_position_sorted: np.array
        containing the locations of the monkey (or agent) when each captured firefly was captured 
    ff_real_position_sorted: np.array
        containing the real locations of the fireflies
    ff_life_sorted: np.array
        containing the time that each firefly comes into being and gets captured 
        (if the firefly is never captured, then capture time is replaced by the last point of time in data)
    ff_flash_sorted: list
        containing the flashing-on durations of each firefly 
    ff_flash_end_sorted: np.array
        containing the end of each flash-on duration of each firefly
    sorted_indices_all: np.array
        contains the sorted unique identifiers of all fireflies in the env based on time of capture 

    """

    ff_time_captured_all = ff_information.loc[:, "time_captured"]
    captured_ff_indices = np.where(ff_time_captured_all != -9999)[0]
    not_captured_ff_indices = np.where(ff_time_captured_all == -9999)[0]

    # Sort the indices of the fireflies by the time they are captured
    sorted_indices_captured = captured_ff_indices[np.argsort(ff_time_captured_all[captured_ff_indices])]
    sorted_indices_all = np.concatenate([sorted_indices_captured, not_captured_ff_indices])

    ff_flash_sorted = make_ff_flash_sorted(env_ff_flash, ff_information, sorted_indices_all, env_end_time)

    # Note that the following two arrays will be shorter than the other arrays
    ff_catched_T_sorted = np.array(ff_time_captured_all[sorted_indices_captured])  
    ff_believed_position_sorted = np.array(ff_information.iloc[sorted_indices_captured, 5:7])

    ff_real_position_sorted = np.array(ff_information.iloc[sorted_indices_all, 1:3])
    ff_life_sorted = np.array(ff_information.iloc[sorted_indices_all, 3:5])
    # If the firefly has not been captured, then the end of their lifetime will be replaced by the end time of the environment
    ff_life_sorted[:, 1][np.where(ff_life_sorted[:, 1] == -9999)[0]] = env_end_time
    ff_flash_end_sorted = [flash[-1, 1] if len(flash) > 0 else env_end_time for flash in ff_flash_sorted]
    ff_flash_end_sorted = np.array(ff_flash_end_sorted)

    return ff_catched_T_sorted, ff_believed_position_sorted, ff_real_position_sorted, ff_life_sorted, ff_flash_sorted, ff_flash_end_sorted, sorted_indices_all






def reverse_value_and_position(sorted_indices_all):
    """

    Parameters
    ----------
    sorted_indices_all: np.array
        contains the sorted unique identifiers of all fireflies in the env based on time of capture 


    Returns
    -------
    reversed_sorting: np.array
        if the ith element of sorted_indices_all is j, then the jth element of reversed_sorting is i

    """

    # Initiate an array to contain the new sorted array
    reversed_sorting = np.zeros(len(sorted_indices_all))

    for position in range(len(sorted_indices_all)):
        value = sorted_indices_all[position]
        reversed_sorting[value] = position
    return reversed_sorting






def find_corresponding_info_of_agent(info_of_monkey, currentTrial, num_trials, sac_model, agent_dt, LSTM=False, env_kwargs=None):
    """
    Let the agent replicates part of the monkey's action in the same environment as the monkey, and after that see how the agent's
    actions will differ from those of the monkey's


    Parameters
    ----------
    info_of_monkey: dict
        contains various important arrays, dataframes, or lists derived from the real monkey data
    currentTrial: num
        the current trial to be plotted
    num_trials: num
        the number of trials (counting from the currentTrial into the past) to be plotted
    sac_model: obj
        the RL agent
    agent_dt: num
        the duration of each step for the agent
    LSTM: bool
        whether LSTM is used
    env_kwargs: dict
        keyword arguments to be used when establishing the RL environment

    Returns
    -------
    info_of_agent: dict
        contains various important arrays, dataframes, or lists derived from the RL environmentthe and the agent's behaviours
    graph_whole_duration: list of 2 elements
        containing the start time and the end time (in reference to the monkey data) of the interval to be plotted
    rotation_matrix: np.array
        to be used to rotate the graph when plotting
    num_imitation_steps_monkey: num
        the number of steps used by the monkey for the part of the trajectory shared by the monkey and the agent (with the agent copying the monkey)
    num_imitation_steps_agent: num
        the number of steps used by the agent for the part of the trajectory shared by the monkey and the agent (with the agent copying the monkey)

    """      

    # Set a duration that the plot will encompass; first find the start time
    start_time = min(info_of_monkey['ff_catched_T_sorted'][currentTrial-3], info_of_monkey['ff_catched_T_sorted'][currentTrial]-num_trials)
    graph_whole_duration = [start_time, info_of_monkey['ff_catched_T_sorted'][currentTrial]]
    # We take out a part at the beginning of the graph_whole_duration, where the agent will replicate the monkey's action
    monkey_acting_duration = [start_time, info_of_monkey['ff_catched_T_sorted'][currentTrial]-1.5]
    

    # Find the indices of alive fireflies
    alive_ffs = np.array([index for index, life in enumerate(info_of_monkey['ff_life_sorted']) if (life[1] >= graph_whole_duration[0]) and (life[0] < graph_whole_duration[1])])  
    # Take out relevant information from the monkey data
    M_cum_indices = np.where((info_of_monkey['monkey_information']['monkey_t'] >= monkey_acting_duration[0]) & (info_of_monkey['monkey_information']['monkey_t'] <= monkey_acting_duration[1]))[0]
    M_cum_t = info_of_monkey['monkey_information']['monkey_t'][M_cum_indices]
    M_cum_mx, M_cum_my = info_of_monkey['monkey_information']['monkey_x'][M_cum_indices], info_of_monkey['monkey_information']['monkey_y'][M_cum_indices]
    # Find the correponding agent information (replicated from the monkey's data, but with a bigger time interval)
    A_cum_t, A_cum_mx, A_cum_my, A_cum_speed, A_cum_angle, A_cum_dw = increase_dt_for_monkey_information(M_cum_t, M_cum_mx, M_cum_my, agent_dt)
    num_imitation_steps_agent = len(A_cum_t)
    num_imitation_steps_monkey = len(M_cum_t)


    # Find the angle from the starting point to the target
    theta = pi/2-np.arctan2(M_cum_my[-1]-M_cum_my[0], M_cum_mx[-1]-M_cum_mx[0])     
    c, s = np.cos(theta), np.sin(theta)
    # Find the rotation matrix
    rotation_matrix = np.array(((c, -s), (s, c)))


    # Make the RL environment
    if LSTM:
        env = CollectInformationLSTM(dt=agent_dt, num_ff=len(alive_ffs), **env_kwargs)
        hidden_out = (torch.zeros([1, 1, sac_model.hidden_dim], dtype=torch.float), torch.zeros([1, 1, sac_model.hidden_dim], dtype=torch.float)) 
    else:
        env = CollectInformation(dt=agent_dt, num_ff=len(alive_ffs), **env_kwargs)
    env.flash_on_interval = 0.3
    env.distance2center_cost = 0
    # Replicate the monkey's environment
    env.ff_flash = make_env_ff_flash_from_real_data(info_of_monkey['ff_flash_sorted'], alive_ffs, graph_whole_duration)
    env_ffxy = torch.tensor(info_of_monkey['ff_real_position_sorted'][alive_ffs], dtype=torch.float)
    env.ffxy, env.ffxy_noisy = env_ffxy, env_ffxy
    env.ffx, env.ffx_noisy = env.ffxy[:, 0], env.ffxy[:, 0]
    env.ffy, env.ffy_noisy = env.ffxy[:, 1], env.ffxy[:, 1]
    obs = env.reset(use_random_ff = False)
    

    # Convert monkey's actions into the scale used by the agent; Note, the first action in monkey_action is simply a replication 
    # of the second action, because each action is calculated by the difference between the positions of every two points
    monkey_actions = np.stack((A_cum_dw/env.wgain, (A_cum_speed/env.vgain-0.5)*2), axis=1)
        

    
    monkey_x = []
    monkey_y = []
    monkey_speed = []
    monkey_angle = []  # in radians
    monkey_t = []
    obs_ff_unique_identifiers = []


    #======================================= Agent Replicating Monkey's Actions ==========================================
    # In order to replicate the monkey's action, we need to turn the action noise temporarily to zero.
    original_action_noise_std = env.action_noise_std
    env.action_noise_std = 0
    # Find the right starting time for the environment, so the ff_flash can match
    env.time = M_cum_t[0] - start_time
    env.agentheading = torch.tensor([A_cum_angle[0]])
    env.agentx = torch.tensor([A_cum_mx[0]])
    env.agenty = torch.tensor([A_cum_my[0]])
        
    for step in range(1, num_imitation_steps_agent):
        # Starting to replicate from the second step in the data
        prev_ff_information = env.ff_information.copy()
        if LSTM:
            if step > 0:
                hidden_in = hidden_out
                action, hidden_out = sac_model.policy_net.get_action(state, last_action, hidden_in, deterministic = True)
            last_action = monkey_actions[step]
            next_state, reward, done, _ = env.step(monkey_actions[step])  
            state = next_state
        else: 
            obs, reward, done, info = env.step(monkey_actions[step])
        # We replace the agent's position with monkey's real position so that small differences (due to problems 
        # such as inconsistent time intervals) can be corrected 
        env.agentheading = torch.tensor([A_cum_angle[step]])
        env.agentx = torch.tensor([A_cum_mx[step]])
        env.agenty = torch.tensor([A_cum_my[step]])

        monkey_x.append(env.agentx.item())
        monkey_y.append(env.agenty.item())
        monkey_speed.append((monkey_actions[step, 1]/2+0.5)*env.vgain)
        monkey_angle.append(env.agentheading.item())
        monkey_t.append(env.time)

        indexes_in_ff_information = []
        for index in env.topk_indices:
            # Find and append the row index of the last firefly (the row that has the largest row number) 
            # in ff_information that has the same index_in_ff_lash
            last_corresponding_ff_index = np.where(prev_ff_information.loc[:, "index_in_ff_flash"] == index.item())[0][-1]
            indexes_in_ff_information.append(last_corresponding_ff_index)
        obs_ff_unique_identifiers.append(indexes_in_ff_information)

    #======================================= Agent Moving Independently ==========================================
    env.action_noise_std = original_action_noise_std
    num_total_steps = int(np.ceil((graph_whole_duration[1]-graph_whole_duration[0])/agent_dt))
    for step in range(num_imitation_steps_agent, num_total_steps+10):
        if LSTM:
            hidden_in = hidden_out
            action, hidden_out = sac_model.policy_net.get_action(state, last_action, hidden_in, deterministic = True)
            last_action = action
            next_state, reward, done, _ = env.step(action)  
            state = next_state
        else: 
            action, _ = sac_model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        monkey_x.append(env.agentx.item())
        monkey_y.append(env.agenty.item())
        monkey_speed.append(env.dv.item())
        monkey_angle.append(env.agentheading.item())
        monkey_t.append(env.time)

        indexes_in_ff_information = []
        for index in env.topk_indices:
            last_corresponding_ff_index = np.where(prev_ff_information.loc[:, "index_in_ff_flash"] == index.item())[0][-1]
            indexes_in_ff_information.append(last_corresponding_ff_index)
        obs_ff_unique_identifiers.append(indexes_in_ff_information)



    #======================================= Organize Collected Data ==========================================
    monkey_information = pack_monkey_information(monkey_t, monkey_x, monkey_y, monkey_speed, monkey_angle, env.dt)
    ff_catched_T_sorted, ff_believed_position_sorted, ff_real_position_sorted, ff_life_sorted, ff_flash_sorted, ff_flash_end_sorted, sorted_indices_all = unpack_ff_information_of_agent(env.ff_information, env.ff_flash, env.time)
    catched_ff_num = len(ff_catched_T_sorted)

    # Find the indices of ffs in obs for each time point, keeping the indices that will be used by ff_dataframe
    reversed_sorting = reverse_value_and_position(sorted_indices_all)
    obs_ff_indices_in_ff_dataframe = [reversed_sorting[indices] for indices in obs_ff_unique_identifiers]

    # Make ff_dataframe
    ff_dataframe_args = (monkey_information, ff_catched_T_sorted, ff_flash_sorted,  ff_real_position_sorted, ff_life_sorted)
    ff_dataframe_kargs = {"max_distance": 400, "data_folder_name": None, "num_missed_index": 0}
    ff_dataframe = make_ff_dataframe(*ff_dataframe_args, **ff_dataframe_kargs, player = "agent", truncate = False, \
                                    obs_ff_indices_in_ff_dataframe = obs_ff_indices_in_ff_dataframe)
    # Only keep the relevant part of ff_dataframe
    ff_dataframe = ff_dataframe[ff_dataframe['time'] <= graph_whole_duration[1]-graph_whole_duration[0]]

    # Select trials, indices, and positions where there is a cluster of fireflies around the target
    ffs_around_target_trials, ffs_around_target_indices, ffs_around_target_positions = ffs_around_target_func(ff_dataframe, catched_ff_num, ff_catched_T_sorted, ff_real_position_sorted, max_time_apart = 1.25)

    # Decrease num_imitation_steps_agent by 1 because the step number has started from 1
    num_imitation_steps_agent = num_imitation_steps_agent-1


    info_of_agent = {
      "monkey_information": monkey_information,
      "ff_dataframe": ff_dataframe,
      "ff_catched_T_sorted": ff_catched_T_sorted,
      "ff_real_position_sorted": ff_real_position_sorted,
      "ff_believed_position_sorted": ff_believed_position_sorted,
      "ff_life_sorted": ff_life_sorted,
      "ff_flash_sorted": ff_flash_sorted,
      "ff_flash_end_sorted": ff_flash_end_sorted,
      "ffs_around_target_indices": ffs_around_target_indices
    }


    return info_of_agent, graph_whole_duration, rotation_matrix, num_imitation_steps_monkey, num_imitation_steps_agent
