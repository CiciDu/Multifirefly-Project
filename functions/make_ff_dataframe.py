
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import math
from math import pi
import os, sys
import torch
from contextlib import contextmanager

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


"""### make ff_dataframe"""

def MakeFFDataframe(monkey_information, ff_catched_T_sorted, ff_flash_sorted,  ff_real_position_sorted, ff_life_sorted, player = "monkey", max_distance = 400, data_folder_name = None, num_missed_index = -1, truncate=False):
  if player == "monkey":
    # Let's use data from monkey_information. But we shall cut off portion that is before the time of capturing the first target
    monkey_t_array0 = np.array(monkey_information['monkey_t'])
    monkey_x_array0 = np.array(monkey_information['monkey_x'])
    monkey_y_array0 = np.array(monkey_information['monkey_y'])
    monkey_angle_array0 = np.array(monkey_information['monkey_angle'])
    if num_missed_index < 0:
      valid_index = np.where(monkey_t_array0 > ff_catched_T_sorted[0])[0]
      num_missed_index = valid_index[0]
    monkey_t_array = monkey_t_array0[num_missed_index:]
    monkey_x_array = monkey_x_array0[num_missed_index:]
    monkey_y_array = monkey_y_array0[num_missed_index:]
    monkey_angle_array = monkey_angle_array0[num_missed_index:]
    index_array = np.array(range(len(monkey_t_array)))

    #max_distance = 400
    reward_boundary = 25
    #max_time = 400
    #max_time = ff_catched_T_sorted[1251] #replace with ff_catched_T_sorted[-1]
    ff_index = []
    point_index = []
    time = []
    target_index = []
    ff_x = []
    ff_y = []
    monkey_x = []
    monkey_y = []
    visible = []
    memory = []
    ff_distance = []
    ff_angle = []
    ff_angle_boundary = []
    left_right = []
    num_captures = []
    distance_closest_ff =[]
    distance_2ndclosest_ff = []
    num_ff_within = []
    reward_distance = []
    reward_angle = []
    catched_ff_num = len(ff_catched_T_sorted) - 200
    total_ff_num = len(ff_life_sorted)
    for i in range(total_ff_num):
      if i % 100 == 0:
        print(i," out of ", total_ff_num)
      # visible_indices contains the indices of the points when the ff is visible (within a suitable distance & at the right angle)
      visible_indices = []
      # Go through every visible duration of the same ff
      ff_flash = ff_flash_sorted[i]
      for j in range(len(ff_flash)):
        visible_duration = ff_flash[j]
        ## if visible_duration[0] < max_time:
        # Find the corresponding monkey information:
        cum_indices = np.where((monkey_t_array >= visible_duration[0]) & 
                            (monkey_t_array <= visible_duration[1]))[0]
        cum_t, cum_angle = monkey_t_array[cum_indices], monkey_angle_array[cum_indices]
        cum_mx, cum_my = monkey_x_array[cum_indices], monkey_y_array[cum_indices]
        distances_to_monkey = LA.norm(np.stack([cum_mx, cum_my], axis=1)-ff_real_position_sorted[i], axis = 1)
        valid_distance_indices = np.where(distances_to_monkey < max_distance)[0]
        if len(valid_distance_indices) > 0:
          angles_to_monkey = np.arctan2(ff_real_position_sorted[i,1]-cum_my[valid_distance_indices], ff_real_position_sorted[i,0]-cum_mx[valid_distance_indices])-cum_angle[valid_distance_indices]
          angles_to_monkey[angles_to_monkey > pi] = angles_to_monkey[angles_to_monkey > pi] - 2*pi
          angles_to_monkey[angles_to_monkey < -pi] = angles_to_monkey[angles_to_monkey < -pi] + 2*pi
          angles_adjusted = np.absolute(angles_to_monkey)-np.abs(np.arcsin(np.divide(reward_boundary, np.maximum(distances_to_monkey[valid_distance_indices], reward_boundary) ))) # use torch clip to get valid arcsin input
          angles_adjusted = np.clip(angles_adjusted, 0, pi)
          angles_to_monkey = np.sign(angles_to_monkey)* angles_adjusted
          overall_valid_indices = valid_distance_indices[np.where(np.absolute(angles_to_monkey) <= 2*pi/9)[0]]
          visible_indices = visible_indices + cum_indices[overall_valid_indices].tolist()
      visible_indices = np.array(visible_indices)
      if len(visible_indices) > 0:
        # Make a numpy array of points to denote memory, with 0 means being invisible. We also append 99 extra points after the last point    
        memory_indices0 = np.zeros(visible_indices[-1]+100, dtype=int)
        memory_indices0[visible_indices] = 100
        if len(ff_catched_T_sorted)-1 >= i:
          # Find the index of the time at which the ff is captured
          last_live_time = np.where(monkey_t_array <= ff_catched_T_sorted[i])[0][-1]
          # Truncate memory_indices0 based on that 
          memory_indices0 = memory_indices0[:last_live_time+1]
        # Iterate through memory_indices0 to make a new list to denote memory (replacing some 0s with other numbers based on time)
        # We preserve the first element of memory_indices0. We also separate memory_indices and point_indices (denoted as final_indices)
        memory_indices = [memory_indices0[0]]
        for k in range(1, len(memory_indices0)):
          if memory_indices0[k] == 0:
            memory_indices.append(memory_indices[k-1]-1)
          else: # Else, preserve the current value
            memory_indices.append(memory_indices0[k])
        memory_indices_array = np.array(memory_indices)
        index_array0 = np.arange(len(memory_indices0))
        if len(index_array0) > len(monkey_t_array):
          max_index = len(monkey_t_array)
          index_array0 = index_array0[:max_index]
          memory_indices_array = memory_indices_array[index_array0]
        in_memory_indices = np.where(memory_indices_array > 0)[0]
        memory_indices_array = memory_indices_array[in_memory_indices]
        index_array = index_array0[in_memory_indices]
        in_memory_length = len(memory_indices_array)
        memory_indices = memory_indices_array.tolist()
        final_indices = index_array.tolist()
        # Append the values for this ff; Using list is faster than np.append
        ff_index = ff_index + [i]*in_memory_length
        point_index = point_index+ [point + num_missed_index for point in final_indices]
        relevant_time = monkey_t_array[index_array]
        time = time+relevant_time.tolist()
        target_index = target_index + np.digitize(relevant_time, ff_catched_T_sorted).tolist()
        ff_x = ff_x + [(ff_real_position_sorted[i][0])]*in_memory_length
        ff_y = ff_y + [(ff_real_position_sorted[i][1])]*in_memory_length
        monkey_x = monkey_x + monkey_x_array[index_array].tolist()
        monkey_y = monkey_y + monkey_y_array[index_array].tolist()
        visible = visible + [1 if point ==100 else 0 for point in memory_indices]
        memory = memory + memory_indices
        monkey_xy_relevant = np.stack([monkey_x_array[index_array], monkey_y_array[index_array]],axis=1)
        monkey_angle_relevant = monkey_angle_array[index_array]
        ff_distance_relevant = LA.norm(monkey_xy_relevant-ff_real_position_sorted[i], axis=1)
        ff_distance = ff_distance + ff_distance_relevant.tolist()
        angles_to_monkey = np.arctan2(ff_real_position_sorted[i,1]-monkey_xy_relevant[:,1], ff_real_position_sorted[i,0]-monkey_xy_relevant[:,0]) - monkey_angle_relevant
        angles_to_monkey[angles_to_monkey > pi] = angles_to_monkey[angles_to_monkey > pi] - 2*pi
        angles_to_monkey[angles_to_monkey < -pi] = angles_to_monkey[angles_to_monkey < -pi] + 2*pi
        angles_adjusted = np.absolute(angles_to_monkey)-np.abs(np.arcsin(np.divide(reward_boundary, np.maximum(ff_distance_relevant, reward_boundary) ))) # use torch clip to get valid arcsin input
        angles_adjusted = np.clip(angles_adjusted, 0, pi)
        angles_adjusted = np.sign(angles_to_monkey)* angles_adjusted
        ff_angle = ff_angle + angles_to_monkey.tolist()
        ff_angle_boundary = ff_angle_boundary + angles_adjusted.tolist()
        left_right = left_right + (np.array(angles_to_monkey) > 0).astype(int).tolist()
        # num_captures.append
        # distance_closest_ff.append
        # distance_2ndclosest_ff.append
        # num_ff_within.append
        # reward_distance.append
        # reward_angle.append
    # Now let's create a dictionary of the lists
    ff_dict = {'ff_index':ff_index, 'point_index':point_index, 'time':time, 'target_index':target_index,
                'ff_x':ff_x, 'ff_y':ff_y, 'monkey_x':monkey_x, 'monkey_y':monkey_y, 'visible':visible,
                'memory':memory, 'ff_distance':ff_distance, 'ff_angle':ff_angle, 'ff_angle_boundary': ff_angle_boundary, 'left_right':left_right}
    ff_dataframe = pd.DataFrame(ff_dict)
    if truncate == True:
      ff_dataframe = ff_dataframe[ff_dataframe['time'] < ff_catched_T_sorted[-200]]
    ff_dataframe['target_x'] = ff_real_position_sorted[np.array(ff_dataframe['target_index'])][:,0]
    ff_dataframe['target_y'] = ff_real_position_sorted[np.array(ff_dataframe['target_index'])][:,1]
    ff_dataframe['ffdistance2target'] = LA.norm(np.array(ff_dataframe[['ff_x', 'ff_y']])-np.array(ff_dataframe[['target_x', 'target_y']]), axis = 1)
    if data_folder_name:
      filepath = data_folder_name + '/ff_dataframe.csv'
      os.makedirs(data_folder_name, exist_ok = True)
      ff_dataframe.to_csv(filepath) 
    return ff_dataframe
    """### make ff_dataframe (agent)
    (only include ffs that are in obs space)
    """
  else:
    # Let's use data from monkey_information. But we shall cut off portion that is before the time of capturing the first target
    monkey_t_array0 = np.array(monkey_information['monkey_t'])
    monkey_x_array0 = np.array(monkey_information['monkey_x'])
    monkey_y_array0 = np.array(monkey_information['monkey_y'])
    monkey_angle_array0 = np.array(monkey_information['monkey_angle'])
    valid_index = np.where(monkey_t_array0 > ff_catched_T_sorted[0])[0]
    num_missed_index = valid_index[0]
    monkey_t_array = monkey_t_array0[valid_index]
    monkey_x_array = monkey_x_array0[valid_index]
    monkey_y_array = monkey_y_array0[valid_index]
    monkey_angle_array = monkey_angle_array0[valid_index]
    #index_array = np.array(range(len(monkey_t_array))) this overlaps with another variable later

    max_distance = 400
    reward_boundary = 25
    #max_time = 400
    #max_time = ff_catched_T_sorted[1251] #replace with ff_catched_T_sorted[-1]
    ff_index = []
    point_index = []
    time = []
    target_index = []
    ff_x = []
    ff_y = []
    monkey_x = []
    monkey_y = []
    visible = []
    memory = []
    ff_distance = []
    ff_angle = []
    ff_angle_boundary = []
    left_right = []
    num_captures = []
    distance_closest_ff =[]
    distance_2ndclosest_ff = []
    num_ff_within = []
    reward_distance = []
    reward_angle = []
    total_ff_num = len(ff_life_sorted)


    for i in range(total_ff_num):

      if i % 100 == 0:
        print(i," out of ", total_ff_num)


      # Go through every visible duration of the same ff
      ff_flash = ff_flash_sorted[i]
      whether_in_obs = []
      original_index = sort_indices_all[i]
      for index in valid_index:
        obs_ff_indices = obs_ff_overall_indices_all[index]
        if original_index in obs_ff_indices:
          whether_in_obs.append(True) 
        else:
          whether_in_obs.append(False)

      
      cum_indices = np.array(whether_in_obs).nonzero()[0]
      if len(cum_indices) > 0:

        updated_t_array = monkey_t_array[cum_indices]
        visible_indices = cum_indices
        index_array = cum_indices
        in_memory_length = len(index_array)
          
        cum_t, cum_angle = monkey_t_array[cum_indices], monkey_angle_array[cum_indices]
        cum_mx, cum_my = monkey_x_array[cum_indices], monkey_y_array[cum_indices]
        distances_to_monkey = LA.norm(np.stack([cum_mx, cum_my], axis=1)-ff_real_position_sorted[i], axis = 1)
        angles_to_monkey = np.arctan2(ff_real_position_sorted[i,1]-cum_my, ff_real_position_sorted[i,0]-cum_mx)-cum_angle
        angles_to_monkey[angles_to_monkey > pi] = angles_to_monkey[angles_to_monkey > pi] - 2*pi
        angles_to_monkey[angles_to_monkey < -pi] = angles_to_monkey[angles_to_monkey < -pi] + 2*pi
        angles_adjusted = np.absolute(angles_to_monkey)-np.abs(np.arcsin(np.divide(reward_boundary, np.maximum(distances_to_monkey, reward_boundary) ))) # use torch clip to get valid arcsin input
        angles_adjusted = np.clip(angles_adjusted, 0, pi)
        angles_to_monkey = np.sign(angles_to_monkey)* angles_adjusted



        # Make a numpy array of points to denote memory, with 0 means being invisible. We also append 99 extra points after the last point    
        memory_indices0 = np.zeros(visible_indices[-1]+100, dtype=int)
        memory_indices0[visible_indices] = 100

        if len(ff_catched_T_sorted)-1 >= i:
          # Find the index of the time at which the ff is captured
          last_live_time = np.where(monkey_t_array <= ff_catched_T_sorted[i])[0][-1]
          # Truncate memory_indices0 based on that 
          memory_indices0 = memory_indices0[:last_live_time+1]

        
        # Iterate through memory_indices0 to make a new list to denote memory (replacing some 0s with other numbers based on time)
        # We preserve the first element of memory_indices0. We also separate memory_indices and point_indices (denoted as final_indices)
        memory_indices = [memory_indices0[0]]

        for k in range(1, len(memory_indices0)):
          if memory_indices0[k] == 0:
            memory_indices.append(memory_indices[k-1]-1)
          else: # Else, preserve the current value
            memory_indices.append(memory_indices0[k])
        memory_indices_array = np.array(memory_indices)

        index_array0 = np.arange(len(memory_indices0))
        if len(index_array0) > len(monkey_t_array):
          max_index = len(monkey_t_array)
          index_array0 = index_array0[:max_index]
          memory_indices_array = memory_indices_array[index_array0]

        in_memory_indices = np.where(memory_indices_array > 0)[0]
        memory_indices_array = memory_indices_array[in_memory_indices]
        index_array = index_array0[in_memory_indices]
        in_memory_length = len(memory_indices_array)
        memory_indices = memory_indices_array.tolist()
        final_indices = index_array.tolist()

        ff_index = ff_index + [i]*in_memory_length
        point_index = point_index+ [point + num_missed_index for point in final_indices]
        relevant_time = monkey_t_array[index_array]
        time = time+relevant_time.tolist()
        target_index = target_index + np.digitize(relevant_time, ff_catched_T_sorted).tolist()
        ff_x = ff_x + [(ff_real_position_sorted[i][0])]*in_memory_length
        ff_y = ff_y + [(ff_real_position_sorted[i][1])]*in_memory_length
        monkey_x = monkey_x + monkey_x_array[index_array].tolist()
        monkey_y = monkey_y + monkey_y_array[index_array].tolist()
        visible = visible + [1 if point ==100 else 0 for point in memory_indices]
        memory = memory + memory_indices
        monkey_xy_relevant = np.stack([monkey_x_array[index_array], monkey_y_array[index_array]],axis=1)
        monkey_angle_relevant = monkey_angle_array[index_array]
        ff_distance_relevant = LA.norm(monkey_xy_relevant-ff_real_position_sorted[i], axis=1)
        ff_distance = ff_distance + ff_distance_relevant.tolist()
        angles_to_monkey = np.arctan2(ff_real_position_sorted[i,1]-monkey_xy_relevant[:,1], ff_real_position_sorted[i,0]-monkey_xy_relevant[:,0]) - monkey_angle_relevant
        angles_to_monkey[angles_to_monkey > pi] = angles_to_monkey[angles_to_monkey > pi] - 2*pi
        angles_to_monkey[angles_to_monkey < -pi] = angles_to_monkey[angles_to_monkey < -pi] + 2*pi
        angles_adjusted = np.absolute(angles_to_monkey)-np.abs(np.arcsin(np.divide(reward_boundary, np.maximum(ff_distance_relevant, reward_boundary) ))) # use torch clip to get valid arcsin input
        angles_adjusted = np.clip(angles_adjusted, 0, pi)
        angles_adjusted = np.sign(angles_to_monkey)* angles_adjusted
        ff_angle = ff_angle + angles_to_monkey.tolist()
        ff_angle_boundary = ff_angle_boundary + angles_adjusted.tolist()
        left_right = left_right + (np.array(angles_to_monkey) > 0).astype(int).tolist()
    # Now let's create a dictionary of the lists
    ff_dict = {'ff_index':ff_index, 'point_index':point_index, 'time':time, 'target_index':target_index,
                  'ff_x':ff_x, 'ff_y':ff_y, 'monkey_x':monkey_x, 'monkey_y':monkey_y, 'visible':visible,
                  'memory':memory, 'ff_distance':ff_distance, 'ff_angle':ff_angle, 'ff_angle_boundary': ff_angle_boundary, 'left_right':left_right}
    ff_dataframe = pd.DataFrame(ff_dict)
    ff_dataframe['target_x'] = ff_real_position_sorted[np.array(ff_dataframe['target_index'])][:,0]
    ff_dataframe['target_y'] = ff_real_position_sorted[np.array(ff_dataframe['target_index'])][:,1]
    ff_dataframe['ffdistance2target'] = LA.norm(np.array(ff_dataframe[['ff_x', 'ff_y']])-np.array(ff_dataframe[['target_x', 'target_y']]), axis = 1)
    if data_folder_name:
      filepath = data_folder_name + '/ff_dataframe.csv'
      os.makedirs(data_folder_name, exist_ok = True)
      ff_dataframe.to_csv(filepath)
    ff_dataframe.to_csv(filepath)
    return ff_dataframe 