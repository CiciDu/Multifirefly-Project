from functions.basic_func import *
import os
import numpy as np
import pandas as pd
import math
import collections
import torch
from math import pi
from numpy import linalg as LA
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)



def make_ff_dataframe(monkey_information, ff_catched_T_sorted, ff_flash_sorted,  
                    ff_real_position_sorted, ff_life_sorted, player = "monkey", 
                    max_distance = 400, reward_boundary = 25, max_memory = 100,
                    data_folder_name = None, num_missed_index = None, print_progress = True, 
                    truncate=False, obs_ff_indices_in_ff_dataframe = None):

  """
  Make a dataframe called ff_dataframe that contains various information about all visible or "in-memory" fireflies at each time point


  Parameters
  ----------
  monkey_information: dict
      containing the speed, angle, and location of the monkey at various points of time
  ff_catched_T_sorted: np.array
      containing the time when each captured firefly gets captured
  ff_flash_sorted: list
      containing the time that each firefly flashes on and off
  ff_real_position_sorted: np.array
      containing the real locations of the fireflies
  ff_life_sorted: np.array
      containing the time that each firefly comes into being and gets captured 
      (if the firefly is never captured, then capture time is replaced by the last point of time in data)
  player: str
      "monkey" or "agent" 
  max_distance: num
      the distance beyond which the firefly cannot be considered visible
  reward_boundary: num
      the reward boundary of a firefly; the current setting of the game sets it to be 25
  max_memory: num
      the numeric value of the variable "memory" for a firefly when it's fully visible
  data_folder_name: str, default is None
      the place to store the output as a csv
  num_missed_index: num, default is None
      the number of invalid indices at the beginning of any array in monkey_information;
      if default is used, then it will be calculated as the number of indices till the capture of first the firefly
  print_progress: bool
      whether to print the progress of making ff_dataframe
  truncate: bool
      whether to truncate the end of ff_dataframe by 200 captured fireflies; generally applied to monkey data
      because the information (such as in ff_real_positions_sorted) are incomplete towards the end
  obs_ff_indices_in_ff_dataframe: list
      a variable to be passed if the player is "agent"; it contains the correct indices of fireflies 

  Returns
  -------
  ff_dataframe: pd.dataframe
      containing various information about all visible or "in-memory" fireflies at each time point
      

  """

  # Let's use data from monkey_information. But we shall cut off portion that is before the time of capturing the first target
  monkey_t_array0 = np.array(monkey_information['monkey_t'])
  if num_missed_index is None:
    valid_index = np.where(monkey_t_array0 > ff_catched_T_sorted[0])[0]
    num_missed_index = valid_index[0]
  else:
    valid_index = np.arange(num_missed_index, len(monkey_t_array0))
  monkey_t_array = monkey_t_array0[num_missed_index:]
  monkey_x_array = np.array(monkey_information['monkey_x'])[num_missed_index:]
  monkey_y_array = np.array(monkey_information['monkey_y'])[num_missed_index:]
  monkey_angle_array = np.array(monkey_information['monkey_angle'])[num_missed_index:]

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
  total_ff_num = len(ff_life_sorted)

  # For each firefly in the data (except the one captured first) 

  starting_ff = {"monkey": 1, "agent": 0}
  for i in range(starting_ff[player], total_ff_num):
    current_ff_index = i
    if player == "monkey":
      current_ff_index = i
      # Go through every visible duration of the same ff
      ff_flash = ff_flash_sorted[i]
      # visible_indices contains the indices of the points when the ff is visible (within a suitable distance & at the right angle)
      visible_indices = []
      for j in range(len(ff_flash)):
        visible_duration = ff_flash[j]
        # Find the corresponding monkey information:
        cum_indices = np.where((monkey_t_array >= visible_duration[0]) & (monkey_t_array <= visible_duration[1]))[0]
        cum_mx, cum_my, cum_angle = monkey_x_array[cum_indices], monkey_y_array[cum_indices], monkey_angle_array[cum_indices]
        distance_to_ff = LA.norm(np.stack([cum_mx, cum_my], axis=1)-ff_real_position_sorted[i], axis = 1)
        valid_distance_indices = np.where(distance_to_ff < max_distance)[0]
        if len(valid_distance_indices) > 0:
          angle_to_ff = np.arctan2(ff_real_position_sorted[i,1]-cum_my[valid_distance_indices], ff_real_position_sorted[i,0]-cum_mx[valid_distance_indices])-cum_angle[valid_distance_indices]
          angle_to_ff = np.remainder(angle_to_ff, 2*pi)
          angle_to_ff[angle_to_ff > pi] = angle_to_ff[angle_to_ff > pi] - 2*pi
          # Adjust the angles according to the reward boundary
          angles_adjusted = np.absolute(angle_to_ff)-np.abs(np.arcsin(np.divide(reward_boundary, np.maximum(distance_to_ff[valid_distance_indices], reward_boundary) ))) # use torch clip to get valid arcsin input
          angles_adjusted = np.sign(angle_to_ff) * np.clip(angles_adjusted, 0, pi)
          # Find the indices of the points where the ff is both within a max_distance and valid angles
          overall_valid_indices = valid_distance_indices[np.where(np.absolute(angles_adjusted) <= 2*pi/9)[0]]
          # Store these points from the current duration into visible_indices
          visible_indices = visible_indices + cum_indices[overall_valid_indices].tolist()
      # After iterate through every duration for this firefly, we turn visible_indices into an array
      visible_indices = np.array(visible_indices)
      
    else: # Otherwise, if the player is "agent"  
      # We'll only consider the points of time when the ff of interest was in obs space
      whether_in_obs = []
      # iterate through every point (step taken by the agent)
      for index in valid_index:
        # take out all the fireflies in the obs space
        obs_ff_indices = obs_ff_indices_in_ff_dataframe[index]
        # if the ff of interest was in the obs space
        if current_ff_index in obs_ff_indices:
          whether_in_obs.append(True) 
        else:
          whether_in_obs.append(False)
      # find the point indices where the ff of interest was in the obs space
      cum_indices = np.array(whether_in_obs).nonzero()[0]
      if len(cum_indices) == 0:
        # The ff of interest has never been in the obs space, so we move on to the next ff
        continue
      else: 
        visible_indices = cum_indices

    # The following part is once again shared by "monkey" and "agent"
    if len(visible_indices) > 0:
      # Make an array of points to denote memory, with 0 means being invisible, and 100 being fully visible. 
      # After a firefly turns from being visible to being invisible, memory will decrease by 1 for each additional step taken by the monkey/agent.    
      # We append max_memory elements at the end of initial_memory_array to aid iteration through this array later
      initial_memory_array = np.zeros(visible_indices[-1]+max_memory, dtype=int)
      # Make sure that the points where the ff is fully visible has a memory of max_memory (100 by default)
      initial_memory_array[visible_indices] = max_memory
      
      # See if the current ff has been captured at any point
      # If it has been captured, then its index i should be smaller than the number of caught fireflies (i.e. the number of elements in ff_catched_T_sorted)
      if i < len(ff_catched_T_sorted):
        # Find the index of the time at which the ff is captured
        last_live_time = np.where(monkey_t_array <= ff_catched_T_sorted[i])[0][-1]
        # Truncate initial_memory_array so that its last point does not exceed last_live_time
        initial_memory_array = initial_memory_array[:last_live_time+1]
      
      # We preserve the first element of initial_memory_array and then iterate through initial_memory_array to make a new list to 
      # denote memory (replacing some 0s with other numbers based on time). 
      memory_array = [initial_memory_array[0]]
      for k in range(1, len(initial_memory_array)):
        # If the ff is currently invisible
        if initial_memory_array[k] == 0:
          # Then its memory is the memory from the previous point minus one
          memory_array.append(memory_array[k-1]-1)
        else: # Else, the firefly is visible
          memory_array.append(max_memory)
      memory_array = np.array(memory_array)
      # We need to make sure that the length of memory_array does not exceed the number of data points in monkey_t_array
      if len(memory_array) > len(monkey_t_array):
        # We also truncate memory_array so that its length does not surpass the length of monkey_t_array
        memory_array = memory_array[:len(monkey_t_array)]
      

      # Find the point indices where the firefly is in memory or visible
      in_memory_indices = np.where(memory_array > 0)[0]
      # Find the corresponding memory for these points and only keep those, since we don't need information of the ff
      # when the ff is neither visible nor in memory
      memory_array = memory_array[in_memory_indices]
      num_points_in_memory = len(memory_array)
      
      # Append the values for this ff; Using list operations is faster than np.append here
      ff_index = ff_index + [current_ff_index] * num_points_in_memory
      point_index = point_index + [point + num_missed_index for point in in_memory_indices.tolist()]
      relevant_time = monkey_t_array[in_memory_indices]
      time = time + relevant_time.tolist()
      target_index = target_index + np.digitize(relevant_time, ff_catched_T_sorted).tolist()
      ff_x = ff_x + [(ff_real_position_sorted[current_ff_index][0])]*num_points_in_memory
      ff_y = ff_y + [(ff_real_position_sorted[current_ff_index][1])]*num_points_in_memory
      monkey_x = monkey_x + monkey_x_array[in_memory_indices].tolist()
      monkey_y = monkey_y + monkey_y_array[in_memory_indices].tolist()
      visible = visible + [1 if point == 100 else 0 for point in memory_array.tolist()]
      memory = memory + memory_array.tolist()
      # In the following, "relevant" means in memory
      monkey_xy_relevant = np.stack([monkey_x_array[in_memory_indices], monkey_y_array[in_memory_indices]],axis=1)
      ff_distance_relevant = LA.norm(monkey_xy_relevant-ff_real_position_sorted[current_ff_index], axis=1)
      ff_distance = ff_distance + ff_distance_relevant.tolist()
      monkey_angle_relevant = monkey_angle_array[in_memory_indices]
      angle_to_ff = np.arctan2(ff_real_position_sorted[current_ff_index, 1]-monkey_xy_relevant[:, 1], ff_real_position_sorted[current_ff_index,0]-monkey_xy_relevant[:,0]) - monkey_angle_relevant
      angle_to_ff = np.remainder(angle_to_ff, 2*pi)
      angle_to_ff[angle_to_ff > pi] = angle_to_ff[angle_to_ff > pi] - 2*pi
      angles_adjusted = np.absolute(angle_to_ff)-np.abs(np.arcsin(np.divide(reward_boundary, np.maximum(ff_distance_relevant, reward_boundary) ))) # use torch clip to get valid arcsin input
      angles_adjusted = np.clip(angles_adjusted, 0, pi)
      angles_adjusted = np.sign(angle_to_ff) * angles_adjusted
      ff_angle = ff_angle + angle_to_ff.tolist()
      ff_angle_boundary = ff_angle_boundary + angles_adjusted.tolist()
      left_right = left_right + (np.array(angle_to_ff) > 0).astype(int).tolist()

      if i % 100 == 0:
        if print_progress:
          print(i, " out of ", total_ff_num)


  # Now let's create a dictionary from the lists
  ff_dict = {'ff_index': ff_index, 'point_index': point_index, 'time': time, 'target_index': target_index,
              'ff_x': ff_x, 'ff_y': ff_y, 'monkey_x': monkey_x, 'monkey_y': monkey_y, 'visible': visible,
              'memory': memory, 'ff_distance': ff_distance, 'ff_angle': ff_angle, 'ff_angle_boundary': ff_angle_boundary, 
              'left_right': left_right}
  ff_dataframe = pd.DataFrame(ff_dict)

  if truncate is True:
    if player == "monkey":
        ff_dataframe = ff_dataframe[ff_dataframe['time'] < ff_catched_T_sorted[-200]]
    else:
        ff_dataframe = ff_dataframe[ff_dataframe['time'] < ff_catched_T_sorted[-1]]
      
  # Add some columns
  ff_dataframe['target_x'] = ff_real_position_sorted[np.array(ff_dataframe['target_index'])][:, 0]
  ff_dataframe['target_y'] = ff_real_position_sorted[np.array(ff_dataframe['target_index'])][:, 1]
  ff_dataframe['ffdistance2target'] = LA.norm(np.array(ff_dataframe[['ff_x', 'ff_y']])-np.array(ff_dataframe[['target_x', 'target_y']]), axis = 1)

  # if a path is provided, then we will store the dataframe as a csv in the provided path
  if data_folder_name:
    filepath = data_folder_name + '/ff_dataframe.csv'
    os.makedirs(data_folder_name, exist_ok = True)
    ff_dataframe.to_csv(filepath)
  return ff_dataframe






def n_ff_in_a_row_func(ff_believed_position_sorted, distance_between_ff = 50):
	"""
  For each captured firefly, find how many fireflies have been caught in a row.
  For every two consequtive fireflies to be considered caught in a row, 
  they should not be more than 50 cm (or "distance_between_ff" cm) apart


  Parameters
  ----------
  catched_ff_num: numeric
  	total number of catched firefies
  ff_believed_position_sorted: np.array
    containing the locations of the monkey (or agent) when each captured firefly was captured 
  distance_between_ff: numeric
  	the maximum distance between two consecutive fireflies for them to be considered as caught in a row
  
  Returns
  -------
  n_ff_in_a_row: array
    containing one integer for each captured firefly to indicate how many fireflies have been caught in a row.
    n_ff_in_a_row[k] will denote the number of ff that the monkey has captured in a row at trial k

  """
  # For the first caught firefly, it is apparent that only 1 firefly has been caught in a row
	n_ff_in_a_row = [1]
	# Keep a count of how many fireflies have been caught in a row
	count = 1
	catched_ff_num = len(ff_believed_position_sorted)
	for i in range(1, catched_ff_num):
	  if LA.norm(ff_believed_position_sorted[i]-ff_believed_position_sorted[i-1]) < distance_between_ff:
	    count += 1
	  else:
	  	# Restarting from 1
	    count = 1
	  n_ff_in_a_row.append(count)
	n_ff_in_a_row = np.array(n_ff_in_a_row)
	return n_ff_in_a_row



def on_before_last_one_func(ff_flash_end_sorted, ff_catched_T_sorted, catched_ff_num):
	"""
  Find the trials where the current target has only flashed on before the capture of the previous target;
  In other words, the target hasn’t flashed on during the trial

  Parameters
  ----------
  ff_flash_end_sorted: np.array
      containing the last moment that each firefly flashes on
  ff_catched_T_sorted: np.array
      containing the time when each captured firefly gets captured
  catched_ff_num: numeric
      total number of catched firefies

  
  Returns
  -------
  on_before_last_one_trials: array
      trial numbers that can be categorized as "on before last one"

  """
	on_before_last_one_trials = [] 
	for i in range(1, catched_ff_num):
	  # Evaluate whether the last flash of the current ff finishes before the capture of the previous ff
	  if ff_flash_end_sorted[i] < ff_catched_T_sorted[i-1]:
	    # If the monkey captures 2 fireflies at the same time, then the trial does not count as "on_before_last_one"
	    if ff_catched_T_sorted[i] == ff_catched_T_sorted[i-1]:
	       continue
	    # Otherwise, append the trial number into the list
	    on_before_last_one_trials.append(i)
	on_before_last_one_trials = np.array(on_before_last_one_trials)
	return on_before_last_one_trials



def visible_before_last_one_func(ff_dataframe):
	"""
  Find the trials where the current target has only been visible on before the capture of the previous target;
  In other words, the target hasn’t been visible during the trial;
  Here, a firefly is considered visible if it satisfies: (1) flashes on, (2) Within 40 degrees to the left and right,
  (3) Within 400 cm to the monkey (the distance can be updated when the information of the actual experiment is available)

  Parameters
  ----------
  ff_dataframe: pd.dataframe
      containing various information about all visible or "in-memory" fireflies at each time point

  Returns
  -------
  visible_before_last_one_trials: array
      trial numbers that can be categorized as "visible before last one"

  """
  # We first take out the trials that cannot be categorized as "visible before last one";
  # For these trials, the target has been visible for at least one time point during the trial 
	temp_dataframe = ff_dataframe[(ff_dataframe['target_index'] == ff_dataframe['ff_index']) & (ff_dataframe['visible'] == 1)]
	trials_not_to_select = np.unique(np.array(temp_dataframe['target_index']))
	# Get the numbers for all trials
	all_trials = np.unique(np.array(ff_dataframe['target_index']))
	# Using the difference to get the trials of interest
	visible_before_last_one_trials = np.setdiff1d(all_trials, trials_not_to_select)
	return visible_before_last_one_trials




def disappear_latest_func(ff_dataframe):
	"""
  Find trials where the target has disappeared the latest among all visible fireflies during a trial

  Parameters
  ----------
  ff_dataframe: pd.dataframe
      containing various information about all visible or "in-memory" fireflies at each time point
  
  Returns
  -------
  disappear_latest_trials: array
      trial numbers that can be categorized as "disappear latest"

  """
	ff_dataframe_visible = ff_dataframe[(ff_dataframe['visible'] == 1)]
	# For each trial, find out the point index where the monkey last sees a ff
	last_visible_index = ff_dataframe_visible[['point_index', 'target_index']].groupby('target_index').max()
	# Take out all the rows corresponding to these points
	last_visible_ffs = pd.merge(last_visible_index, ff_dataframe_visible, how="left")
	# Select trials where the target disappears the latest
	disappear_latest_trials = np.array(last_visible_ffs[last_visible_ffs['target_index']==last_visible_ffs['ff_index']]['target_index'])
	return disappear_latest_trials


    



def make_point_vs_cluster(ff_dataframe, max_ff_distance_from_monkey = 250, max_cluster_distance = 100, max_time_past = 1, 
						  print_progress = True, data_folder_name = None):
    """
    Find trials where the target has disappeared the latest among all visible fireflies during a trial


    Parameters
    ----------
    ff_dataframe: pd.dataframe
    	containing various information about all visible or "in-memory" fireflies at each time point
  	max_ff_distance_from_monkey: numeric
  		the maximum distance a firefly can be from the monkey to be included in the consideration of whether it belongs to a cluster 
  	max_cluster_distance: numeric
  		the maximum distance a firefly can be from the closest firefly in a cluster to be considered as part of that same cluster
  	max_time_past: numeric
  		how long a firefly can be stored in memomry after it becomes invisible for it to be included in the consideration of whether it belongs to a cluster
  	print_progress: bool
  		whether to print the progress of making point_vs_cluster

    Returns
    -------
    point_vs_cluster: array 
    	contains indices of fireflies belonging to a cluster at each time point
    	structure: [[point_index, ff_index, cluster_label], [point_index, ff_index, cluster_label], ...]
        

    """

    # Find the beginning and ending of the indices of all steps
    min_point_index = np.min(np.array(ff_dataframe['point_index']))
    max_point_index = np.max(np.array(ff_dataframe['point_index']))
    
    # Convert max_time_past to max_steps_past
    duration_per_step = ff_dataframe['time'][100]-ff_dataframe['time'][99]
    max_steps_past = math.floor(max_time_past/duration_per_step)

    # Since the "memory" of a firefly is 100 when it is visible, and decrease by 1 for each step that it is not visible,
    # thus, the minimum "memory" of a firefly to be included in the consideration of whether it belongs to a cluster is
    # 100 - max_steps_apart 
    min_memory_of_ff = 100-max_steps_past

    # Initiate a list to store the result
    # Structure: [[point_index, ff_index, cluster_label], [point_index, ff_index, cluster_label], ...]
    point_vs_cluster = []         

    for i in range(min_point_index, max_point_index+1):
      # Take out fireflies that meet the criteria
      selected_ff = ff_dataframe[(ff_dataframe['point_index']==i)&(ff_dataframe['memory']> (min_memory_of_ff))& (ff_dataframe['ff_distance']<max_ff_distance_from_monkey)][['ff_x', 'ff_y', 'ff_index']]
      # Put their x, y coordinates into an array
      ffxy_array = selected_ff[['ff_x', 'ff_y']].to_numpy()
      if len(ffxy_array) > 1:
      	# Find their indices
        ff_indices = selected_ff[['ff_index']].to_numpy()

        # Use the function scipy.cluster.hierarchy.linkage
        # method = "single" means that we're using the Nearest Point Algorithm
        # In the returned array, each row is a procedure of combining two components into one cluster (the closer two components are, the sooner they'll be combined); 
        # there are n-1 rows in total in the returned variable
        linked = linkage(ffxy_array, method='single')

        # In order to find the number of clusters that satisfy our requirements, we'll first find the number of rows from linked
        # where the two components of the cluster are greater than max_cluster_distance (100 cm as default). 
        # If this number is n, then it means that at one point in the procedure, there are n+1 clusters that all satisfy our requirements, 
        # and then these clusters are condensed into 1 big cluster through combining two clusters into one and repearting the procedure;
        # However, these final steps of combining are invalid for our use since all the n+1 clusters are further apart than max_cluster_distance;
        # Thus, we know that we can separate the fireflies into at most n+1 clusters that satisfy our requirement
        # that each firefly is within 50 cm of its nearest firefly in the cluster.
        num_clusters = sum(linked[:, 2] > max_cluster_distance)+1  

        # If the number of clusters is smaller than the number of fireflies, then there is at least one cluster that has two or more fireflies
        if num_clusters < len(ff_indices):
          # This time, we assign the fireflies into the number of clusters we just found, so that we know which cluster each firefly belongs to.
          # The variable "cluster_labels" contains the cluster label for each firefly
          cluster_labels = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='single').fit_predict(ffxy_array)
          # Find the clusters that have more than one firefly
          unique_cluster_labels, ff_counts = np.unique(cluster_labels, return_counts=True)
          clusters_w_more_than_1_ff = unique_cluster_labels[ff_counts > 1]
          # Find the indices of the fireflies belonging to those clusters
          for index in np.isin(cluster_labels, clusters_w_more_than_1_ff).nonzero()[0]:
          	# Store the firefly index with its cluster label along with the point index i
            point_vs_cluster.append([i, ff_indices[index].item(), cluster_labels[index]])
      
      if i % 1000 == 0:
        if print_progress == True:
          print("Progress of making point_vs_cluster: ", i, " out of ", max_point_index)
    
    point_vs_cluster = np.array(point_vs_cluster)

    filepath = data_folder_name + '/point_vs_cluster.csv'
    os.makedirs(data_folder_name, exist_ok = True)
    np.savetxt(filepath, point_vs_cluster, delimiter=',')
    return point_vs_cluster




def clusters_of_ffs_func(point_vs_cluster, monkey_information, ff_catched_T_sorted):
	"""
  Find clusters of fireflies that appear during a trial based on point_vs_cluster

  Parameters
  ----------
  point_vs_cluster: array 
  	contains indices of fireflies belonging to a cluster at each time point
  	structure: [[point_index, ff_index, cluster_label], [point_index, ff_index, cluster_label], ...]
  monkey_information: dict
    containing the speed, angle, and location of the monkey at various points of time
  ff_catched_T_sorted: np.array
    containing the time when each captured firefly gets captured

  Returns
  -------
	cluster_exist_trials: array
		trial numbers of the trials where at least one cluster exists
	cluster_dataframe_point: dataframe
		information of the clusters for each time point that has at least one cluster
	cluster_dataframe_trial: dataframe
		information of the clusters for each trial that has at least one cluster
  """

  # Turn point_vs_cluster from np.array into a dataframe
	temp_dataframe1 = pd.DataFrame(point_vs_cluster, columns=['point_index', 'ff_index', 'cluster_label'])
	# Find indices of unique points and their counts and make them into a dataframe as well
	unique_points, counts = np.unique(point_vs_cluster[:, 0], return_counts=True)
	temp_dataframe2 = pd.DataFrame(np.concatenate([unique_points.reshape(-1, 1), counts.reshape(-1, 1)], axis=1),
								   columns=['point_index', 'num_ff_at_point'])
	# Combine the information of the above 2 dataframes
	temp_dataframe3 = temp_dataframe1.merge(temp_dataframe2, how="left", on="point_index")
	# Find the corresponding time to all the points
	corresponding_t = monkey_information['monkey_t'][np.array(temp_dataframe3['point_index'])]
	temp_dataframe3['time'] = corresponding_t
	# From the time of each point, find the target index that corresponds to that point
	temp_dataframe3['target_index'] = np.digitize(corresponding_t, ff_catched_T_sorted)
	# Only keep the part of the data up to the capture of the last firefly
	temp_dataframe3 = temp_dataframe3[temp_dataframe3['target_index'] < len(ff_catched_T_sorted)]
	# Thus we have the information of the clusters for each time point that has at least one cluster
	cluster_dataframe_point = temp_dataframe3
	# By grouping the information into trials, we can have the information of the clusters for each trial that has at least one cluster;
	# For each trial, we'll have the maximum number of fireflies in all clusters as well as the number of clusters
	cluster_dataframe_trial = cluster_dataframe_point[['target_index', 'num_ff_at_point']].groupby('target_index',
									as_index=True).agg({'num_ff_at_point': ['max', 'count']})
	cluster_dataframe_trial.columns = ["max_ff_in_cluster", "num_points_w_cluster"]
	# We can also take out the trials during which at least one cluster exists
	cluster_exist_trials = np.array(cluster_dataframe_point.target_index.unique())
	return cluster_exist_trials, cluster_dataframe_point, cluster_dataframe_trial



def ffs_around_target_func(ff_dataframe, catched_ff_num, ff_catched_T_sorted, ff_real_position_sorted, 
						max_time_apart = 1.25, max_ff_distance_from_monkey = 250, max_ff_distance_from_target = 50):
	"""
  Find the trials where the target is within a cluster, as well as the locations of the fireflies in the cluster


  Parameters
  ----------
  ff_dataframe: pd.dataframe
  	containing various information about all visible or "in-memory" fireflies at each time point
  catched_ff_num: numeric
  	total number of catched firefies
  ff_catched_T_sorted: np.array
    containing the time when each captured firefly gets captured
  ff_real_position_sorted: np.array
    containing the real locations of the fireflies
	max_time_apart: numeric
		how long a firefly can be stored in memomry after it becomes invisible for it to be included in the consideration of whether it belongs to a cluster
	max_ff_distance_from_monkey: numeric
		the maximum distance a firefly can be from the monkey to be included in the consideration of whether it belongs to the cluster of the target
	max_ff_distance_from_target: numeric
    the maximum distance a firefly can be from the target to be included in the consideration of whether it belongs to the cluster 

  Returns
  -------
  ffs_around_target_trials: array
    trials where a cluster exists around the target about the time of capture
  ffs_around_target_indices: list
    for each trial, it contains the indices of fireflies around the target; 
    it contains an empty array when there is no firefly around the target
  ffs_around_target_positions: list
  	positions of the fireflies around the target for each trial 
  	(similarly, if there's none for trial i, then ffs_around_target_positions[i] is an empty numpy array)

  """
	ffs_around_target = []
	ffs_around_target_indices = []
	ffs_around_target_positions = [] 
	temp_frame = ff_dataframe[['ff_index', 'target_index', 'ff_distance', 'visible', 'time']]

	# For each trial
	for i in range(1, catched_ff_num):
	  # Take out the time of that the target is captured
	  time = ff_catched_T_sorted[i]
	  # Set the duration such that only fireflies visible in this duration will be included in the consideration of 
	  # whether it belongs to the same cluster as the target
	  duration = [time-max_time_apart, time+max_time_apart]
	  temp_frame2 = temp_frame[(temp_frame['time'] > duration[0])&(temp_frame['time'] < duration[1])]
	  # We also don't want to include the previous target and the next target into the consideration
	  target_nums = np.arange(i-1, i+2)
	  temp_frame2 = temp_frame2[~temp_frame2['ff_index'].isin(target_nums)]
	  # Lastly, we want to make sure that these fireflies are visible
	  temp_frame2 = temp_frame2[(temp_frame2['visible'] ==1)]
	  temp_frame2 = temp_frame2[temp_frame2['ff_distance'] < max_time_apart]
	  # Take out the indices
	  past_visible_ff_indices = np.unique(np.array(temp_frame2.ff_index))
	  # Get positions of these ffs
	  past_visible_ff_positions = ff_real_position_sorted[past_visible_ff_indices]
	  # See if any one of it is within max_ff_distance_from_target (50 cm by default) of the target
	  distance2target = LA.norm(past_visible_ff_positions - ff_real_position_sorted[i], axis=1)
	  close_ff_indices = np.where(distance2target < max_ff_distance_from_target)[0]
	  num_ff = len(close_ff_indices)
	  ffs_around_target.append(num_ff)
	  if num_ff > 0:
	    ffs_around_target_positions.append(past_visible_ff_positions[close_ff_indices])
	    ffs_around_target_indices.append(past_visible_ff_indices[close_ff_indices])
	  else:
	    ffs_around_target_positions.append(np.array([]))
	    ffs_around_target_indices.append(np.array([]))
	ffs_around_target = np.array(ffs_around_target)
	ffs_around_target_trials = np.where(ffs_around_target > 0)[0]
	return ffs_around_target_trials, ffs_around_target_indices, ffs_around_target_positions




def try_a_few_times_func(ff_catched_T_sorted, monkey_information, ff_believed_position_sorted, 
						 player, max_point_index, max_cluster_distance = 50):
  """
  Find the trials where the monkey has stopped more than one times to catch a target

  Parameters
  ----------
  ff_catched_T_sorted: np.array
    containing the time when each captured firefly gets captured
  monkey_information: dict
    containing the speed, angle, and location of the monkey at various points of time
  ff_believed_position_sorted: np.array
    containing the locations of the monkey (or agent) when each captured firefly was captured 
  player: str
    "monkey" or "agent"  
  max_point_index: numeric
  	the maximum point_index in ff_dataframe      
  max_cluster_distance: numeric
  	the maximum distance a firefly can be from the closest firefly in a cluster to be considered as part of that same cluster

    
  Returns
  -------
  try_a_few_times_trials: array
    trials that can be categorized as "try a few times"
  try_a_few_times_indices: array
  	indices of moments that can be categorized as "try a few times"
  try_a_few_times_indices_for_anim: array
  	indices of monkey_information that can be annotated as "try a few times" during animation; the difference between this variable and give_up_after_trying_indices 
  	is that the this variable supplies 20 points before and after each intervals to make the annotations last longer and easier to read

  """
  try_a_few_times_trials = []
  try_a_few_times_indices = []
  try_a_few_times_indices_for_anim = []
  catched_ff_num = len(ff_catched_T_sorted)
  for i in range(1, catched_ff_num): 
    # Find clusters based on a distance of max_cluster_distance (default is 50 cm)
    clusters = put_stops_into_clusters(i, max_cluster_distance, ff_catched_T_sorted, monkey_information, player = player)
    # if there is at least one cluster:
    if len(clusters) > 0:
    	# We are only interested in the last cluster 
      label_of_last_cluster = clusters[-1]
      # If the last cluster has more than 2 stops
      if clusters.count(label_of_last_cluster) > 1:
        # Find the locations of all the stops during the trial
        distinct_stops = find_stops(i, ff_catched_T_sorted, monkey_information, player = player)
        # Also find the indices (in monkey_information) of these stops
        distinct_stops_indices = find_stops(i, ff_catched_T_sorted, monkey_information, player = player, return_index = True)
        # If the last stop is close enough to the believed position of the target, then we know that the last cluster of stops 
        # are likely aimed towards the target; then, the trial can be categorized as "try a few times"
        if LA.norm(distinct_stops[-1]-ff_believed_position_sorted[i]) < max_cluster_distance:
          # Store by trials
          try_a_few_times_trials.append(i)
          # Store by indices (in regards to monkey_information)
          num_stops_in_last_cluster = clusters.count(label_of_last_cluster)
          min_index = distinct_stops_indices[-num_stops_in_last_cluster]
          max_index = distinct_stops_indices[-1]
          try_a_few_times_indices = try_a_few_times_indices + list(range(min_index, max_index+1))
          try_a_few_times_indices_for_anim = try_a_few_times_indices_for_anim + list(range(min_index-20, max_index+21))

  try_a_few_times_trials = np.array(try_a_few_times_trials)
  try_a_few_times_indices = np.array(try_a_few_times_indices)
  try_a_few_times_indices_for_anim = np.array(try_a_few_times_indices_for_anim)
  # make sure that the maximum index does not exceed the maximum point_index in ff_dataframe
  try_a_few_times_indices = try_a_few_times_indices[try_a_few_times_indices < max_point_index]
  try_a_few_times_indices_for_anim = try_a_few_times_indices_for_anim[try_a_few_times_indices_for_anim < max_point_index]

  return try_a_few_times_trials, try_a_few_times_indices, try_a_few_times_indices_for_anim



def give_up_after_trying_func(ff_catched_T_sorted, monkey_information, ff_believed_position_sorted, player, max_point_index, max_cluster_distance = 50):
  """
  Find the trials where the monkey has stopped more than once to catch a firefly but failed to succeed, and the monkey gave up

  Parameters
  ----------
  ff_catched_T_sorted: np.array
      containing the time when each captured firefly gets captured
  monkey_information: dict
      containing the speed, angle, and location of the monkey at various points of time
  ff_believed_position_sorted: np.array
      containing the locations of the monkey (or agent) when each captured firefly was captured 
  player: str
      "monkey" or "agent"  
  max_point_index: numeric
	   the maximum point_index in ff_dataframe      
  max_cluster_distance: numeric
	   the maximum distance a firefly can be from the closest firefly in a cluster to be considered as part of that same cluster

  
  Returns
  -------
  give_up_after_trying_trials: array
      trials that can be categorized as "give up after trying"
  give_up_after_trying_indices: array
      indices in monkey_information that can be categorized as "give up after trying"
  give_up_after_trying_indices_for_anim: array
      indices of monkey_information that can be annotated as "give up after trying" during animation; the difference between this variable and give_up_after_trying_indices 
      is that the this variable supplies 20 points before and after each intervals to make the annotations last longer and easier to read

  """

  # General strategy: find the trials where there at least one cluster has more than 2 stops, and this cluster is neither at the beginning nor at the end
  give_up_after_trying_trials = []
  give_up_after_trying_indices = []
  give_up_after_trying_indices_for_anim = []
  catched_ff_num = len(ff_catched_T_sorted)
  for i in range(1, catched_ff_num):
    # Find clusters based on a distance of max_cluster_distance
    clusters = put_stops_into_clusters(i, max_cluster_distance, ff_catched_T_sorted, monkey_information, player = player)
    # if clusters is not empty:
    if len(clusters) > 0:
      clusters_counts = collections.Counter(clusters) # count the number of elements in each cluster
      distinct_stop_positions = find_stops(i, ff_catched_T_sorted, monkey_information, player = player)
      distinct_stops_indices = find_stops(i, ff_catched_T_sorted, monkey_information, player = player, return_index = True)
      last_cluster_label = clusters[-1]
      for k in range(1, last_cluster_label + 1):  # for each cluster
        # if the cluster has more than one element:
        if clusters_counts[k] > 1:
          # Get positions of these points
          stop_positions = [distinct_stop_positions[index] for index, value in enumerate(clusters) if value == k]
          # If the first stop is not close to beginning, and the last stop is not too close to the end:
          if LA.norm(stop_positions[0]-ff_believed_position_sorted[i-1]) > max_cluster_distance and LA.norm(stop_positions[-1]-ff_believed_position_sorted[i]) > max_cluster_distance:
            # Store the trial number
            give_up_after_trying_trials.append(i)
            # Store indices
            stop_positions_indices = [distinct_stops_indices[index] for index, value in enumerate(clusters) if value == k]
            give_up_after_trying_indices = give_up_after_trying_indices + stop_positions_indices
            give_up_after_trying_indices_for_anim = give_up_after_trying_indices_for_anim + list(range(min(stop_positions_indices)-20, max(stop_positions_indices)+21))
            	
  give_up_after_trying_trials = np.unique(np.array(give_up_after_trying_trials))
  give_up_after_trying_indices = np.array(give_up_after_trying_indices)
  give_up_after_trying_indices_for_anim = np.array(give_up_after_trying_indices_for_anim)
  # make sure that the maximum index does not exceed the maximum point_index in ff_dataframe
  give_up_after_trying_indices = give_up_after_trying_indices[give_up_after_trying_indices < max_point_index]
  give_up_after_trying_indices_for_anim = give_up_after_trying_indices_for_anim[give_up_after_trying_indices_for_anim < max_point_index]

  return give_up_after_trying_trials, give_up_after_trying_indices, give_up_after_trying_indices_for_anim



def ignore_sudden_flash_func(ff_dataframe, ff_real_position_sorted, max_point_index, max_ff_distance_from_monkey = 50):
  """
  Find the trials where a firefly other than the target or the next target suddenly becomes visible, is within in 
  50 cm (or max_ff_distance_from_monkey) of the monkey, and is closer than the target.


  Parameters
  ----------
  ff_dataframe: pd.dataframe
  	containing various information about all visible or "in-memory" fireflies at each time point
  ff_real_position_sorted: np.array
    containing the real locations of the fireflies
  max_point_index: numeric
    the maximum point_index in ff_dataframe      
  max_ff_distance_from_monkey: numeric
    the maximum distance a firefly can be from the monkey to be included in the consideration of whether it belongs to the cluster of the target

  Returns
  -------
  ignore_sudden_flash_trials: array
    trials that can be categorized as "ignore sudden flash"
  ignore_sudden_flash_indices: array
  	indices of ff_dataframe that can be categorized as "ignore sudden flash"
  ignore_sudden_flash_indices_for_anim: array
  	indices of monkey_information that can be annotated as "ignore sudden flash" during animation; the difference between this variable and the previous one 
  	is that the the current variable supplies 121 points (2s in the original dataset) after each intervals to make the annotations last longer and easier to read


  """
  # These are the indices in ff_dataframe where a ff changes from being invisible to become visible
  start_index1 = np.where(np.ediff1d(np.array(ff_dataframe['visible'])) == 1)[0]+1
  # These are the indices in ff_dataframe where the ff_index has changed, meaning that an invisible firefly has become visible
  start_index2 = np.where(np.ediff1d(np.array(ff_dataframe['ff_index']))!= 0)[0]+1
  # Combine the two to get the indices in ff_dataframe where a ff suddenly becomes visible
  start_index3 = np.concatenate((start_index1, start_index2))
  start_index = np.unique(start_index3)

  # Among those points, take out those where the distance between the monkey and the firefly is smaller than max_ff_distance_from_monkey
  df_ffdistance = np.array(ff_dataframe['ff_distance'])
  suddent_flash_index = start_index[np.where(df_ffdistance[start_index] < 50)]



  # On the other hand, we find the indices of ff_dataframe where the suddenly visible ff is the target or next target
  condition = np.logical_or((np.array(ff_dataframe['ff_index'])[suddent_flash_index] == np.array(ff_dataframe['target_index'])[suddent_flash_index]), 
  						  (np.array(ff_dataframe['ff_index'])[suddent_flash_index] == np.array(ff_dataframe['target_index']+1)[suddent_flash_index]))



  # # If we're interested in finding the cases where the suddenly visible fireflies are captured in the current or the next trial:
  # capture_sudden_flash = suddent_flash_index[condition]
  # capture_sudden_flash_trials = np.array(ff_dataframe['target_index'])[capture_sudden_flash]
  # capture_sudden_flash_trials = np.unique(capture_sudden_flash_trials)


  # Thus, we can find the indices and trials of ff_dataframe where the suddenly visible ff is not the target
  ignore_sudden_flash = suddent_flash_index[~condition]

  # Find the distance from the monkey to the target at these points
  cum_target_distances = np.array(ff_dataframe['ffdistance2target'])[ignore_sudden_flash]
  # And find the distance from the monkey to the suddenly visible fireflies
  cum_ff_distances = df_ffdistance[ignore_sudden_flash]

  # Only keep the trials where the suddenly visible firefly is closer than the target
  valid_indices = np.where(cum_target_distances > cum_ff_distances)
  cum_target_indices = np.array(ff_dataframe.target_index)[ignore_sudden_flash]
  ignore_sudden_flash_trials = np.unique(cum_target_indices[valid_indices])

  # Find the indices in ff_dataframe corresponding to such a sudden flash
  ignore_sudden_flash_indices = np.array(ff_dataframe['point_index'])[ignore_sudden_flash[valid_indices]]

  ignore_sudden_flash_indices_for_anim = []
  for i in ignore_sudden_flash_indices:
  	# Append each point into a list and the following n points so that the message can be visible for 2 seconds
  	ignore_sudden_flash_indices_for_anim = ignore_sudden_flash_indices_for_anim+list(range(i, i+121))
  # make sure that the maximum index does not exceed the maximum point_index in ff_dataframe
  ignore_sudden_flash_indices = ignore_sudden_flash_indices[ignore_sudden_flash_indices < max_point_index]
  ignore_sudden_flash_indices_for_anim = np.array(ignore_sudden_flash_indices_for_anim)[np.where(ignore_sudden_flash_indices_for_anim <= max_point_index)]

  return ignore_sudden_flash_trials, ignore_sudden_flash_indices, ignore_sudden_flash_indices_for_anim




def whether_current_and_last_targets_are_captured_simultaneously(trial_number_arrays, ff_catched_T_sorted):
    if len(trial_number_arrays) > 0:
        dif_time = ff_catched_T_sorted[trial_number_arrays] - ff_catched_T_sorted[trial_number_arrays-1]
        trial_number_arrays_simul = trial_number_arrays[np.where(dif_time <= 0.1)[0]]
        trial_number_arrays_non_simul = trial_number_arrays[np.where(dif_time > 0.1)[0]]
    else:
        trial_number_arrays_simul = np.array([])
        trial_number_arrays_non_simul = np.array([])
    return trial_number_arrays_simul, trial_number_arrays_non_simul




def make_trials_char(ff_dataframe, monkey_information, ff_catched_T_sorted, ff_believed_position_sorted, player = "monkey", max_cluster_distance = 50, data_folder_name = None):
	"""
  Make a dataframe called trials_char that includes some information about each trial

  Parameters
  ---------- 
  ff_dataframe: pd.dataframe
  	containing various information about all visible or "in-memory" fireflies at each time point
  monkey_information: dict
      containing the speed, angle, and location of the monkey at various points of time
  ff_catched_T_sorted: np.array
      containing the time when each captured firefly gets captured
  ff_believed_position_sorted: np.array
      containing the locations of the monkey (or agent) when each captured firefly was captured    
  max_cluster_distance: numeric
      the maximum distance a firefly can be from the closest firefly in a cluster to be considered as part of that same cluster
  player: str
      "monkey" or "agent" 
  data_folder_name: str
      name or path of the folder to store the output as csv
  
  Returns
  -------
  trials_char: dataframe containing various characteristics of each trial, with the following columns:
      'trial': trial number
      't': duration of a trial 
      't_last_visible': duration since the target or a firefly near the target (within 25 cm) is last seen
      'd_last_visible': distance from the target since the target or a nearby firefly is last seen
      'abs_angle_last_visible': angle to the reward boundary of the target since the target or a nearby firefly is last seen
      'hit_boundary': whether the monkey/agent hits the boundary during a trial
      'num_stops': number of stops made during a trial 
      'num_stops_since_last_visible': number of stops made since the target or a nearby firefly is last seen
      'num_stops_near_target': number of stops made made near the target (the closest stop should be within 50 cm, or max_cluster_distance, of the target)
      'n_ff_in_a_row': number of fireflies the monkey/agent has caught in a row after catching the current target

  """

	catched_ff_num = len(ff_catched_T_sorted)
	# Make an array of trial numbers. Trial number starts at 1 and is named after the index of the target. 
	trial_array = [i for i in range(1, catched_ff_num)]

	# Find the duration of each trial
	t_array = ff_catched_T_sorted[1:catched_ff_num] - ff_catched_T_sorted[:catched_ff_num-1]
	t_array = t_array.tolist()

	# Question: How long can the monkey remember a target?
	# For each trial, find the time that elapses between the target last being visible and its capture.
	# Also find the distance and angle of the target when the target is last visible.
	t_last_visible = []
	d_last_visible = []
	abs_angle_last_visible = []
	# Take out the information about visible fireflies
	visible_ff = ff_dataframe[ff_dataframe['visible'] == 1]
	for i in range(1, catched_ff_num):
	  # Because sometimes the monkey aims for a firefly near the target but ends up catching the target, 
	  # we also consider fireflies near the targets along with the targets
	  info_of_nearby_ff = ((visible_ff['target_index']==i) & (visible_ff['ffdistance2target'] < 25))
	  info_of_target = (visible_ff['ff_index']==i)
	  relevant_df = visible_ff[ info_of_nearby_ff | info_of_target]
	  if len(relevant_df) > 0:
	    t_last_visible.append(ff_catched_T_sorted[i] - max(np.array(relevant_df.time)))
	    d_last_visible.append(max(np.array(relevant_df.ff_distance)))
	    abs_angle_last_visible.append(max(np.absolute(np.array(relevant_df.ff_angle_boundary))))
	  else:
	    t_last_visible.append(9999)
	    d_last_visible.append(9999)
	    abs_angle_last_visible.append(9999)


	# Create an array that shows whether the monkey has hit the boundary at least once during each trial
	hit_boundary = []
	for i in range(1, catched_ff_num):
	  duration = [ff_catched_T_sorted[i-1], ff_catched_T_sorted[i]]
	  cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
	  if len(cum_indices) > 1:
	    cum_t, cum_angle = monkey_information['monkey_t'][cum_indices],  monkey_information['monkey_angle'][cum_indices]
	    cum_mx, cum_my = monkey_information['monkey_x'][cum_indices], monkey_information['monkey_y'][cum_indices] 
	    if np.any(cum_mx[1:]-cum_mx[:-1] > 55) or np.any(cum_my[1:]-cum_my[:-1] > 55):
	      hit_boundary.append(True)
	    else:
	      hit_boundary.append(False)
	  else:
	    hit_boundary.append(False)

	# Create arrays about the number of stops made by the agent during each trial
	num_stops_array = [len(find_stops(i, ff_catched_T_sorted, monkey_information, player = player)) for i in range(1, catched_ff_num)]
	# Also count the number of stops since the target (or a nearby firefly) is last seen
	num_stops_since_last_visible = [len(find_stops(i, ff_catched_T_sorted, monkey_information, player = player, since_target_last_seen = True, t_last_visible = t_last_visible)) for i in range(1, catched_ff_num)]


	num_stops_near_target = []
	for i in range(1, catched_ff_num):
	  clusters = put_stops_into_clusters(i, max_cluster_distance, ff_catched_T_sorted, monkey_information, player = player)
	  # Find the locations of the stops
	  distinct_stops = find_stops(i, ff_catched_T_sorted, monkey_information, player = player)
	  if len(distinct_stops) > 0:
	    # If the last stop is close enough to the believed position of the target
	    if LA.norm(distinct_stops[-1]-ff_believed_position_sorted[i]) < max_cluster_distance:
	      # Append the number of stops in the last cluster
	      num_stops_near_target.append(clusters.count(clusters[-1]))
	    else: 
	      num_stops_near_target.append(0)
	  else: 
	    num_stops_near_target.append(0)

	n_ff_in_a_row = n_ff_in_a_row_func(ff_believed_position_sorted, distance_between_ff = 50)

	# Put all the information first into a dictionary and then into a dataframe
	trials_dict = {'trial':trial_array, 
	               't':t_array,  
	               't_last_visible':t_last_visible,
	               'd_last_visible':d_last_visible,
	               'abs_angle_last_visible': abs_angle_last_visible,
	               'hit_boundary': hit_boundary,  
	               'num_stops': num_stops_array, 
	               'num_stops_since_last_visible': num_stops_since_last_visible,
	               'num_stops_near_target': num_stops_near_target,
	               'n_ff_in_a_row': n_ff_in_a_row[:len(trial_array)].tolist()                            
	                      }
	trials_char = pd.DataFrame(trials_dict)
	
	if data_folder_name:
		filepath = data_folder_name + '/trials_char.csv'
		os.makedirs(data_folder_name, exist_ok = True)
		trials_char.to_csv(filepath)
		print("new trials_char is stored in ", filepath)
	return trials_char




def make_medians_dict(trials_char):
	"""
	Make medians_dict which contains some median values about the data

  Parameters
  ---------- 
  trials_char: dataframe containing various characteristics of each trial
  
  Returns
  -------
  medians_dict: dictionary containing some median values of the data

  """	

  # We only want to include trials where the monkey/agent has been actively engaged in the game. Thus, we decided to 
  # include only the trials whose durations are under 50s.
  # Also, to make sure that the distances and angles are valid, we only use trials in which the monkey/agent never hits the boundary of the arena
	valid_trials = trials_char[(trials_char['t_last_visible'] < 50) & (trials_char['hit_boundary']==False)].reset_index()
	median_values = valid_trials.median(axis=0)

	medians_dict = {"Median time": median_values['t'],
	"Median time target last seen": median_values['t_last_visible'],
	"Median distance target last seen": median_values['d_last_visible'], # distance is distance from the target to the monkey/agent
	"Median abs angle target last seen ": median_values['abs_angle_last_visible'],
	"Median num stops": median_values['num_stops'],
	"Median num stops near target": median_values['num_stops_near_target'],
	}

	return medians_dict



def make_category_frequency_dict_from_data(ff_dataframe, monkey_information, ff_real_position_sorted, ff_catched_T_sorted, ff_believed_position_sorted, player):
	"""
	Make category_frequency_dict which contains some statistics about the data;
	
	Note: when counting the number of trials for a category, we do not take into account the first trial (i.e. trial number = 0)


  Parameters
  ---------- 
  ff_dataframe: pd.dataframe
      containing various information about all visible or "in-memory" fireflies at each time point
  monkey_information: dict
      containing the speed, angle, and location of the monkey at various points of time
  ff_real_position_sorted: np.array
      containing the real locations of the fireflies
  ff_catched_T_sorted: np.array
      containing the time when each captured firefly gets captured
  ff_believed_position_sorted: np.array
      containing the locations of the monkey (or agent) when each captured firefly was captured    
  player: str
      "monkey" or "agent" 

  
  Returns
  -------
  category_frequency_dict: dictionary containing the following statistics of the data:
      "ff capture rate": number of captured fireflies divided by overall duration
    	"Stop success rate": number of captured fireflies divided by number of stops
    	"Two in a row" : percentage of trials categorized as "two in a row"
    	"Visible before last capture" : percentage of trials categorized as "visible before last capture"
    	"Target disappears latest" : percentage of trials categorized as "target disappears latest"
    	"Waste cluster around last target": percentage of trials categorized as "waste cluster around last target"
    	"Ignore sudden flash": percentage of trials categorized as "ignore sudden flash"
    	"Try a few times": percentage of trials categorized as "try a few times"
    	"Give up after trying": percentage of trials categorized as "give up after trying"
  """	

	max_point_index = np.max(np.array(ff_dataframe['point_index']))

	if player == "monkey":
		num_trials = len(ff_catched_T_sorted) - 200
	else:
		num_trials = len(ff_catched_T_sorted)


	beginning_of_time = ff_catched_T_sorted[0]
	end_of_time = ff_catched_T_sorted[num_trials]

	# Make an array of the number of stops made during each trial
	num_stops_array = [len(find_stops(i, ff_catched_T_sorted, monkey_information, player = player)) for i in range(1, num_trials)]


	# Find the trial numbers that belong to a category such as "n ff in a row"
	n_ff_in_a_row = n_ff_in_a_row_func(ff_believed_position_sorted, distance_between_ff = 50)

	visible_before_last_one_trials = visible_before_last_one_func(ff_dataframe)

	disappear_latest_trials = disappear_latest_func(ff_dataframe)

	ffs_around_target_trials, ffs_around_target_indices, ffs_around_target_positions = ffs_around_target_func(ff_dataframe, catched_ff_num, ff_catched_T_sorted, ff_real_position_sorted, max_time_apart = 1.25)

	waste_cluster_last_target_trials = np.intersect1d(ffs_around_target_trials+1, np.where(n_ff_in_a_row == 1)[0])

	ignore_sudden_flash_trials, ignore_sudden_flash_indices, ignore_sudden_flash_indices_for_anim = ignore_sudden_flash_func(ff_dataframe, ff_real_position_sorted, max_point_index)

	try_a_few_times_trials, try_a_few_times_indices, try_a_few_times_indices_for_anim = try_a_few_times_func(ff_catched_T_sorted, monkey_information, ff_believed_position_sorted, player, max_point_index)

	give_up_after_trying_trials, give_up_after_trying_indices, give_up_after_trying_indices_for_anim = give_up_after_trying_func(ff_catched_T_sorted, monkey_information, ff_believed_position_sorted, player, max_point_index)

	category_frequency_dict = {
	"ff capture rate": num_trials / (end_of_time - beginning_of_time),
	"Stop success rate": num_trials / sum(num_stops_array),
	"Two in a row" : len(n_ff_in_a_row) / (num_trials - 1),
	"Visible before last capture" : len(visible_before_last_one_trials) / (num_trials - 1),
	"Target disappears latest" : len(np.where(disappear_latest_trials < num_trials)[0]) / num_trials,
	"Waste cluster around last target": len(np.where(waste_cluster_last_target_trials < num_trials)[0]) / (num_trials - 1),
	"Ignore sudden flash": len(np.where(ignore_sudden_flash_trials < num_trials)[0]) / num_trials,
	"Try a few times": len(np.where(np.array(try_a_few_times_trials) < num_trials)[0]) / num_trials,
	"Give up after trying": len(np.where(np.array(give_up_after_trying_trials) < num_trials)[0]) / num_trials,
	}
	return category_frequency_dict





def make_category_frequency_dict_from_categories(ff_catched_T_sorted, monkey_information, n_ff_in_a_row, visible_before_last_one_trials, disappear_latest_trials, waste_cluster_last_target_trials, \
                    ignore_sudden_flash_trials, give_up_after_trying_trials, try_a_few_times_trials):
    # category_frequency_dict
    max_time = min(ff_catched_T_sorted[-1]-100, 2000)
    n_trial = np.where(ff_catched_T_sorted < max_time)[0][-1]
    catched_T_bounded = ff_catched_T_sorted[np.where(ff_catched_T_sorted < max_time)[0]]
    num_stops_array = [len(find_stops(i, ff_catched_T_sorted, monkey_information, player = "monkey")) for i in range(1, n_trial+1)]
    category_frequency_dict = {
    "Two in a row": len(np.where(n_ff_in_a_row[:n_trial]>=2)[0])/(n_trial-2),
    "Visible before last capture": len(np.where(visible_before_last_one_trials < n_trial)[0])/(n_trial-2),
    "Target disappears latest": len(np.where(disappear_latest_trials < n_trial)[0])/(n_trial-1),
    "Waste cluster around last target": len(np.where(waste_cluster_last_target_trials < n_trial)[0])/(n_trial-2),
    "Ignore sudden flash": len(np.where(ignore_sudden_flash_trials < n_trial)[0])/(n_trial-1),
    "Give up after trying": len(np.where(np.array(give_up_after_trying_trials) < n_trial)[0])/(n_trial-1),
    "Try a few times": len(np.where(np.array(try_a_few_times_trials) < n_trial)[0])/(n_trial-1),
    "ff capture rate": (len(catched_T_bounded)-1)/(catched_T_bounded[-1]-catched_T_bounded[0]),
    "Stop success rate": n_trial/sum(num_stops_array),
    }
    return category_frequency_dict





def compare_agents_monkey_medians_dict(medians_dict_agent, medians_dict_monkey, agent_names = ["Agent", "Agent2", "Agent3"], medians_dict_agent2 = None, medians_dict_agent3 = None):
	"""
	Make a dataframe that combines the medians_dict from the monkey and the agent(s);
	This function can include medians_dict from up to three agents. 

  Parameters
  ---------- 
  medians_dict_agent: dict
  	containing some median values about the agent data
  medians_dict_monkey: dict
  	containing some median values about the monkey data
  agent_names: list, optional
  	names of the agents used to identify the agents, if more than one agent is used
  medians_dict_agent: dict, optional
  	containing some median values about the 2nd agent's data
  medians_dict_agent: dict, optional
  	containing some median values about the 3rd agent's data

  Returns
  -------
  merged_medians_df: dataframe that combines medians_dict from the monkey and the agent(s)

  """	
  # Turn dictionaries into dataframe and add a column for identification 
	agent_medians_df = pd.DataFrame.from_dict(medians_dict_agent, orient='index', columns=['Value'])
	agent_medians_df['Player'] = agent_names[0]
	monkey_medians_df = pd.DataFrame.from_dict(medians_dict_monkey, orient='index', columns=['Value'])
	monkey_medians_df['Player'] = 'Monkey'

	if medians_dict_agent3:
		# Then a 2nd agent and a 3rd agent are used
		agent_medians_df2 = pd.DataFrame.from_dict(medians_dict_agent2, orient='index', columns=['Value'])
		agent_medians_df2['Player'] = agent_names[1]
		agent_medians_df3 = pd.DataFrame.from_dict(medians_dict_agent3, orient='index', columns=['Value'])
		agent_medians_df3['Player'] = agent_names[2]
		merged_medians_df = pd.concat([agent_medians_df, monkey_medians_df, agent_medians_df2, agent_medians_df3], axis=0)
	elif medians_dict_agent2:
		# Then a 2nd agent is used
		agent_medians_df2 = pd.DataFrame.from_dict(medians_dict_agent2, orient='index', columns=['Value'])
		agent_medians_df2['Player'] = agent_names[1]
		merged_medians_df = pd.concat([agent_medians_df, monkey_medians_df, agent_medians_df2], axis=0)
	else:
		merged_medians_df = pd.concat([agent_medians_df, monkey_medians_df], axis=0)

	merged_medians_df = merged_medians_df.reset_index()
	merged_medians_df.rename(columns = {'index':'Category'}, inplace = True)
	return merged_medians_df





def compare_agents_monkey_category_frequency_dict(category_frequency_dict_agent, category_frequency_dict_monkey, agent_names = ["Agent", "Agent2", "Agent3"], category_frequency_dict_agent2 = None, category_frequency_dict_agent3 = None):
	"""
	Make a dataframe that combines the category_frequency_dict from the monkey and the agent(s);
	This function can include category_frequency_dict from up to three agents. 

  Parameters
  ---------- 
  category_frequency_dict_agent: dict
  	containing some statistics about the agent data
  category_frequency_dict_monkey: dict
  	containing some statistics about the monkey data
  agent_names: list, optional
  	names of the agents used to identify the agents, if more than one agent is used
  category_frequency_dict_agent: dict, optional
  	containing some statistics about the 2nd agent's data
  category_frequency_dict_agent: dict, optional
  	containing some statistics about the 3rd agent's data

  Returns
  -------
  merged_stats_df: dataframe that combines category_frequency_dict from the monkey and the agent(s)

  """	
  # Turn dictionaries into dataframe and add a column for identification 
	agent_stats_df = pd.DataFrame.from_dict(category_frequency_dict_agent, orient='index', columns=['Value'])
	agent_stats_df['Player'] = agent_names[0]
	monkey_stats_df = pd.DataFrame.from_dict(category_frequency_dict_monkey, orient='index', columns=['Value'])
	monkey_stats_df['Player'] = 'Monkey'
	if category_frequency_dict_agent3:
		# Then a 2nd agent and a 3rd agent are used
		agent_stats_df2 = pd.DataFrame.from_dict(category_frequency_dict_agent2, orient='index', columns=['Value'])
		agent_stats_df2['Player'] = agent_names[1]
		agent_stats_df3 = pd.DataFrame.from_dict(category_frequency_dict_agent3, orient='index', columns=['Value'])
		agent_stats_df3['Player'] = agent_names[2]
		# Merge the dataframes
		merged_stats_df = pd.concat([agent_stats_df, monkey_stats_df, agent_stats_df2, agent_stats_df3], axis=0)
	elif category_frequency_dict_agent2:
		# Then a 2nd agent is used
		agent_stats_df2 = pd.DataFrame.from_dict(category_frequency_dict_agent2, orient='index', columns=['Value'])
		agent_stats_df2['Player'] = agent_names[1]
		# Merge the dataframes
		merged_stats_df = pd.concat([agent_stats_df, monkey_stats_df, agent_stats_df2], axis=0)
	else:
		# Then no more agent is used; we can directly merge the dataframes
		merged_stats_df = pd.concat([agent_stats_df, monkey_stats_df], axis=0)

	merged_stats_df = merged_stats_df.reset_index()
	merged_stats_df.rename(columns = {'index':'Category'}, inplace = True)
	return merged_stats_df








