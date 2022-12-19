from functions.basic_func import *
from functions.make_ff_dataframe import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
import pandas as pd
import math
import collections
import torch
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering


torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)



def n_ff_in_a_row_func(catched_ff_num, ff_believed_position_sorted, distance_between_ff = 50):
	n_ff_in_a_row = [1]
	distance_between_ff = 50
	count = 1
	# n_ff_in_a_row[k] will denote the number of ff that the monkey has captured in a row at trial k
	for i in range(1, catched_ff_num):
	  if LA.norm(ff_believed_position_sorted[i]-ff_believed_position_sorted[i-1]) < distance_between_ff:
	    count += 1
	  else:
	    count = 1
	  n_ff_in_a_row.append(count)
	n_ff_in_a_row = np.array(n_ff_in_a_row)
	return n_ff_in_a_row



## on before last one
def on_before_last_one_func(ff_flash_end_sorted, ff_catched_T_sorted, catched_ff_num):
	on_before_last_one_trials = [] # Note, the formula has been changed from the previous stages
	for i in range(1, catched_ff_num):
	  # Evaluate whether the last flash of the current ff finishes before the capture of the previous ff
	  if ff_flash_end_sorted[i] < ff_catched_T_sorted[i-1]:
	    # Then we consider special situations
	    # Make sure that the index is not out of bound
	    if i > 1: 
	      # If the monkey captures 2 fireflies at the same time, then the trial does not count as "on_before_last_one"
	      if ff_catched_T_sorted[i] == ff_catched_T_sorted[i-1]:
	        continue
	    # Append the index into the list
	    on_before_last_one_trials.append(i)
	on_before_last_one_trials = np.array(on_before_last_one_trials)
	return on_before_last_one_trials



def visible_before_last_one_func(ff_dataframe):
	temp_dataframe = ff_dataframe[(ff_dataframe['target_index']==ff_dataframe['ff_index']) & (ff_dataframe['visible'] == 1)]
	trials_not_to_select = np.unique(np.array(temp_dataframe['target_index']))
	all_trials = np.unique(np.array(ff_dataframe['target_index']))
	visible_before_last_one_trials = np.setdiff1d(all_trials, trials_not_to_select)
	return visible_before_last_one_trials




def disappear_latest_func(ff_dataframe):
	#By trial
	ff_dataframe_visible = ff_dataframe[(ff_dataframe['visible']==1)]
	# For each trial, find out the point index where the monkey last sees a ff
	last_visible_index = ff_dataframe_visible[['point_index','target_index']].groupby('target_index').max()
	# Take out all the rows corresponding to these points
	last_visible_ffs = pd.merge(last_visible_index, ff_dataframe_visible, how="left")
	# Select trials where the target disappears the latest
	disappear_latest_trials = np.array(last_visible_ffs[last_visible_ffs['target_index']==last_visible_ffs['ff_index']]['target_index'])
	return disappear_latest_trials


def clusters_of_ffs_func(point_vs_cluster, monkey_information, ff_catched_T_sorted):
	temp_dataframe1 = pd.DataFrame(point_vs_cluster, columns=['point_index', 'ff_index', 'cluster_label'])
	u, c = np.unique(point_vs_cluster[:, 0], return_counts=True)
	temp_dataframe2 = pd.DataFrame(np.concatenate([u.reshape(-1, 1), c.reshape(-1, 1)], axis=1),
								   columns=['point_index', 'num_ff_at_point'])
	temp_dataframe3 = temp_dataframe1.merge(temp_dataframe2, how="left", on="point_index")
	corresponding_t = monkey_information['monkey_t'][np.array(temp_dataframe3['point_index'])]
	temp_dataframe3['time'] = corresponding_t
	temp_dataframe3['target_index'] = np.digitize(corresponding_t, ff_catched_T_sorted)
	temp_dataframe3 = temp_dataframe3[temp_dataframe3['target_index'] < len(ff_catched_T_sorted)]
	cluster_dataframe_point = temp_dataframe3
	cluster_dataframe_trial = cluster_dataframe_point[['target_index', 'num_ff_at_point']].groupby('target_index',
																								   as_index=True).agg(
		{'num_ff_at_point': ['max', 'count']})
	cluster_dataframe_trial.columns = ["max_ff_in_cluster", "num_points_w_cluster"]
	cluster_exist_trials = cluster_dataframe_point.target_index.unique()
	return cluster_exist_trials, cluster_dataframe_point, cluster_dataframe_trial



def ffs_around_target_func(ff_dataframe, catched_ff_num, ff_catched_T_sorted, ff_real_position_sorted, max_time_apart = 1.25):
	# See if the target is close to any ff that has been visible in the past 5s
	ffs_around_target = []
	ffs_around_target_indices = []
	ffs_around_target_positions = [] # Stores the positions of the ffs around the target
	temp_frame = ff_dataframe[['ff_index', 'target_index', 'ff_distance', 'visible', 'time']]
	for i in range(catched_ff_num):
	  time = ff_catched_T_sorted[i]
	  duration = [time-max_time_apart, time+max_time_apart]
	  target_nums = np.arange(i-1, i+2)
	  temp_frame2 = temp_frame[(temp_frame['time']>duration[0])&(temp_frame['time']<duration[1])]
	  #temp_frame2 = temp_frame[temp_frame['target_index'].isin(target_nums)]
	  temp_frame2 = temp_frame2[~temp_frame2['ff_index'].isin(target_nums)]
	  temp_frame2 = temp_frame2[(temp_frame2['visible'] ==1)]
	  temp_frame2 = temp_frame2[temp_frame2['ff_distance'] < 250]
	  past_visible_ff_indices = np.unique(np.array(temp_frame2.ff_index))
	  # Get positions of these ffs
	  past_visible_ff_positions = ff_real_position_sorted[past_visible_ff_indices]
	  # See if any one of it is within 50 cm of the target
	  distance2target = LA.norm(past_visible_ff_positions - ff_real_position_sorted[i], axis=1)
	  close_ff_indices = np.where(distance2target < 50)[0]
	  num_ff = len(close_ff_indices)
	  ffs_around_target.append(num_ff)
	  if len(close_ff_indices) > 0:
	    ffs_around_target_positions.append(past_visible_ff_positions[close_ff_indices])
	    ffs_around_target_indices.append(past_visible_ff_indices[close_ff_indices])
	  else:
	    ffs_around_target_positions.append(np.array([]))
	    ffs_around_target_indices.append(np.array([]))
	ffs_around_target = np.array(ffs_around_target)
	ffs_around_target_trials = np.where(ffs_around_target > 0)[0]
	return ffs_around_target_trials, ffs_around_target_positions




def sudden_flash_ignore_func(ff_dataframe, ff_real_position_sorted):
	df_ffdistance = np.array(ff_dataframe['ff_distance'])
	max_ff_index = max(np.array(ff_dataframe['ff_index']))

	# These are the indices of points where a ff changes from being invisible to become visible
	start_index1 = np.where(np.ediff1d(np.array(ff_dataframe['visible'])) == 1)[0]+1
	# These are the indices of points where the ff_index has changed
	start_index2 = np.where(np.ediff1d(np.array(ff_dataframe['ff_index']))!= 0)[0]+1
	# Combine the two
	start_index3 = np.concatenate((start_index1, start_index2))
	start_index = np.unique(start_index3)

	# Find the indices of ff_dataframe where the monkey encounters a closeby ff that suddenly becomes visible
	suddent_flash_index = start_index[np.where(df_ffdistance[start_index] < 50)]

	# The indices and trials of ff_dataframe where the suddenly visible ff is the target or next target
	#sudden_flash_capture = suddent_flash_index[np.where(np.array(ff_dataframe['ff_index'])[suddent_flash_index] == np.array(ff_dataframe['target_index'])[suddent_flash_index])[0]]
	condition = np.logical_or((np.array(ff_dataframe['ff_index'])[suddent_flash_index] == np.array(ff_dataframe['target_index'])[suddent_flash_index]), (np.array(ff_dataframe['ff_index'])[suddent_flash_index] == np.array(ff_dataframe['target_index']+1)[suddent_flash_index]))
	sudden_flash_capture = suddent_flash_index[condition]
	sudden_flash_capture_trials = np.array(ff_dataframe['target_index'])[sudden_flash_capture]
	sudden_flash_capture_trials = np.unique(sudden_flash_capture_trials)

	# The indices and trials of ff_dataframe where the suddenly visible ff is not the target
	#sudden_flash_ignore = suddent_flash_index[np.where(np.array(ff_dataframe['ff_index'])[suddent_flash_index] != np.array(ff_dataframe['target_index'])[suddent_flash_index])[0]]
	sudden_flash_ignore = suddent_flash_index[~condition]

	# Find the distance to targets at these points
	cum_x = np.array(ff_dataframe.monkey_x)[sudden_flash_ignore]
	cum_y = np.array(ff_dataframe.monkey_y)[sudden_flash_ignore]
	cum_target_indices = np.array(ff_dataframe.target_index)[sudden_flash_ignore]
	cum_target_distances = LA.norm(np.stack([cum_x, cum_y], axis=1)-ff_real_position_sorted[cum_target_indices], axis=1)
	cum_ff_distances = df_ffdistance[sudden_flash_ignore]
	valid_indices = np.where(cum_target_distances > cum_ff_distances)
	sudden_flash_ignore_trials_non_unique = cum_target_indices[valid_indices]
	sudden_flash_ignore_trials = np.unique(sudden_flash_ignore_trials_non_unique)

	## By points
	# Find the points corresponding to such a sudden flash
	sudden_flash_ignore_indices = np.array(ff_dataframe['point_index'])[sudden_flash_ignore[valid_indices]]
	sudden_flash_ignore_ff = np.array(ff_dataframe['ff_index'])[sudden_flash_ignore[valid_indices]]
	# Append each point into a list and the following n points so that the message can be visible for 2 seconds
	sudden_flash_ignore_points = []
	for i in sudden_flash_ignore_indices:
	  sudden_flash_ignore_points = sudden_flash_ignore_points+list(range(i, i+121))
	return sudden_flash_ignore_trials, sudden_flash_ignore_trials_non_unique, sudden_flash_ignore_indices, sudden_flash_ignore_points


def try_a_few_times_func(catched_ff_num, ff_catched_T_sorted, monkey_information, ff_believed_position_sorted, player, max_point_index):
	# Show the trials where the last cluster has more than 2 stops
	try_a_few_times_trials = []
	try_a_few_times_indices = []
	for i in range(catched_ff_num): 
	  # Find clusters based on a distance of 50
	  clusters = find_clusters(i, 50, ff_catched_T_sorted, monkey_information, player)
	  # if clusters is not empty:
	  if len(clusters) > 0:
	    num_for_last_cluster = clusters[-1]
	    # If the last cluster has more than 2 stops
	    if clusters.count(num_for_last_cluster) > 1:
	      distinct_stops = num_of_stops(i, ff_catched_T_sorted, monkey_information, player = "monkey")
	      distinct_stops_indices = num_of_stops_indices(i, ff_catched_T_sorted, monkey_information, player = "monkey")
	      # If the last stop is close enough to the believed position of the target
	      if LA.norm(distinct_stops[-1]-ff_believed_position_sorted[i]) < 50:
	        try_a_few_times_trials.append(i)
			# By points 
	        min_index = distinct_stops_indices[-clusters.count(num_for_last_cluster)]
	        max_index = distinct_stops_indices[-1]
	        try_a_few_times_indices = try_a_few_times_indices + \
	        list(range(min_index-20, min(max_index+20, max_point_index)))
	return try_a_few_times_trials, try_a_few_times_indices


def give_up_after_trying_func(catched_ff_num, ff_catched_T_sorted, monkey_information, ff_believed_position_sorted, PLAYER):
	# By trials
	# Show the trials where there at least one cluster has more than 2 stops, and this cluster is neither at the beginning nor at the end
	give_up_after_trying_trials = []
	for i in range(catched_ff_num):
	  # Find clusters based on a distance of 50
	  clusters = find_clusters(i, 50, ff_catched_T_sorted, monkey_information, player = PLAYER)
	  # if clusters is not empty:
	  if len(clusters) > 0:
	    clusters_counts = collections.Counter(clusters)
	    distinct_stop_positions = num_of_stops(i, ff_catched_T_sorted, monkey_information, player = "monkey")
	    for k in range(1, clusters[-1]+1):  # for each cluster
	      # if it has more than one element:
	      if clusters_counts[k] > 1:
	        # Get positions of these points
	        stop_positions = [distinct_stop_positions[index] for index, value in enumerate(clusters) if value == k]
	        # If the first stop is not close to beginning, and the last stop is not too close to the end:
	        if LA.norm(stop_positions[0]-ff_believed_position_sorted[i-1]) > 50 and LA.norm(stop_positions[-1]-ff_believed_position_sorted[i]) > 50:
	          give_up_after_trying_trials.append(i)
	# By points
	give_up_after_trying_indices = []
	for i in range(catched_ff_num):
	  # Find clusters based on a distance of 50
	  clusters = find_clusters(i, 50, ff_catched_T_sorted, monkey_information, player = PLAYER)
	  # if clusters is not empty:
	  if len(clusters) > 0:
	    clusters_counts = collections.Counter(clusters)
	    distinct_stop_positions = num_of_stops(i, ff_catched_T_sorted, monkey_information, player = "monkey")
	    distinct_stops_indices = num_of_stops_indices(i, ff_catched_T_sorted, monkey_information, player = "monkey")
	    for k in range(1, clusters[-1]+1):  # for each cluster
	      # if it has more than one element:
	      if clusters_counts[k] > 1:
	        # Get positions of these points
	        stop_positions = [distinct_stop_positions[index] for index, value in enumerate(clusters) if value == k]
	        # If the first stop is not close to beginning, and the last stop is not too close to the end:
	        if LA.norm(stop_positions[0]-ff_believed_position_sorted[i-1]) > 50 and LA.norm(stop_positions[-1]-ff_believed_position_sorted[i]) > 50:
	          # Get indices of these points
	          stop_positions_indices = [distinct_stops_indices[index] for index, value in enumerate(clusters) if value == k]
	          give_up_after_trying_indices = give_up_after_trying_indices + list(range(min(stop_positions_indices)-20, max(stop_positions_indices)+21))
	return give_up_after_trying_trials, give_up_after_trying_indices



def trials_char_func(PLAYER, ff_dataframe, monkey_information, catched_ff_num, ff_catched_T_sorted, ff_believed_position_sorted, n_ff_in_a_row, data_folder_name = None):
	relevant_df0 = ff_dataframe[ff_dataframe['visible'] == 1]
	last_trial = catched_ff_num-1
	trial_array = [i for i in range(1, last_trial+1)]
	# Trial number is named after the index of the target. 
	# Trial number starts at 1

	t_array = ff_catched_T_sorted[1:last_trial+1] - ff_catched_T_sorted[:last_trial]
	t_array = t_array.tolist()

	## How long can the monkey remember a target?
	## time elapses between the target last being visible and its capture
	t_last_visible = []
	d_last_visible = []
	abs_angle_last_visible = []
	for i in range(1, last_trial+1):
	  relevant_df = relevant_df0[((relevant_df0['target_index']==i) & (relevant_df0['ffdistance2target']<25))| (relevant_df0['ff_index']==i)]
	  if len(relevant_df) > 0:
	    t_last_visible.append(ff_catched_T_sorted[i] - max(np.array(relevant_df.time)))
	    d_last_visible.append(max(np.array(relevant_df.ff_distance)))
	    abs_angle_last_visible.append(max(np.absolute(np.array(relevant_df.ff_angle_boundary))))
	  else:
	    t_last_visible.append(9999)
	    d_last_visible.append(9999)
	    abs_angle_last_visible.append(9999)


	hit_boundary = []
	for i in range(1, last_trial+1):
	  duration = [ff_catched_T_sorted[i-1], ff_catched_T_sorted[i]]
	  cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
	  if len(cum_indices) > 1:
	    cum_t, cum_angle = monkey_information['monkey_t'][cum_indices],  monkey_information['monkey_angle'][cum_indices]
	    cum_mx, cum_my = monkey_information['monkey_x'][cum_indices], monkey_information['monkey_y'][cum_indices] 
	    if np.any(cum_mx[1:]-cum_mx[:-1] > 55) or np.any(cum_my[1:]-cum_my[:-1] > 55):
	      hit_boundary.append(False)
	    else:
	      hit_boundary.append(False)
	  else:
	    hit_boundary.append(False)

	# num_stops
	num_stops_array = [len(num_of_stops(i, ff_catched_T_sorted, monkey_information, player = "monkey")) for i in range(1, last_trial+1)]

	num_stops_since_last_seen = [len(num_of_stops_since_target_last_seen(i, ff_catched_T_sorted, monkey_information, t_last_visible)) for i in range(1, last_trial+1)]


	num_stops_near_target = []
	for i in range(1, last_trial+1):
	  clusters = find_clusters(i, 50, ff_catched_T_sorted, monkey_information, player = PLAYER)
	  # for each trial, append the trial number and the number of stop in the last cluster, if the cluster is close enough to the target
	  distinct_stops = num_of_stops(i, ff_catched_T_sorted, monkey_information, player = PLAYER)
	  if len(distinct_stops) > 0:
	    # If the last stop is close enough to the believed position of the target
	    if LA.norm(distinct_stops[-1]-ff_believed_position_sorted[i]) < 50:
	      num_stops_near_target.append(clusters.count(clusters[-1]))
	    else: 
	      num_stops_near_target.append(0)
	  else: 
	    num_stops_near_target.append(0)

	# trials_char
	trials_dict = {'trial':trial_array, 
	               't':t_array,  
	               't_last_visible':t_last_visible,
	               'd_last_visible':d_last_visible,
	               'abs_angle_last_visible': abs_angle_last_visible,
	               'hit_boundary': hit_boundary,  
	               'num_stops': num_stops_array, 
	               'num_stops_near_target': num_stops_near_target                            
	                      }
	trials_char = pd.DataFrame(trials_dict)
	trials_char['n_ff_in_a_row'] = n_ff_in_a_row[:len(trials_char)]
	if data_folder_name:
		filepath = data_folder_name + '/trials_char.csv'
		os.makedirs(data_folder_name, exist_ok = True)
		trials_char.to_csv(filepath)
	print("new trials_char is made.")
	return trials_char



def compare_agents_monkey_patterns(stats_dict, stats_dict_m, agent_names = ["Agent", "Agent2", "Agent3"], stats_dict2 = None, stats_dict3 = None):
	#filename = f"noise = {noise}"
	agent_stats_df = pd.DataFrame.from_dict(stats_dict, orient='index', columns=['Value'])
	agent_stats_df['Player'] = agent_names[0]
	monkey_stats_df = pd.DataFrame.from_dict(stats_dict_m, orient='index', columns=['Value'])
	monkey_stats_df['Player'] = 'Monkey'
	if stats_dict3:
		agent_stats_df2 = pd.DataFrame.from_dict(stats_dict2, orient='index', columns=['Value'])
		agent_stats_df2['Player'] = agent_names[1]
		agent_stats_df3 = pd.DataFrame.from_dict(stats_dict3, orient='index', columns=['Value'])
		agent_stats_df3['Player'] = agent_names[2]
		merged_stats_df = pd.concat([agent_stats_df, monkey_stats_df, agent_stats_df2, agent_stats_df3], axis=0)
	elif stats_dict2:
		agent_stats_df2 = pd.DataFrame.from_dict(stats_dict2, orient='index', columns=['Value'])
		agent_stats_df2['Player'] = agent_names[1]
		merged_stats_df = pd.concat([agent_stats_df, monkey_stats_df, agent_stats_df2], axis=0)
	else:
		merged_stats_df = pd.concat([agent_stats_df, monkey_stats_df], axis=0)
	merged_stats_df = merged_stats_df.reset_index()
	merged_stats_df.rename(columns = {'index':'Category'}, inplace = True)
	return merged_stats_df


def compare_agents_monkey_medians(stats_dict_median, stats_dict_median_m, agent_names = ["Agent", "Agent2", "Agent3"], stats_dict_median2 = None, stats_dict_median3 = None):
	#filename = f"noise = {noise}"
	agent_stats_df = pd.DataFrame.from_dict(stats_dict_median, orient='index', columns=['Value'])
	agent_stats_df['Player'] = agent_names[0]
	monkey_stats_df = pd.DataFrame.from_dict(stats_dict_median_m, orient='index', columns=['Value'])
	monkey_stats_df['Player'] = 'Monkey'

	if stats_dict_median3:
		agent_stats_df2 = pd.DataFrame.from_dict(stats_dict_median2, orient='index', columns=['Value'])
		agent_stats_df2['Player'] = agent_names[1]
		agent_stats_df3 = pd.DataFrame.from_dict(stats_dict_median3, orient='index', columns=['Value'])
		agent_stats_df3['Player'] = agent_names[2]
		merged_stats_df = pd.concat([agent_stats_df, monkey_stats_df, agent_stats_df2, agent_stats_df3], axis=0)
	elif stats_dict_median2:
		agent_stats_df2 = pd.DataFrame.from_dict(stats_dict_median2, orient='index', columns=['Value'])
		agent_stats_df2['Player'] = agent_names[1]
		merged_stats_df = pd.concat([agent_stats_df, monkey_stats_df, agent_stats_df2], axis=0)
	else:
		merged_stats_df = pd.concat([agent_stats_df, monkey_stats_df], axis=0)
	merged_stats_df = merged_stats_df.reset_index()

	merged_stats_df = pd.concat([agent_stats_df, monkey_stats_df], axis=0)
	merged_stats_df = merged_stats_df.reset_index()
	merged_stats_df.rename(columns = {'index':'Category'}, inplace = True)
	return merged_stats_df


def plot_merged_stats(merged_stats_df):
	sns.set(style="darkgrid")
	# Set the figure size
	plt.figure(figsize=(8, 8))
	# grouped barplot
	ax = sns.barplot(x="Category", y="Value", hue="Player", data=merged_stats_df, ci=None);
	ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
	plt.tight_layout()
	plt.show()
	plt.close()


def graphs_by_category_all(merged_stats_df):
	for cate in merged_stats_df.Category.unique():
		category_df = merged_stats_df[merged_stats_df['Category']==cate]
		# set plot style: grey grid in the background:
		sns.set(style="darkgrid")
		# Set the figure size
		plt.figure(figsize=(4, 8))
		# grouped barplot
		##ax = sns.barplot(x="Player", y="Value", hue="Player", data=category_df, ci=None);
		ax = sns.barplot(x="Player", y="Value", data=category_df, ci=None);
		##ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
		#ax.set_ylabel("Percentage of captured fireflies", fontsize = 22)
		ax.set_ylabel("")
		plt.title(str(cate), fontsize= 22)
		plt.xticks(fontsize= 22)
		plt.yticks(fontsize= 15) 
		##ax.set_xticklabels("")
		ax.set_xlabel("")
		ax.yaxis.set_major_formatter(mtick.PercentFormatter())
		plt.tight_layout()
		plt.show()
		plt.close()

def histogram_per_attribute_all(valid_trials, valid_trials_m):
	variable_of_interest = "t_last_visible"
	fig, axes = plt.subplots(figsize=(4, 5))
	sns.histplot(data = valid_trials[variable_of_interest], kde = False, alpha = 0.4, color = "green", binwidth = 0.25)
	sns.histplot(data = valid_trials_m[variable_of_interest], kde = False, alpha = 0.4, color = "blue", binwidth = 0.22)
	axes.legend(labels=["Agent(LSTM)","Monkey"], fontsize = 14)
	#axes.set_ylabel("")
	#axes.set_yticklabels("")
	plt.xticks(fontsize = 18)
	plt.title("Time Since Target Last Visible", fontsize=18)
	plt.xlabel("Time (s)", fontsize = 18)
	plt.xlim([0, 6])
	#axes.xaxis.set_major_locator(mtick.NullLocator())
	axes.yaxis.set_major_locator(mtick.NullLocator())
	plt.show()
	plt.close()



	variable_of_interest = "num_stops_near_target"
	fig, axes = plt.subplots(figsize=(4, 5))
	sns.histplot(data = valid_trials[variable_of_interest], kde = False, alpha = 0.3, color = "green", binwidth=0.5, binrange=(-0.25,5.25), stat="probability", edgecolor='grey')
	sns.histplot(data = valid_trials_m[variable_of_interest], kde = False, alpha = 0.3, color = "blue", binwidth=0.5, binrange=(-0.1,5.4), stat="probability", edgecolor='grey')
	axes.legend(labels=["Agent(LSTM)","Monkey"], fontsize = 14)
	# sns.kdeplot(data = valid_trials, x = variable_of_interest, bw=1, color = "green")
	# sns.kdeplot(data = valid_trials_m, x = variable_of_interest, bw=1, color = "blue")
	axes.set_ylabel("Probability",fontsize=15)
	#axes.set_yticklabels("")
	plt.xlim(-0.25,5)
	plt.title("Number of Stops Near Targets", fontsize=17)
	plt.xlabel("Number of Stops", fontsize=15)
	axes.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.show()
	plt.close()



	variable_of_interest = "n_ff_in_a_row"
	fig, axes = plt.subplots(figsize=(4, 5))
	sns.histplot(data = valid_trials[variable_of_interest], kde = False, alpha = 0.3, binrange=(-0.25,5.25), color = "green", binwidth=0.5, stat="probability",  edgecolor='grey')
	sns.histplot(data = valid_trials_m[variable_of_interest], kde = False, alpha = 0.3, binrange=(-0.1,5.4), color = "blue", binwidth=0.5, stat="probability",  edgecolor='grey')
	axes.legend(labels=["Agent(LSTM)","Monkey"], fontsize = 14)
	# sns.kdeplot(data = valid_trials, x = variable_of_interest, bw=1, color = "green")
	# sns.kdeplot(data = valid_trials_m, x = variable_of_interest, bw=1, color = "blue")
	axes.set_ylabel("Probability",fontsize=15)
	#axes.set_yticklabels("")
	plt.xlim(0.25,5.25)
	plt.title("Number of fireflies caught in a cluster", fontsize=17)
	plt.xlabel("Number of Fireflies", fontsize=15)
	axes.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.show()
	plt.close()



	variable_of_interest = "d_last_visible"
	fig, axes = plt.subplots(figsize=(4, 5))
	sns.histplot(data = valid_trials[variable_of_interest]/100, kde = False, alpha = 0.3,  color = "green", binwidth=40, stat="probability",  edgecolor='grey')
	sns.histplot(data = valid_trials_m[variable_of_interest]/100, kde = False, alpha = 0.3,  color = "blue", binwidth=30, stat="probability",  edgecolor='grey')
	axes.legend(labels=["Agent(LSTM)","Monkey"], fontsize = 14)
	# sns.kdeplot(data = valid_trials, x = variable_of_interest, color = "green", bw=1)
	# sns.kdeplot(data = valid_trials_m, x = variable_of_interest, color = "blue", bw=1)
	axes.set_ylabel("Probability",fontsize=15)
	#axes.set_yticklabels("")
	plt.xlim(0, 400)
	plt.title("Distance of Target Last Visible", fontsize=17)
	plt.xlabel("Distance (100 cm)", fontsize=15)
	axes.xaxis.set_major_locator(MaxNLocator(integer=True))
	axes.tick_params(axis = "x", width=0)
	xticklabels=axes.get_xticks().tolist()
	xticklabels = [str(int(label)) for label in xticklabels]
	xticklabels[-1]='400+'
	axes.set_xticklabels(xticklabels)
	plt.show()
	plt.close()



	variable_of_interest = "abs_angle_last_visible"
	fig, axes = plt.subplots(figsize=(4, 5))
	sns.histplot(data = valid_trials[variable_of_interest], kde = False, binwidth=0.04, alpha = 0.3, color = "green", stat="probability", edgecolor='grey')
	sns.histplot(data = valid_trials_m[variable_of_interest], kde = False,  binwidth=0.05, alpha = 0.3, color = "blue", stat="probability", edgecolor='grey')
	axes.legend(labels=["Agent(LSTM)","Monkey"], fontsize = 14)
	# sns.kdeplot(data = valid_trials, x = variable_of_interest, color = "green", bw=1)
	# sns.kdeplot(data = valid_trials_m, x = variable_of_interest, color = "blue", bw=1)
	axes.set_ylabel("Probability",fontsize=15)
	#axes.set_yticklabels("")
	plt.title("Abs Angle of Target Last Visible", fontsize=17)
	plt.xlabel("Angle (rad)", fontsize=15)
	axes.xaxis.set_major_locator(MaxNLocator(integer=True))
	axes.tick_params(axis = "x", width=0)
	axes.set_xticks(np.arange(0.0, 0.9, 0.2))
	axes.set_xticklabels(np.arange(0.0, 0.9, 0.2).round(1))
	plt.xlim(0, 0.7)
	plt.show()
	plt.close()


	variable_of_interest = "t"
	fig, axes = plt.subplots(figsize=(4, 5))
	sns.histplot(data = valid_trials[variable_of_interest], kde = False, binwidth=1,  alpha = 0.3, color = "green", stat="probability", edgecolor='grey')
	sns.histplot(data = valid_trials_m[variable_of_interest], kde = False, binwidth=1.1,  alpha = 0.3, color = "blue", stat="probability", edgecolor='grey')
	axes.legend(labels=["Agent(LSTM)","Monkey"], fontsize = 14)
	# sns.kdeplot(data = valid_trials, x = variable_of_interest, color = "green", bw=1)
	# sns.kdeplot(data = valid_trials_m, x = variable_of_interest, color = "blue", bw=1)
	axes.set_ylabel("Probability",fontsize=15)
	#axes.set_yticklabels("")
	plt.title("Trial Duration", fontsize=17)
	plt.xlabel("Duration (s)", fontsize=15)
	axes.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.show()
	plt.close()



	variable_of_interest = "num_stops"
	fig, axes = plt.subplots(figsize=(4, 5))
	sns.histplot(data = valid_trials[variable_of_interest], binwidth=1, binrange=(0.5, 10.5), alpha = 0.3, color = "green", stat="probability", edgecolor='grey')
	sns.histplot(data = valid_trials_m[variable_of_interest], binwidth=1, binrange=(0.6, 10.6), alpha = 0.3, color = "blue", stat="probability", edgecolor='grey')
	axes.legend(labels=["Agent(LSTM)","Monkey"], fontsize = 14)
	# sns.kdeplot(data = valid_trials, x = variable_of_interest,  bw=2, color = "green")
	# sns.kdeplot(data = valid_trials_m, x = variable_of_interest, bw=2, color = "blue")
	axes.set_ylabel("Probability",fontsize=15)
	plt.xlim(0.7,12)
	#axes.set_yticklabels("")
	plt.title("Number of Stops During Trials", fontsize=17)
	plt.xlabel("Number of Stops", fontsize=15)
	axes.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.show()
	plt.close()



def make_point_vs_cluster(data_folder_name, monkey_information, ff_dataframe, min_point_index, max_point_index, max_cluster_distance = 100, max_ff_distance_from_monkey = 250, max_time_past = 1):
	max_points_past = math.floor(
		max_time_past / (monkey_information['monkey_t'][100] - monkey_information['monkey_t'][99]))
	min_memory_of_ff = 100 - max_points_past
	point_vs_cluster = []
	# new structure: [[point_index, ff_index, cluster_label], [point_index, ff_index, cluster_label], ...]
	# for i in range(7893, 7893+10):  #for testing purpose
	for i in range(min_point_index, max_point_index + 1):
		selected_ff = ff_dataframe[
			(ff_dataframe['point_index'] == i) & (ff_dataframe['memory'] > (min_memory_of_ff)) & (
					ff_dataframe['ff_distance'] < max_ff_distance_from_monkey)][['ff_x', 'ff_y', 'ff_index']]
		ffxy_array = selected_ff[['ff_x', 'ff_y']].to_numpy()
		if len(ffxy_array) > 1:
			ff_indices = selected_ff[['ff_index']].to_numpy()
			linked = linkage(ffxy_array, method='single')
			num_clusters = sum(linked[:, 2] > max_cluster_distance) + 1  # This is a formula I developed
			if num_clusters < len(ff_indices):
				cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='single')
				labels = cluster.fit_predict(ffxy_array)
				u, c = np.unique(labels, return_counts=True)
				dup = u[c > 1]
				# cluster_info = np.concatenate([np.repeat([i], len(ff_indices)).reshape(-1,1), ff_indices, labels.reshape([-1,1])], axis=1)
				for index in np.isin(labels, dup).nonzero()[0]:
					point_vs_cluster.append([i, ff_indices[index].item(), labels[index]])
		if i % 1000 == 0:
			print(i, " out of ", max_point_index, " for point_vs_cluster")
	point_vs_cluster = np.array(point_vs_cluster)
	filepath = data_folder_name + '/point_vs_cluster.csv'
	os.makedirs(data_folder_name, exist_ok=True)
	np.savetxt(filepath, point_vs_cluster, delimiter=',')
	return point_vs_cluster