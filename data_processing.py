from config import *
from functions.process_raw_data import*
from functions.find_patterns import*
from data.manufactured_data.pattern_data import stats_dict2, stats_dict_median2, stats_dict3, stats_dict_median3, stats_dict_m, stats_dict_median_m



import torch
import numpy as np
import pandas as pd
import os
import matplotlib
from os.path import exists
from matplotlib import rc
os.environ['KMP_DUPLICATE_LIB_OK']='True'


matplotlib.rcParams.update(matplotlib.rcParamsDefault)
rc('animation', html='jshtml')
matplotlib.rcParams['animation.embed_limit'] = 2**128


torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)







Channel_signal_output,marker_list,smr_sampling_rate = smr_extractor(data_folder_name = raw_data_folder_name).extract_data()
#Considering the first smr file, use marker_list[0], Channel_signal_output[0]
juice_timestamp = marker_list[0]['values'][marker_list[0]['labels']==4]
Channel_signal_smr1 = Channel_signal_output[0]
Channel_signal_smr1['section'] = np.digitize(Channel_signal_smr1.Time,juice_timestamp) # seperate analog signal by juice timestamps
# Remove tail of analog data
Channel_signal_smr1 = Channel_signal_smr1[Channel_signal_smr1['section']<Channel_signal_smr1['section'].unique()[-1]]
Channel_signal_smr1.loc[Channel_signal_smr1.index[-1], 'Time'] = juice_timestamp[-1]
# Remove head of analog data
Channel_signal_smr1 = Channel_signal_smr1[Channel_signal_smr1['Time']>marker_list[0]['values'][marker_list[0]['labels']==1][0]]
# monkey_smr_dataframe = Channel_signal_smr1[["Time", "Signal stream 1", "Signal stream 2", "Signal stream 3", "Signal stream 10"]].reset_index(drop=True)
# monkey_smr_dataframe.columns = ['monkey_t', 'monkey_x', 'monkey_y', 'monkey_speed', 'AngularV']
# monkey_smr = dict(zip(monkey_smr_dataframe.columns.tolist(), np.array(monkey_smr_dataframe.values.T.tolist())))






# Data from monkey
#sort ff by catched time
ff_information,monkey_information = log_extractor(data_folder_name = raw_data_folder_name, file_name = "m51s936.txt").extract_data()
ff_index = []
ff_catched_T = []
ff_real_position = []
ff_believed_position = []
ff_life = []
ff_flash = []
ff_flash_end = []  # This is the time that the firefly last stops flash
for item in ff_information:
    item['Life'] = np.array([item['ff_flash_T'][0][0],item['ff_catched_T']])
    ff_index = np.hstack((ff_index,item['ff_index']))
    ff_catched_T = np.hstack((ff_catched_T,item['ff_catched_T']))
    ff_real_position.append(item['ff_real_position'])
    ff_believed_position.append(item['ff_believed_position'])
    ff_life.append(item['Life'])
    ff_flash.append(item['ff_flash_T'])
    ff_flash_end.append(item['ff_flash_T'][-1][-1])
sort_index = np.argsort(ff_catched_T)
ff_index_sorted = ff_index[sort_index]
ff_catched_T_sorted = ff_catched_T[sort_index]
ff_real_position_sorted = np.array(ff_real_position)[sort_index]
ff_believed_position_sorted = np.array(ff_believed_position)[sort_index]
ff_life_sorted = np.array(ff_life)[sort_index]
ff_flash_sorted  = np.array(ff_flash, dtype=object)[sort_index].tolist()
ff_flash_end_sorted = np.array(ff_flash_end)[sort_index]

# Use accurate juice timestamps, ff_catched_T_sorted for smr1 (so that the time frame is correct)
ff_catched_T_sorted = ff_catched_T_sorted[:np.where(ff_catched_T_sorted<=Channel_signal_smr1.Time.values[-1])[0][-1]+1]

catched_ff_num = len(ff_catched_T_sorted) - 200
total_ff_num = len(ff_life_sorted)
M_catched_ff_num = len(ff_catched_T_sorted) - 200
M_total_ff_num = len(ff_life_sorted)
M_ff_catched_T_sorted = ff_catched_T_sorted.copy()
M_ff_real_position_sorted = ff_real_position_sorted.copy()
M_ff_life_sorted = ff_life_sorted.copy()
M_ff_flash_sorted = ff_flash_sorted.copy()
M_ff_believed_position_sorted = ff_believed_position_sorted.copy()



monkey_information = process_monkey_information(monkey_information)
M_monkey_information = monkey_information.copy()


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



filepath = data_folder_name + '/trials_char.csv'
if exists(filepath):
    trials_char = pd.read_csv(filepath)
else:
    trials_char = trials_char_func(PLAYER, ff_dataframe, monkey_information, catched_ff_num, ff_catched_T_sorted, ff_believed_position_sorted, n_ff_in_a_row, data_folder_name = data_folder_name)
valid_trials = trials_char[(trials_char['t_last_visible']<50) & (trials_char['hit_boundary']==False)].reset_index()
median_values = valid_trials.median(axis=0)


# stats_dict
max_time = min(ff_catched_T_sorted[-1]-100, 2000)
n_trial = np.where(ff_catched_T_sorted < max_time)[0][-1]
catched_T_bounded = ff_catched_T_sorted[np.where(ff_catched_T_sorted < max_time)[0]]
num_stops_array = [len(num_of_stops(i, ff_catched_T_sorted, monkey_information, player = "monkey")) for i in range(1, n_trial+1)]
stats_dict = {
"Two in a row" : len(np.where(n_ff_in_a_row[:n_trial]>=2)[0])/(n_trial-2),
"Visible before last capture" : len(np.where(visible_before_last_one_trials < n_trial)[0])/(n_trial-2),
"Target disappears latest" : len(np.where(disappear_latest_trials < n_trial)[0])/(n_trial-1),
"Waste cluster around last target": len(np.where(waste_cluster_last_target_trials < n_trial)[0])/(n_trial-2),
"Ignore sudden flash": len(np.where(sudden_flash_ignore_trials < n_trial)[0])/(n_trial-1),
"Try a few times": len(np.where(np.array(try_a_few_times_trials) < n_trial)[0])/(n_trial-1),
"Give up after trying": len(np.where(np.array(give_up_after_trying_trials) < n_trial)[0])/(n_trial-1),
"ff capture rate": (len(catched_T_bounded)-1)/(catched_T_bounded[-1]-catched_T_bounded[0]),
"Stop success rate": n_trial/sum(num_stops_array),
}


stats_dict_median = {"Median time": median_values['t'],
"Median time target last seen": median_values['t_last_visible'],
"Median distance target last seen": median_values['d_last_visible'],
"Median abs angle target last seen ": median_values['abs_angle_last_visible'],
"Median num stops": median_values['num_stops'],
"Median num stops near target": median_values['num_stops_near_target'],
}


merged_stats_categories = compare_agents_monkey_patterns(stats_dict, stats_dict_m, agent_names = ["Agent", "Agent2", "Agent3"], stats_dict2 = None, stats_dict3 = None)
merged_stats_medians = compare_agents_monkey_medians(stats_dict_median, stats_dict_median_m, agent_names = ["Agent", "Agent2", "Agent3"], stats_dict_median2 = None, stats_dict_median3 = None)

#plot_merged_stats(merged_stats_categories)
#plot_merged_stats(merged_stats_medians)

#graphs_by_category_all(merged_stats_categories)
#graphs_by_category_all(merged_stats_medians)



trials_char_m = pd.read_csv('data/manufactured_data/monkey_info0219/trials_char.csv')
valid_trials_m = trials_char_m[(trials_char_m['t_last_visible']<50) & (trials_char_m['hit_boundary']==False)].reset_index()
median_values_m = valid_trials_m.median(axis=0)

trials_char_lstm = pd.read_csv('RL/LSTM/LSTM_data/LSTM_July_26_2/trials_char.csv')
valid_trials_lstm = trials_char_lstm[(trials_char_lstm['t_last_visible']<50) & (trials_char_lstm['hit_boundary']==False)].reset_index()
median_values_lstm = valid_trials_lstm.median(axis=0)



trials_char_m = pd.read_csv(data_folder_name + '/trials_char.csv')
valid_trials_m = trials_char_m[(trials_char_m['t_last_visible']<50) & (trials_char_m['hit_boundary']==False)].reset_index()
median_values_m = valid_trials_m.median(axis=0)
#histogram_per_attribute_all(valid_trials, valid_trials_m)
