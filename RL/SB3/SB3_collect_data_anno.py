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





monkey_information, ff_flash_sorted, ff_catched_T_sorted, ff_believed_position_sorted, ff_real_position_sorted, \
    ff_life_sorted, ff_flash_end_sorted, catched_ff_num, total_ff_num = 
data_from_SB3(env, sac_model, retrieve_dir, n_steps = 1000, retrieve_buffer = False)


ff_dataframe


n_ff_in_a_row = n_ff_in_a_row_func(catched_ff_num, ff_believed_position_sorted, distance_between_ff = 50)

visible_before_last_one_trials = visible_before_last_one_func(ff_dataframe)

disappear_latest_trials = disappear_latest_func(ff_dataframe)

sudden_flash_ignore_trials, sudden_flash_ignore_trials_non_unique, sudden_flash_ignore_indices, sudden_flash_ignore_points = sudden_flash_ignore_func(ff_dataframe, ff_real_position_sorted)

try_a_few_times_trials, try_a_few_times_indices = try_a_few_times_func(catched_ff_num, ff_catched_T_sorted, monkey_information, ff_believed_position_sorted, PLAYER, max_point_index)

give_up_after_trying_trials, give_up_after_trying_indices = give_up_after_trying_func(catched_ff_num, ff_catched_T_sorted, monkey_information, ff_believed_position_sorted, PLAYER)



