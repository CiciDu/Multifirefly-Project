from functions.find_patterns import *
import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'




def make_anim_monkey_info(monkey_information, cum_index, k = 5):
    cum_t, cum_angle = monkey_information['monkey_t'][cum_index], monkey_information['monkey_angle'][cum_index]
    cum_mx, cum_my = monkey_information['monkey_x'][cum_index], monkey_information['monkey_y'][cum_index]
    anim_index = cum_index[0][0:-1:k]
    anim_t = cum_t[0:-1:k]
    anim_mx = cum_mx[0:-1:k]
    anim_my = cum_my[0:-1:k]
    anim_angle = cum_angle[0:-1:k]
    xmin, xmax = np.min(cum_mx), np.max(cum_mx)
    ymin, ymax = np.min(cum_my), np.max(cum_my)
    anim_monkey_info = {"anim_index": anim_index, "anim_t": anim_t, "anim_angle": anim_angle, "anim_mx": anim_mx, "anim_my": anim_my,
                        "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    return anim_monkey_info



def make_annotation_info(catched_ff_num, max_point_index, n_ff_in_a_row, visible_before_last_one_trials, disappear_latest_trials, \
                         ignore_sudden_flash_indices, give_up_after_trying_indices, try_a_few_times_indices):
    zero_array = np.zeros(catched_ff_num, dtype=int)

    visible_before_last_one_trial_dummy = zero_array.copy()
    if len(visible_before_last_one_trials) > 0:
        visible_before_last_one_trial_dummy[visible_before_last_one_trials] = 1

    disappear_latest_trial_dummy = zero_array.copy()
    if len(disappear_latest_trials) > 0:
        disappear_latest_trial_dummy[disappear_latest_trials] = 1 

    ignore_sudden_flash_point_dummy = np.zeros(max_point_index+1, dtype=int)
    if len(ignore_sudden_flash_indices) > 0:
        ignore_sudden_flash_point_dummy[ignore_sudden_flash_indices] = 1

    give_up_after_trying_point_dummy = np.zeros(max_point_index+1, dtype=int)
    if len(give_up_after_trying_indices) > 0:
        give_up_after_trying_point_dummy[give_up_after_trying_indices] = 1

    try_a_few_times_point_dummy = np.zeros(max_point_index+1, dtype=int)
    if len(try_a_few_times_indices) > 0:
        try_a_few_times_point_dummy[try_a_few_times_indices] = 1

    annotation_info = {"n_ff_in_a_row": n_ff_in_a_row, "visible_before_last_one_trial_dummy": visible_before_last_one_trial_dummy, "disappear_latest_trial_dummy": disappear_latest_trial_dummy, 
                       "ignore_sudden_flash_point_dummy": ignore_sudden_flash_point_dummy, "try_a_few_times_point_dummy": try_a_few_times_point_dummy, "give_up_after_trying_point_dummy": give_up_after_trying_point_dummy}
    return annotation_info






def animate(frame, ax, anim_monkey_info, margin, ff_dataframe, ff_real_position_sorted, ff_position_during_this_trial, \
            flash_on_ff_dict, believed_ff_dict): 
    ax.cla()
    ax.axis('off')
    ax.set_xlim((anim_monkey_info['xmin']-margin, anim_monkey_info['xmax']+margin))
    ax.set_ylim((anim_monkey_info['ymin']-margin, anim_monkey_info['ymax']+margin))
    ax.set_aspect('equal')
    index = anim_monkey_info['anim_index'][frame]
    # time = anim_tframe
    # trial_num = np.where(ff_catched_T_sorted > time)[0][0]
    flashing_on_ff = ff_real_position_sorted[flash_on_ff_dict[anim_monkey_info['anim_t'][frame]]]
    relevant_ff = ff_dataframe[['point_index', 'visible', 'ff_index', 'ff_x', 'ff_y']]
    relevant_ff = relevant_ff[relevant_ff['point_index'] == index]
    # in_memory_ffs = relevant_ff[relevant_ff['visible']==0]
    visible_ffs = relevant_ff[relevant_ff['visible'] == 1]
    
    # Plot the arena
    circle_theta = np.arange(0, 2*pi, 0.01)
    ax.plot(np.cos(circle_theta)*1000, np.sin(circle_theta)*1000)

    ax.scatter(ff_position_during_this_trial[:, 0], ff_position_during_this_trial[:, 1], alpha=0.7, c="gray", s=20)
    # ax.scatter(ff_real_position_sorted[trial_num][0], ff_real_position_sorted[trial_num][1], marker='*', c='blue', s = 130, alpha = 0.5)
    ax.scatter(flashing_on_ff[:, 0], flashing_on_ff[:, 1], alpha=1, c="red", s=30)
    # ax.scatter(in_memory_ffs.ff_x , in_memory_ffs.ff_y , alpha=1, c="green", s=30)
    ax.scatter(anim_monkey_info['anim_mx'][:frame+1], anim_monkey_info['anim_my'][:frame+1], s=15, c='royalblue')
    

    # for j in range(len(in_memory_ffs)):
    #   circle = plt.Circle((in_memory_ffs.ff_x.iloc[j], in_memory_ffs.ff_y.iloc[j]), 25, facecolor='grey', edgecolor='orange', alpha=0.3, zorder=1)
    #   ax.add_patch(circle)

    for k in range(len(visible_ffs)):
      circle = plt.Circle((visible_ffs.ff_x.iloc[k], visible_ffs.ff_y.iloc[k]), 25, facecolor='yellow', edgecolor='gray', alpha=0.5, zorder=1)
      ax.add_patch(circle)
      
    if len(believed_ff_dict[frame]) > 0:
      for z in believed_ff_dict[frame]:
        ax.scatter(z[0], z[1], color="purple", s=30)

    # Plot a triangular shape to indicate the direction of the agent
    left_end_x = anim_monkey_info['anim_mx'][frame] + 30 * np.cos(anim_monkey_info['anim_angle'][frame] + 2*pi/9) 
    left_end_y = anim_monkey_info['anim_my'][frame] + 30 * np.sin(anim_monkey_info['anim_angle'][frame] + 2*pi/9)
    right_end_x = anim_monkey_info['anim_mx'][frame] + 30 * np.cos(anim_monkey_info['anim_angle'][frame] - 2*pi/9) 
    right_end_y = anim_monkey_info['anim_my'][frame] + 30 * np.sin(anim_monkey_info['anim_angle'][frame] - 2*pi/9)
    ax.plot(np.array([anim_monkey_info['anim_mx'][frame], left_end_x]), np.array([anim_monkey_info['anim_my'][frame] , left_end_y]), linewidth = 2)
    ax.plot(np.array([anim_monkey_info['anim_mx'][frame], right_end_x]), np.array([anim_monkey_info['anim_my'][frame] , right_end_y]), linewidth = 2)
    return ax





def animate_annotated(i, ax, anim_monkey_info, margin, ff_dataframe, ff_real_position_sorted, ff_position_during_this_trial, \
                      flash_on_ff_dict, believed_ff_dict, ff_catched_T_sorted, annotation_info
    ):
    animate(i, ax, anim_monkey_info, margin, ff_dataframe, ff_real_position_sorted, ff_position_during_this_trial, \
            flash_on_ff_dict, believed_ff_dict)
    index = anim_monkey_info['anim_index'][i]
    time = anim_monkey_info['anim_t'][i]
    trial_num = np.where(ff_catched_T_sorted > time)[0][0]
    annotation = ""
    # If the monkey has captured more than one 1 ff in a cluster
    if annotation_info['n_ff_in_a_row'][trial_num] > 1:
      annotation = annotation + f"Captured {annotation_info['n_ff_in_a_row'][trial_num]} ffs in a cluster\n"
    # If the target stops being on before the monkey captures the previous firefly
    if annotation_info['visible_before_last_one_trial_dummy'][trial_num] == 1:
      annotation = annotation + "Target visible before last captre\n"
    # If the target disappears the latest among visible ffs
    if annotation_info['disappear_latest_trial_dummy'][trial_num] == 1:   
      annotation = annotation + "Target disappears latest\n"
    # If the monkey ignored a closeby ff that suddenly became visible
    if annotation_info['ignore_sudden_flash_point_dummy'][index] > 0:
      annotation = annotation + "Ignored sudden flash\n"
    # If the monkey uses a few tries to capture a firefly
    if annotation_info['try_a_few_times_point_dummy'][index] > 0:
      annotation = annotation + "Try a few times to catch ff\n"
    # If during the trial, the monkey fails to capture a firefly with a few tries and moves on to capture another one 
    if annotation_info['give_up_after_trying_point_dummy'][index] > 0:
      annotation = annotation + "Give up after trying\n"
    ax.text(0.5, 1.04, annotation, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, 
            fontsize=12, color="black", bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return ax
