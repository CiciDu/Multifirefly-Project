
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


"""### Plots"""
def plt_config(title=None, xlim=None, ylim=None, xlabel=None, ylabel=None, colorbar=False, sci=False):
    for field in ['title', 'xlim', 'ylim', 'xlabel', 'ylabel']:
        if eval(field) != None: getattr(plt, field)(eval(field))
    if isinstance(sci, str): plt.ticklabel_format(style='sci', axis=sci, scilimits=(0,0))
    if isinstance(colorbar,str): plt.colorbar(label=colorbar)
    elif colorbar: plt.colorbar(label = '$Number\ of\ Entries$')

@contextmanager
def initiate_plot(dimx=24, dimy=9, dpi=100, fontweight='normal'):
    plt.rcParams['figure.figsize'] = (dimx, dimy)
    plt.rcParams['font.weight'] = fontweight
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams["font.family"] = 'Arial'
    global fig; fig = plt.figure(dpi=dpi)
    yield
    plt.show()

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout






"""## Functions

### find_intersection
source: https://codereview.stackexchange.com/questions/203468/find-the-intervals-which-have-a-non-empty-intersection-with-a-given-interval
"""

def find_intersection(intervals, query):
    """Find intersections between intervals.
    Intervals are open and are represented as pairs (lower bound,
    upper bound).

    Arguments:
    intervals: array_like, shape=(N, 2) -- Array of intervals.
    query: array_like, shape=(2,) -- Interval to query.

    Returns:
    Array of indexes of intervals that overlap with query.

    """
    intervals = np.asarray(intervals)
    lower, upper = query
    return np.where((lower < intervals[:, 1]) & (intervals[:, 0] < upper))[0]

"""### flashing_ff (based on trial)
Find the indices of the fireflies that have flashed during the trial
"""

def flashing_ff(ff_flash_sorted, duration):
  ## Find the index of the fireflies that have flashed during the trial
  ## Input: ff_flash_sorted contains the time that each firefly flashes on and off
  ##        duration is the duration of the trial
  ## Output: flash_index contains the indices of the fireflies that have flashed during the trial (among all fireflies)
  flash_index = []
  for index in range(total_ff_num):
    ff = ff_flash_sorted[index]
    if len(find_intersection(ff, duration)) > 0:
      flash_index.append(index)
  return flash_index

"""###flash_on_ff (dict, based on points)
Find the fireflies that are visible at each time point (for animation)
"""

# Create a dictionary of {time: [indices of fireflies that are visible], ...}
def flash_on_ff(anim_t, currentTrial, num_trials, ff_flash_sorted, ff_life_sorted, ff_catched_T_sorted):
  alive_ff_during_this_trial = np.where((ff_life_sorted[:,1] > ff_catched_T_sorted[currentTrial-num_trials])\
                                        & (ff_life_sorted[:,0] < ff_catched_T_sorted[currentTrial]))[0]
  flash_on_ff_dict={}
  for time in anim_t:
    visible_ff_indices = [index for index in alive_ff_during_this_trial \
     if len(np.where(np.logical_and(ff_flash_sorted[index][:,0] <= time, \
                                       ff_flash_sorted[index][:,1] >= time))[0]) > 0]
    flash_on_ff_dict[time] = visible_ff_indices
  return flash_on_ff_dict

# To call:
# flash_on_ff_dict = flash_on_ff(anim_t, currentTrial, num_trials, ff_flash_sorted)

"""### flash_starttime
Find the earliest time that each firefly begins to flash during that trial
"""

def flash_starttime(flash_index, duration, ff_flash_sorted):
  # Find the earliest time that each firefly begins to flash during that trial
  firefly_on = []
  for index in flash_index:
    ff = ff_flash_sorted[index]
    overlapped_intervals = find_intersection(ff, duration)
    first_start_time = ff[overlapped_intervals].flatten()[0]
    if first_start_time >= duration[0]:
      firefly_on.append(first_start_time)
    else:
      firefly_on.append(duration[0])
  return firefly_on

# Another method
'''
def flash_starttime(flash_index, duration):
  # Find the earliest time that each firefly begins to flash during that trial
  firefly_on = []
  for index in flash_index:
    ff = ff_flash_sorted[index]
    ff_all_starttime = ff[:,0]
    start_time_index = np.argmax(ff_all_starttime >= duration[0])
    start_time = ff_all_starttime[start_time_index]
    # Consider extreme cases
    if start_time >= duration[1]:
      start_time = max(ff_all_starttime[start_time_index-1], duration[0])
    elif ff[start_time_index-1][1] >= duration[0]:
      start_time = duration[0]
    firefly_on.append(start_time)
  firefly_on = np.array(firefly_on)
  return firefly_on
  '''

"""###believed_ff (updated version)
Match the believed positions of the fireflies to the time when they are captured (for animation)
"""

# Create a dictionary of {time: [[believed_ff_position], [believed_ff_position2], ...], ...}
def believed_ff(anim_t, currentTrial, num_trials, ff_believed_position_sorted, ff_catched_T_sorted):
  believed_ff_dict={}
  # For each time point:
  for index in range(len(anim_t)):
    time = anim_t[index]
    believed_ff_indices = [ff_believed_position_sorted[ff] for ff in range(currentTrial, currentTrial+num_trials) if time > ff_catched_T_sorted[ff]]
    believed_ff_dict[index] = believed_ff_indices

  # # The last point
  # believed_ff_indices = [(ff_believed_position_sorted[ff]) for ff in range(currentTrial-num_trials+1, currentTrial+1)]
  # believed_ff_dict[len(anim_t)-1] = believed_ff_indices

  return believed_ff_dict

# To call:
# believed_ff_dict = believed_ff(anim_t, currentTrial, num_trials, ff_believed_position_sorted, ff_catched_T_sorted)

"""### distance_traveled
Find the length of the trajectory run by the monkey in this duration
"""

def distance_traveled(currentTrial, ff_catched_T_sorted, monkey_information):
  duration = [ff_catched_T_sorted[currentTrial-1], ff_catched_T_sorted[currentTrial]]
  cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
  if len(cum_indices) > 5:
    cum_t, cum_angle = monkey_information['monkey_t'][cum_indices],  monkey_information['monkey_angle'][cum_indices]
    cum_mx, cum_my = monkey_information['monkey_x'][cum_indices], monkey_information['monkey_y'][cum_indices] 
    cum_speed = monkey_information['monkey_speed'][cum_indices]
    distance = np.sum((cum_t[1:] - cum_t[:-1])*cum_speed)
  return distance

# To Run:
# distance = distance_traveled(currentTrial)

"""###abs_displacement
Find the absolute displacement between the target for the currentTrial and the target for currentTrial.
Return 9999 if the monkey has hit the border at one point.
"""

def abs_displacement(currentTrial, ff_catched_T_sortedmonkey_information, ff_believed_position_sorted):
  duration = [ff_catched_T_sorted[currentTrial-1], ff_catched_T_sorted[currentTrial]]
  displacement = 0
  cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
  if len(cum_indices) > 5:
    cum_t, cum_angle = monkey_information['monkey_t'][cum_indices],  monkey_information['monkey_angle'][cum_indices]
    cum_mx, cum_my = monkey_information['monkey_x'][cum_indices], monkey_information['monkey_y'][cum_indices] 
    flag = True
    # If the monkey has hit the boundary
    if np.any(cum_mx[1:]-cum_mx[:-1] > 10) or np.any(cum_my[1:]-cum_my[:-1] > 10):
      displacement = 9999
    else:
      displacement = LA.norm(ff_believed_position_sorted[currentTrial]-ff_believed_position_sorted[currentTrial-1])
  return displacement

# To call:
# displacement = abs_displacement(currentTrial):

"""### num_of_stops
Find the stops the monkey made between currentTrial and currentTrial + 1

"""

def num_of_stops(currentTrial, ff_catched_T_sorted, monkey_information, player = "monkey"):
  if player == "monkey":
    duration = [ff_catched_T_sorted[currentTrial-1], ff_catched_T_sorted[currentTrial]]
    cum_indices = np.where((monkey_information['monkey_t'] > duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
    if len(cum_indices) > 5:
      cum_t, cum_angle = monkey_information['monkey_t'][cum_indices],  monkey_information['monkey_angle'][cum_indices]
      cum_mx, cum_my = monkey_information['monkey_x'][cum_indices], monkey_information['monkey_y'][cum_indices] 
      cum_speeddummy = monkey_information['monkey_speeddummy'][cum_indices]
      zerospeed_index = np.where(cum_speeddummy==0)[0]
      if len(zerospeed_index) > 0 :
        zerospeedx, zerospeedy = cum_mx[zerospeed_index], cum_my[zerospeed_index]
        zerospeedindex = cum_indices[zerospeed_index]
        stop0 = np.array(list(zip(zerospeedx,zerospeedy)))
        _, stops_index = np.unique(stop0, axis=0, return_index=True)
        stops = stop0[stops_index[np.argsort(stops_index)]]
        stop_indices = zerospeedindex[stops_index[np.argsort(stops_index)]]
        distinct_stops = [stops[0]] + [stops[i+1] for i in range(len(stops)-1) if LA.norm(np.array((stops[i+1][0]-stops[i][0], stops[i+1][1]-stops[i][1]))) > 0.6]
      else:
        distinct_stops = []
    else:
      distinct_stops = []
    return distinct_stops
  else:
    duration = [ff_catched_T_sorted[currentTrial-1], ff_catched_T_sorted[currentTrial]]
    cum_indices = np.where((monkey_information['monkey_t'] > duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
    if len(cum_indices) > 5:
      cum_mx, cum_my = monkey_information['monkey_x'][cum_indices], monkey_information['monkey_y'][cum_indices] 
      cum_speeddummy = monkey_information['monkey_speeddummy'][cum_indices]
      zerospeed_index = np.where(cum_speeddummy==0)[0]
      if len(zerospeed_index) > 0 :
        zerospeedx, zerospeedy = cum_mx[zerospeed_index], cum_my[zerospeed_index]
        zerospeedindex = cum_indices[zerospeed_index]
        stop0 = np.array(list(zip(zerospeedx,zerospeedy)))
        _, stops_index = np.unique(stop0, axis=0, return_index=True)
        stops = stop0[stops_index[np.argsort(stops_index)]]
        stop_indices = zerospeedindex[stops_index[np.argsort(stops_index)]]
        distinct_stops = stops
      else:
        distinct_stops = []
    else:
      distinct_stops = []
    return distinct_stops

# To call:
# distinct_stops = num_of_stops(currentTrial, ff_catched_T_sorted, monkey_information, player = "monkey")

"""### num_of_stops_indices
Find the stops the monkey made between currentTrial and currentTrial + 1 and return the indices
"""

def num_of_stops_indices(currentTrial, ff_catched_T_sorted, monkey_information, player = "monkey"):
  if player == "monkey":
    duration = [ff_catched_T_sorted[currentTrial-1], ff_catched_T_sorted[currentTrial]]
    distinct_stops_indices = []
    cum_indices = np.where((monkey_information['monkey_t'] > duration[0]) & 
                              (monkey_information['monkey_t'] <= duration[1]))[0]
    if len(cum_indices) > 5:
      cum_t, cum_angle = monkey_information['monkey_t'][cum_indices],  monkey_information['monkey_angle'][cum_indices]
      cum_mx, cum_my = monkey_information['monkey_x'][cum_indices], monkey_information['monkey_y'][cum_indices] 
      cum_speeddummy = monkey_information['monkey_speeddummy'][cum_indices]
      zerospeed_index = np.where(cum_speeddummy==0)[0]
      if len(zerospeed_index) > 0 :
        zerospeedx, zerospeedy = cum_mx[zerospeed_index], cum_my[zerospeed_index]
        zerospeedindex = cum_indices[zerospeed_index]
        stop0 = np.array(list(zip(zerospeedx,zerospeedy)))
        _, stops_index = np.unique(stop0, axis=0, return_index=True)
        stops = stop0[stops_index[np.argsort(stops_index)]]
        stop_indices = zerospeedindex[stops_index[np.argsort(stops_index)]]
        if len(stops) > 1: 
          distinct_stops_indices = [stop_indices[0]] + [stop_indices[i+1] for i in range(len(stops)-1) if LA.norm(np.array((stops[i+1][0]-stops[i][0], stops[i+1][1]-stops[i][1]))) > 0.6]
    return distinct_stops_indices
  else:
    duration = [ff_catched_T_sorted[currentTrial-1], ff_catched_T_sorted[currentTrial]]
    distinct_stops_indices = []
    cum_indices = np.where((monkey_information['monkey_t'] > duration[0]) & 
                              (monkey_information['monkey_t'] <= duration[1]))[0]
    if len(cum_indices) > 5:
      cum_mx, cum_my = monkey_information['monkey_x'][cum_indices], monkey_information['monkey_y'][cum_indices] 
      cum_speeddummy = monkey_information['monkey_speeddummy'][cum_indices]
      zerospeed_index = np.where(cum_speeddummy==0)[0]
      if len(zerospeed_index) > 0 :
        zerospeedx, zerospeedy = cum_mx[zerospeed_index], cum_my[zerospeed_index]
        zerospeedindex = cum_indices[zerospeed_index]
        stop0 = np.array(list(zip(zerospeedx,zerospeedy)))
        _, stops_index = np.unique(stop0, axis=0, return_index=True)
        stops = stop0[stops_index[np.argsort(stops_index)]]
        stop_indices = zerospeedindex[stops_index[np.argsort(stops_index)]]
      return stop_indices
# To call:
# distinct_stops = num_of_stops(currentTrial, ff_catched_T_sorted, monkey_information, player = "monkey")

"""### num_of_stops_since_target_last_seen"""

def num_of_stops_since_target_last_seen(currentTrial, ff_catched_T_sorted, monkey_information, t_last_visible):
  duration = [ff_catched_T_sorted[currentTrial]-t_last_visible[currentTrial-1], ff_catched_T_sorted[currentTrial]]
  if t_last_visible[currentTrial-1] > 50:
    distinct_stops = []
    return distinct_stops
  cum_indices = np.where((monkey_information['monkey_t'] > duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
  if len(cum_indices) > 5:
    cum_t = monkey_information['monkey_t'][cum_indices]
    cum_mx, cum_my = monkey_information['monkey_x'][cum_indices], monkey_information['monkey_y'][cum_indices] 
    cum_speeddummy = monkey_information['monkey_speeddummy'][cum_indices]
    zerospeed_index = np.where(cum_speeddummy==0)[0]
    if len(zerospeed_index) > 0 :
      zerospeedx, zerospeedy = cum_mx[zerospeed_index], cum_my[zerospeed_index]
      zerospeedindex = cum_indices[zerospeed_index]
      stop0 = np.array(list(zip(zerospeedx,zerospeedy)))
      _, stops_index = np.unique(stop0, axis=0, return_index=True)
      stops = stop0[stops_index[np.argsort(stops_index)]]
      stop_indices = zerospeedindex[stops_index[np.argsort(stops_index)]]
      distinct_stops = [stops[0]] + [stops[i+1] for i in range(len(stops)-1) if LA.norm(np.array((stops[i+1][0]-stops[i][0], stops[i+1][1]-stops[i][1]))) > 0.6]
    else:
      distinct_stops = []
  else:
    distinct_stops = []
  return distinct_stops

# To call:
# distinct_stops = num_of_stops(currentTrial, ff_catched_T_sorted, monkey_information, player = "monkey")

"""### find_clusters
Assign each stop with a number that indicates which cluster it belongs to

Note: the algorithm is slightly different from previous notebooks
"""

def find_clusters(currentTrial, distance_between_points, ff_catched_T_sorted, monkey_information, player = "monkey"):
  distinct_stops = num_of_stops(currentTrial, ff_catched_T_sorted, monkey_information, player = player)
  if len(distinct_stops) == 0: 
    clusters = []
  else:
    distinct_stops2 = np.array([[stop[0], stop[1]] for stop in distinct_stops])
    current_cluster = 1
    clusters = [1]
    for i in range(1, len(distinct_stops2)):
      flag = False
      if LA.norm(distinct_stops2[i]-distinct_stops2[i-1]) < distance_between_points:
          clusters.append(current_cluster)
      else: # Create a new cluster
        current_cluster = current_cluster+1
        clusters.append(current_cluster)
  return clusters

"""### angle2ff
Calculate the angle from the monkey to the firefly (from the monkey's perspective) at a given time
"""

def angle2ff(firefly_xy, time, monkey_information):
  num_i = np.where(monkey_information['monkey_t'] == time)[0]
  if len(num_i) == 0:
    time_before = np.where(monkey_information['monkey_t'] <= time)[0]
    if len(time_before) > 0:
      num_i = time_before[-1]
    else:
      num_i = 0

  # Calculate the angle from the monkey to the firefly (overhead view)
  ffradians = math.atan2(firefly_xy[1]-monkey_information['monkey_y'][num_i], firefly_xy[0]-monkey_information['monkey_x'][num_i])
  ffdegrees = math.degrees(ffradians)

  # Find the angle of the firefly from the monkey's perspective
  ff_angle = ffdegrees-monkey_information['monkey_angle'][num_i]
  while abs(ff_angle) > 180:
    if ff_angle > 180:
      ff_angle = ff_angle-360
    elif ff_angle < -180:
      ff_angle = ff_angle + 360
  return ff_angle
# to call:
# ff_angle = angle2ff(firefly_xy, time)

"""### connect_path_ff
Find the lines to be drawn between the nearby fireflies and the path based on the starting time to flash (excluding the target)
"""

def connect_path_ff(cum_t, cum_mx, cum_my, cum_angle, currentTrial, max_distance, ff_flash_sorted, ff_real_position_sorted, total_ff_num):
  x = []
  y = []

  if len(cum_t) > 5: # When cum_t is not empty
    # For each firefly
    total_ff_array = np.arange(total_ff_num)
    non_targets_ff_array = np.delete(total_ff_array, currentTrial)
    for i in non_targets_ff_array:
      # For each duration of being visible
      overlapped_intervals = ff_flash_sorted[i][find_intersection(ff_flash_sorted[i], [cum_t[0], cum_t[-1]])]
      for interval in overlapped_intervals:
        overlapped_indices = np.where((cum_t >= interval[0]) & (cum_t <= interval[1]))[0]
        distances_to_monkey = LA.norm(ff_real_position_sorted[i] - np.stack([cum_mx[overlapped_indices], cum_my[overlapped_indices]], axis=1), axis=1)
        valid_distance_indices = overlapped_indices[np.where(distances_to_monkey < max_distance)[0]]
        
        if len(valid_distance_indices) > 0:
          angles_to_monkey = np.arctan2(ff_real_position_sorted[i,1]-cum_my[valid_distance_indices], \
                                        ff_real_position_sorted[i,0]-cum_mx[valid_distance_indices])-cum_angle[valid_distance_indices]
          ## The following lines turn out to be unnecessary because of the range of output for np.arctan2
          angles_to_monkey[angles_to_monkey > pi] = angles_to_monkey[angles_to_monkey > pi] - 2*pi
          angles_to_monkey[angles_to_monkey < -pi] = angles_to_monkey[angles_to_monkey < -pi] + 2*pi

          angles_adjusted = np.absolute(angles_to_monkey)-np.abs(np.arcsin(np.divide(25, np.maximum(distances_to_monkey[np.where(distances_to_monkey < max_distance)[0]], 25) ))) # use torch clip to get valid arcsin input
          angles_adjusted = np.clip(angles_adjusted, 0, pi)
          angles_adjusted = np.sign(angles_to_monkey)* angles_adjusted


          overall_valid_indices = valid_distance_indices[np.where(np.absolute(angles_adjusted) <= 2*pi/9)[0]]
          
          x = x + [[ff_real_position_sorted[i][0]] + [cum_mx[index]] for index in overall_valid_indices]
          y = y + [[ff_real_position_sorted[i][1]] + [cum_my[index]] for index in overall_valid_indices]
  return x, y
# To call:
# x, y = connect_path_ff(cum_t, cum_mx, cum_my, cum_angle, currentTrial, 250, ff_flash_sorted, ff_real_position_sorted, total_ff_num) total_ff_num)

"""### connect_path_ff2
This deals with the case that there are multiple targets to avoid
"""

def connect_path_ff2(cum_t, cum_mx, cum_my, cum_angle, target_nums, max_distance, ff_flash_sorted, ff_real_position_sorted, total_ff_num):
  x = []
  y = []
  if len(cum_t) > 5: # When cum_t is not empty
    # For each firefly
    total_ff_array = np.arange(total_ff_num)
    non_targets_ff_array = np.delete(total_ff_array, target_nums)
    for i in non_targets_ff_array:
      # For each duration of being visible
      overlapped_intervals = ff_flash_sorted[i][find_intersection(ff_flash_sorted[i], [cum_t[0], cum_t[-1]])]
      for interval in overlapped_intervals:
        overlapped_indices = np.where((cum_t >= interval[0]) & (cum_t <= interval[1]))[0]
        distances_to_monkey = LA.norm(ff_real_position_sorted[i] - np.stack([cum_mx[overlapped_indices], cum_my[overlapped_indices]], axis=1), axis=1)
        valid_distance_indices = overlapped_indices[np.where(distances_to_monkey < max_distance)[0]]
        
        if len(valid_distance_indices) > 0:
          angles_to_monkey = np.arctan2(ff_real_position_sorted[i,1]-cum_my[valid_distance_indices], \
                                        ff_real_position_sorted[i,0]-cum_mx[valid_distance_indices])-cum_angle[valid_distance_indices]
          ## The following lines turn out to be unnecessary because of the range of output for np.arctan2
          angles_to_monkey[angles_to_monkey > pi] = angles_to_monkey[angles_to_monkey > pi] - 2*pi
          angles_to_monkey[angles_to_monkey < -pi] = angles_to_monkey[angles_to_monkey < -pi] + 2*pi

          angles_adjusted = np.absolute(angles_to_monkey)-np.abs(np.arcsin(np.divide(25, np.maximum(distances_to_monkey[np.where(distances_to_monkey < max_distance)[0]], 25) ))) # use torch clip to get valid arcsin input
          angles_adjusted = np.clip(angles_adjusted, 0, pi)
          angles_adjusted = np.sign(angles_to_monkey)* angles_adjusted


          overall_valid_indices = valid_distance_indices[np.where(np.absolute(angles_adjusted) <= 2*pi/9)[0]]
          
          x = x + [[ff_real_position_sorted[i][0]] + [cum_mx[index]] for index in overall_valid_indices]
          y = y + [[ff_real_position_sorted[i][1]] + [cum_my[index]] for index in overall_valid_indices]
  return x, y
# To call:
# x, y = connect_path_ff2(cum_t, cum_mx, cum_my, cum_angle, target_nums, 250, ff_flash_sorted, ff_real_position_sorted, total_ff_num)

"""### connect_path_ff_with_target
No longer exclude target
"""

def connect_path_ff_with_target(cum_t, cum_mx, cum_my, cum_angle, currentTrial, max_distance, ff_flash_sorted, ff_real_position_sorted, total_ff_num):
  x = []
  y = []

  if len(cum_t) > 5: # When cum_t is not empty
    # For each firefly
    for i in range(total_ff_num):
      # For each duration of being visible
      overlapped_intervals = ff_flash_sorted[i][find_intersection(ff_flash_sorted[i], [cum_t[0], cum_t[-1]])]
      for interval in overlapped_intervals:
        overlapped_indices = np.where((cum_t >= interval[0]) & (cum_t <= interval[1]))[0]
        distances_to_monkey = LA.norm(ff_real_position_sorted[i] - np.stack([cum_mx[overlapped_indices], cum_my[overlapped_indices]], axis=1), axis=1)
        valid_distance_indices = overlapped_indices[np.where(distances_to_monkey < max_distance)[0]]
        
        if len(valid_distance_indices) > 0:
          angles_to_monkey = np.arctan2(ff_real_position_sorted[i,1]-cum_my[valid_distance_indices], \
                                        ff_real_position_sorted[i,0]-cum_mx[valid_distance_indices])-cum_angle[valid_distance_indices]
          ## The following lines turn out to be unnecessary because of the range of output for np.arctan2
          angles_to_monkey[angles_to_monkey > pi] = angles_to_monkey[angles_to_monkey > pi] - 2*pi
          angles_to_monkey[angles_to_monkey < -pi] = angles_to_monkey[angles_to_monkey < -pi] + 2*pi

          angles_adjusted = np.absolute(angles_to_monkey)-np.abs(np.arcsin(np.divide(25, np.maximum(distances_to_monkey[np.where(distances_to_monkey < max_distance)[0]], 25) ))) # use torch clip to get valid arcsin input
          angles_adjusted = np.clip(angles_adjusted, 0, pi)
          angles_adjusted = np.sign(angles_to_monkey)* angles_adjusted
          overall_valid_indices = valid_distance_indices[np.where(np.absolute(angles_adjusted) <= 2*pi/9)[0]]
          x = x + [[ff_real_position_sorted[i][0]] + [cum_mx[index]] for index in overall_valid_indices]
          y = y + [[ff_real_position_sorted[i][1]] + [cum_my[index]] for index in overall_valid_indices]
  return x, y
  # To call:
  # x, y = connect_path_ff(cum_t, cum_mx, cum_my, cum_angle, currentTrial, 250, ff_flash_sorted, ff_real_position_sorted, total_ff_num)

"""### onoff_lines
Find the parts of the path where the target has been on. The angle of the firefly is considered. The firefly is considered visible only when it's at the right angle.
"""

def onoff_lines(cum_mx, cum_my, cum_angle, currentTrial, duration, ff_real_position_sorted, ff_flash_sorted):
  target_ff_flash = ff_flash_sorted[currentTrial]
  overlapped_intervals = target_ff_flash[find_intersection(target_ff_flash, duration)]
  x_onoff_lines=np.array([])
  y_onoff_lines=np.array([])
  for interval in overlapped_intervals:
    correspondingIndex_onPath = np.where((cum_t > np.maximum(interval[0], duration[0])) & \
                                          (cum_t < np.minimum(interval[1], duration[1])))[0]
    x_onoff_lines=np.append(x_onoff_lines, cum_mx[correspondingIndex_onPath])
    y_onoff_lines=np.append(y_onoff_lines, cum_my[correspondingIndex_onPath])
  return x_onoff_lines, y_onoff_lines

"""### onoff_lines_rightangle
Find the parts of the path where the target has been on. The angle of the firefly is considered. The firefly is considered visible only when it's at the right angle.
"""

def onoff_lines_rightangle(cum_t, cum_mx, cum_my, cum_angle, currentTrial, duration, ff_real_position_sorted, ff_flash_sorted):
  target_ff_flash = ff_flash_sorted[currentTrial]
  overlapped_intervals = target_ff_flash[find_intersection(target_ff_flash, duration)]

  indices_onoff_lines = []
  for interval in overlapped_intervals:
    correspondingIndex_onPath = np.where((cum_t > np.maximum(interval[0], duration[0])) & \
                                          (cum_t < np.minimum(interval[1], duration[1])))[0]
    distances_to_monkey = LA.norm(ff_real_position_sorted[currentTrial] - np.stack([cum_mx[correspondingIndex_onPath], cum_my[correspondingIndex_onPath]], axis=1), axis=1)
    angles_to_monkey = np.arctan2(ff_real_position_sorted[currentTrial,1]-cum_my[correspondingIndex_onPath], \
                                  ff_real_position_sorted[currentTrial,0]-cum_mx[correspondingIndex_onPath]) -cum_angle[correspondingIndex_onPath]                                 
    angles_to_monkey[angles_to_monkey > pi] = angles_to_monkey[angles_to_monkey > pi] - 2*pi
    angles_to_monkey[angles_to_monkey < -pi] = angles_to_monkey[angles_to_monkey < -pi] + 2*pi
    angles_adjusted = np.absolute(angles_to_monkey)-np.abs(np.arcsin(np.divide(25, np.maximum(distances_to_monkey, 25) ))) # use torch clip to get valid arcsin input
    angles_adjusted = np.clip(angles_adjusted, 0, pi)
    angles_adjusted = np.sign(angles_to_monkey)* angles_adjusted

    correspondingIndex_onPath_angle = correspondingIndex_onPath[np.where(np.absolute(angles_adjusted) <= 2*pi/9)[0]]
    indices_onoff_lines = indices_onoff_lines + correspondingIndex_onPath_angle.tolist()
  return indices_onoff_lines

# To call:
# indices_onoff_lines = onoff_lines_rightangle(cum_t, cum_mx, cum_my, cum_angle, currentTrial, duration, ff_real_position_sorted, ff_flash_sorted)

