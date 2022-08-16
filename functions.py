
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
def flash_on_ff(anim_t, currentTrial, num_trials, ff_flash_sorted):
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

def flash_starttime(flash_index, duration):
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

def distance_traveled(currentTrial):
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

def abs_displacement(currentTrial):
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

if MONKEY_DATA == True:
  def num_of_stops(currentTrial):
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
  def num_of_stops(currentTrial):
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
# distinct_stops = num_of_stops(currentTrial)

"""### num_of_stops_indices
Find the stops the monkey made between currentTrial and currentTrial + 1 and return the indices
"""

if MONKEY_DATA == True:
  def num_of_stops_indices(currentTrial):
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
  def num_of_stops_indices(currentTrial):
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
# distinct_stops = num_of_stops(currentTrial)

"""### num_of_stops_since_target_last_seen"""

def num_of_stops_since_target_last_seen(currentTrial):
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
# distinct_stops = num_of_stops(currentTrial)

"""### find_clusters
Assign each stop with a number that indicates which cluster it belongs to

Note: the algorithm is slightly different from previous notebooks
"""

def find_clusters(currentTrial, distance_between_points):
  distinct_stops = num_of_stops(currentTrial)
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

def angle2ff(firefly_xy, time):
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

def onoff_lines(cum_t, cum_mx, cum_my, ff_real_position_sorted, currentTrial, duration):
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

"""### **PlotTrials** (for LSTM)

Among other things, here I eliminate the condition "ff_distance" < 250
"""

def PlotTrials(currentTrial,
                num_trials, 
                trail_color = "orange", # "orange" or "viridis" or None
                show_reward_boundary = False,
                show_stops = False,
                show_colorbar = False, 
                show_believed_target_positions = False,
                show_connect_path_target = np.array([]), # np.array([]) or target_nums
                show_connect_path_pre_target = np.array([]), # np.array([]) or target_nums
                show_connect_path_ff = np.array([]),
                trial_to_show_cluster = None, # None, 0, or -1
                show_scale_bar = False,
                trial_to_show_cluster_around_target = None,
                cluster_on_off_lines = False,
                show_start = True,
                #target_nums = "default", 
                ):
  


    cum_mxy_rotate = np.matmul(R, np.stack((cum_mx, cum_my)))
    if trail_color == "orange":
      axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1],marker = 'o',s = 70, color="orange", zorder=2)
    elif trail_color == "viridis":
      axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1],marker = 'o',s = 70, c = cum_speed, zorder=2)
    else:
      axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1],marker = 'o',s = 70, color = "yellow", zorder=2)

    if show_start:
      # Plot the start
      axes.scatter(cum_mxy_rotate[0,0], cum_mxy_rotate[1,0],marker = '^',s = 220, color="gold", zorder=3, alpha=0.5)



    if show_stops:
      zerospeed_index = np.where(cum_speeddummy==0)
      zerospeedx, zerospeedy = cum_mx[zerospeed_index], cum_my[zerospeed_index]
      zerospeed_rotate = np.matmul(R, np.stack((zerospeedx, zerospeedy)))
      axes.scatter(zerospeed_rotate[0], zerospeed_rotate[1],marker = '*',s = 160, alpha = 0.7, color="black", zorder=2)

    ff_position_rotate = np.matmul(R, np.stack((ff_position_during_this_trial.T[0], ff_position_during_this_trial.T[1])))
    axes.scatter(ff_position_rotate[0], ff_position_rotate[1], marker='o', s=10, color="magenta",  zorder=2)

    if show_believed_target_positions:
      ff_believed_position_rotate = np.matmul(R, np.stack((ff_believed_position_sorted[currentTrial-num_trials+1:currentTrial+1].T[0], ff_believed_position_sorted[currentTrial-num_trials+1:currentTrial+1].T[1])))
      axes.scatter(ff_believed_position_rotate[0], ff_believed_position_rotate[1], marker = '*',s=185, color="red", alpha=0.75, zorder=2)
    
    if show_reward_boundary:
      for i in ff_position_rotate.T:
        circle2 = plt.Circle((i[0], i[1]), 25, facecolor='grey', edgecolor='orange', alpha=0.45, zorder=1)
        axes.add_patch(circle2)

    if trial_to_show_cluster != None:
    # Find the indices of ffs in the cluster
      cluster_indices = np.unique(cluster_dataframe_point[cluster_dataframe_point['target_index']==currentTrial+trial_to_show_cluster].ff_index.to_numpy())
      cluster_ff_positions = ff_real_position_sorted[np.array(cluster_indices)]
      cluster_ff_rotate = np.matmul(R, np.stack((cluster_ff_positions.T[0], cluster_ff_positions.T[1])))
      axes.scatter(cluster_ff_rotate[0], cluster_ff_rotate[1], marker='o', c = "blue", s=25, zorder = 4) 

    if show_connect_path_target.any():
      xy_onoff_lines = ff_dataframe.loc[ff_dataframe['target_index'].isin(show_connect_path_target)]
      xy_onoff_lines = np.array(xy_onoff_lines.loc[(xy_onoff_lines['ff_index']==currentTrial) & (xy_onoff_lines['visible']==1)][['monkey_x', 'monkey_y']])
      onoff_lines_rotate = np.matmul(R, xy_onoff_lines.T)
      axes.scatter(onoff_lines_rotate[0], onoff_lines_rotate[1], s=50, c="green", alpha=0.8, zorder=5) 

    if show_connect_path_pre_target.any():
      xy_onoff_lines = ff_dataframe.loc[ff_dataframe['target_index'].isin(show_connect_path_pre_target)]
      xy_onoff_lines = np.array(xy_onoff_lines.loc[(xy_onoff_lines['ff_index']==currentTrial-1) & (xy_onoff_lines['visible']==1)][['monkey_x', 'monkey_y']])
      onoff_lines_rotate = np.matmul(R, xy_onoff_lines.T)
      axes.scatter(onoff_lines_rotate[0], onoff_lines_rotate[1], s=65, c="aqua",  alpha=0.8, zorder=3)

    if show_connect_path_ff.any():
      temp_dataframe = ff_dataframe.loc[ff_dataframe['target_index'].isin(show_connect_path_ff)]
      temp_dataframe = temp_dataframe.loc[(temp_dataframe['visible']==1)][['ff_x', 'ff_y', 'monkey_x', 'monkey_y']]
      #temp_dataframe = temp_dataframe.loc[~temp_dataframe['ff_index'].isin(target_nums)]
      temp_array = temp_dataframe.to_numpy()
      temp_ff_positions = np.matmul(R, temp_array[:,:2].T)
      temp_monkey_positions = np.matmul(R, temp_array[:,2:].T)
      for j in range(len(temp_array)):
        axes.plot(np.stack([temp_ff_positions[0,j], temp_monkey_positions[0,j]]), np.stack([temp_ff_positions[1,j], temp_monkey_positions[1,j]]), '-', alpha=0.3, linewidth=1.5, c="#a940f5")
        #axes.plot(temp_ff_positions[0,j], temp_ff_positions[1,j], '-', alpha=0.2, marker="o", markersize=5, color="brown")

    # if show_connect_path_ff.any():
    #   if num_trials == 1:
    #     x, y = connect_path_ff(cum_t, cum_mx, cum_my, cum_angle, currentTrial, 400, ff_flash_sorted, ff_real_position_sorted, total_ff_num)
    #   else:
    #     target_nums = np.arange(currentTrial-num_trials+1, currentTrial+1)
    #     x, y = connect_path_ff2(cum_t, cum_mx, cum_my, cum_angle, target_nums, 400, ff_flash_sorted, ff_real_position_sorted, total_ff_num)
    #   for j in range(len(x)):
    #     xy_rotate = np.matmul(R, np.stack((x[j], y[j])))
    #     axes.plot(xy_rotate[0], xy_rotate[1], '-', alpha=0.2, c="#a940f5")
    #     axes.plot(xy_rotate[0][0], xy_rotate[1][0], '-', alpha=0.2, marker="o", markersize=5, color="brown")

    if trial_to_show_cluster_around_target != None:
      cluster_ff_pos = ffs_around_target_positions[currentTrial+trial_to_show_cluster_around_target]
      if len(cluster_ff_pos) > 0:
        ffs_around_target_rotate = np.matmul(R, np.stack((cluster_ff_pos.T[0], cluster_ff_pos.T[1])))
        axes.scatter(ffs_around_target_rotate[0],ffs_around_target_rotate[1], marker='o', s=30, color="blue", zorder=4) 
      if cluster_on_off_lines:
        # Find on_off_lines for ffs in the cluster
        for i in range(len(cluster_ff_pos)):
          index = np.array(ff_dataframe[(np.isclose(np.array(ff_dataframe['ff_x']),cluster_ff_pos[i, 0])) & (np.isclose(np.array(ff_dataframe['ff_y']),cluster_ff_pos[i, 1]))]['ff_index'])
          if len(index) > 0:
            index = index[0]
            #index = ffs_around_target_indices[currentTrial-trial_to_show_cluster_around_target][i]
            xy_onoff_lines = ff_dataframe.loc[(ff_dataframe['time']>= duration[0])&(ff_dataframe['time']<= duration[1])]
            xy_onoff_lines = xy_onoff_lines.loc[(xy_onoff_lines['ff_index']==index) & (xy_onoff_lines['visible']==1)]
            xy_onoff_lines2 = np.array(xy_onoff_lines[['monkey_x', 'monkey_y']])
            onoff_lines_rotate = np.matmul(R, xy_onoff_lines2.T)
            axes.scatter(onoff_lines_rotate[0], onoff_lines_rotate[1], s=80-10*i, color=list_of_colors[i], alpha=0.8, zorder=3+i) 
            # Use corresponding color for that ff
            xy_onoff_lines3 = np.array(xy_onoff_lines[['ff_x', 'ff_y']]) 
            ffs_around_target_rotate = np.matmul(R, xy_onoff_lines3.T)
            axes.scatter(ffs_around_target_rotate[0],ffs_around_target_rotate[1], marker='o', s=140, alpha = 0.8, color=list_of_colors[i], zorder=3) 


    
    
    
    if show_scale_bar:
      scale1 = ScaleBar(  
      dx=1, length_fraction=0.2, fixed_value=100,
          location='upper left',  # in relation to the whole plot
          label_loc='left', scale_loc='bottom'  # in relation to the line
      )
      axes.add_artist(scale1)
      axes.xaxis.set_major_locator(ticker.NullLocator())
      axes.yaxis.set_major_locator(ticker.NullLocator())

    xmin, xmax = np.min(cum_mxy_rotate[0]), np.max(cum_mxy_rotate[0])
    ymin, ymax = np.min(cum_mxy_rotate[1]), np.max(cum_mxy_rotate[1])
    bigger_width = max(xmax-xmin, ymax-ymin)
    xmiddle, ymiddle = (xmin+xmax)/2, (ymin+ymax)/2
    xmin, xmax = xmiddle-bigger_width/2, xmiddle+bigger_width/2
    ymin, ymax = ymiddle-bigger_width/2, ymiddle+bigger_width/2
    margin = max(bigger_width/5, 150)
    axes.set_xlim((xmin-margin, xmax+margin))
    axes.set_ylim((ymin-margin, ymax+margin))
    axes.set_aspect('equal')


    if show_colorbar == True:
      # Make the black and red colorbar
      A = np.reshape([1,2,1,2,2,2], (2,3))# The numbers don't matter much
      norm_bins = np.array([0.5, 1.5, 2.5])
      # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
      col_dict={1:"black",2:"red"}
      # We create a colormar from our list of colors
      speed_cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
      ## Make normalizer and formatter
      norm = matplotlib.colors.BoundaryNorm(norm_bins, 2, clip=True)
      labels = np.array(["No Reward", "Reward"])
      fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
      # Plot our figure
      im = axes.imshow(A, cmap=speed_cm, extent=[0,0,0,0], norm=norm)
      cax2 = fig.add_axes([0.95, 0.15, 0.05, 0.2])
      cb = fig.colorbar(im, format=fmt, ticks=np.array([1., 2.]), cax=cax2)
      cb.ax.tick_params(width=0)
      cb.ax.set_title('Stopping Points', ha='left')
      if trail_color == "orange":
        A = np.reshape([1,2,1,2,2,2], (2,3))# The numbers don't matter much
        norm_bins = np.array([0.5, 1.5, 2.5])
        # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
        col_dict={1:"green",2:"orange"}
        # We create a colormar from our list of colors
        speed_cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
        ## Make normalizer and formatter
        norm = matplotlib.colors.BoundaryNorm(norm_bins, 2, clip=True)
        labels = np.array(["Top Target Visible", "Top Target Not Visible"])
        fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
        # Plot our figure
        im = axes.imshow(A, cmap=speed_cm, extent=[0,0,0,0], norm=norm)
        cax2 = fig.add_axes([0.95, 0.5, 0.05, 0.2])
        cb = fig.colorbar(im, format=fmt, ticks=np.array([1., 2.]), cax=cax2)
        cb.ax.tick_params(width=0)
        cb.ax.set_title('Path Colors', ha='left', y=1.04)
      elif trail_color == "viridis":
        cmap = cm.viridis
        norm = matplotlib.colors.Normalize(vmin=0, vmax=200)
        cax = fig.add_axes([0.95, 0.4, 0.05, 0.43])
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=cax, orientation='vertical')
        cbar.ax.set_title('Speed(cm/s)', ha='left', y=1.04)
        cbar.ax.tick_params(axis='y', color='white', direction="in", right=True,length=5, width=1.5)
        cbar.outline.remove()
      global Show_Colorbar 
      Show_Colorbar = False

"""### **PlotTrials** """

def PlotTrials(currentTrial,
                num_trials, 
                trail_color = "orange", # "orange" or "viridis" or None
                show_reward_boundary = False,
                show_stops = False,
                show_colorbar = False, 
                show_believed_target_positions = False,
                show_connect_path_target = np.array([]), # np.array([]) or target_nums
                show_connect_path_pre_target = np.array([]), # np.array([]) or target_nums
                show_connect_path_ff = np.array([]),
                trial_to_show_cluster = None, # None, 0, or -1
                show_scale_bar = False,
                show_start = False,
                trial_to_show_cluster_around_target = None,
                cluster_on_off_lines = False,
                #target_nums = "default", 
                ):
   

    cum_mxy_rotate = np.matmul(R, np.stack((cum_mx, cum_my)))
    if trail_color == "orange":
      axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1],marker = 'o',s = 10, color="orange", zorder=2)
    elif trail_color == "viridis":
      axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1],marker = 'o',s = 10, c = cum_speed, zorder=2)
    else:
      axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1],marker = 'o',s = 10, color = "yellow", zorder=2)

    if show_start:
      # Plot the start
      axes.scatter(cum_mxy_rotate[0,0], cum_mxy_rotate[1,0],marker = 'o',s = 100, color="purple", zorder=3, alpha=0.5)

    if show_stops:
      zerospeed_index = np.where(cum_speeddummy==0)
      zerospeedx, zerospeedy = cum_mx[zerospeed_index], cum_my[zerospeed_index]
      zerospeed_rotate = np.matmul(R, np.stack((zerospeedx, zerospeedy)))
      axes.scatter(zerospeed_rotate[0], zerospeed_rotate[1],marker = '*',s = 150, color="black", zorder=2)

    ff_position_rotate = np.matmul(R, np.stack((ff_position_during_this_trial.T[0], ff_position_during_this_trial.T[1])))
    axes.scatter(ff_position_rotate[0], ff_position_rotate[1], marker='o', s=10, color="magenta",  zorder=2)

    if show_believed_target_positions:
      ff_believed_position_rotate = np.matmul(R, np.stack((ff_believed_position_sorted[currentTrial-num_trials+1:currentTrial+1].T[0], ff_believed_position_sorted[currentTrial-num_trials+1:currentTrial+1].T[1])))
      axes.scatter(ff_believed_position_rotate[0], ff_believed_position_rotate[1], marker = '*',s=120, color="red", alpha=0.75, zorder=2)
    
    if show_reward_boundary:
      for i in ff_position_rotate.T:
        circle2 = plt.Circle((i[0], i[1]), 25, facecolor='grey', edgecolor='orange', alpha=0.45, zorder=1)
        axes.add_patch(circle2)

    if trial_to_show_cluster != None:
    # Find the indices of ffs in the cluster
      cluster_indices = np.unique(cluster_dataframe_point[cluster_dataframe_point['target_index']==currentTrial+trial_to_show_cluster].ff_index.to_numpy())
      cluster_ff_positions = ff_real_position_sorted[np.array(cluster_indices)]
      cluster_ff_rotate = np.matmul(R, np.stack((cluster_ff_positions.T[0], cluster_ff_positions.T[1])))
      axes.scatter(cluster_ff_rotate[0], cluster_ff_rotate[1], marker='o', c = "blue", s=25, zorder = 4) 

    if show_connect_path_target.any():
      xy_onoff_lines = ff_dataframe.loc[ff_dataframe['target_index'].isin(show_connect_path_target)]
      xy_onoff_lines = np.array(xy_onoff_lines.loc[(xy_onoff_lines['ff_index']==currentTrial) & (xy_onoff_lines['visible']==1)&(xy_onoff_lines['ff_distance']<250)][['monkey_x', 'monkey_y']])
      onoff_lines_rotate = np.matmul(R, xy_onoff_lines.T)
      axes.scatter(onoff_lines_rotate[0], onoff_lines_rotate[1], s=30, c="green", alpha=0.4, zorder=4) 

    if show_connect_path_pre_target.any():
      xy_onoff_lines = ff_dataframe.loc[ff_dataframe['target_index'].isin(show_connect_path_pre_target)]
      xy_onoff_lines = np.array(xy_onoff_lines.loc[(xy_onoff_lines['ff_index']==currentTrial-1) & (xy_onoff_lines['visible']==1)&(xy_onoff_lines['ff_distance']<250)][['monkey_x', 'monkey_y']])
      onoff_lines_rotate = np.matmul(R, xy_onoff_lines.T)
      axes.scatter(onoff_lines_rotate[0], onoff_lines_rotate[1], s=40, c="aqua",  alpha=0.6, zorder=3)

    if show_connect_path_ff.any():
      target_nums = np.arange(currentTrial-num_trials+1, currentTrial+1)
      temp_dataframe = ff_dataframe.loc[ff_dataframe['target_index'].isin(target_nums)]
      temp_dataframe = temp_dataframe.loc[(temp_dataframe['ff_distance']<250) & (temp_dataframe['visible']==1)]
      temp_dataframe = temp_dataframe.loc[~temp_dataframe['ff_index'].isin(target_nums)][['ff_x', 'ff_y', 'monkey_x', 'monkey_y']]
      temp_array = temp_dataframe.to_numpy()
      temp_ff_positions = np.matmul(R, temp_array[:,:2].T)
      temp_monkey_positions = np.matmul(R, temp_array[:,2:].T)
      for j in range(len(temp_array)):
        axes.plot(np.stack([temp_ff_positions[0,j], temp_monkey_positions[0,j]]), np.stack([temp_ff_positions[1,j], temp_monkey_positions[1,j]]), '-', alpha=0.2, c="#a940f5")
        axes.plot(temp_ff_positions[0,j], temp_ff_positions[1,j], '-', alpha=0.2, marker="o", markersize=5, color="brown")

    # if show_connect_path_ff.any():
    #   if num_trials == 1:
    #     x, y = connect_path_ff(cum_t, cum_mx, cum_my, cum_angle, currentTrial, 250, ff_flash_sorted, ff_real_position_sorted, total_ff_num)
    #   else:
    #     target_nums = np.arange(currentTrial-num_trials+1, currentTrial+1)
    #     x, y = connect_path_ff2(cum_t, cum_mx, cum_my, cum_angle, target_nums, 250, ff_flash_sorted, ff_real_position_sorted, total_ff_num)
    #   for j in range(len(x)):
    #     xy_rotate = np.matmul(R, np.stack((x[j], y[j])))
    #     axes.plot(xy_rotate[0], xy_rotate[1], '-', alpha=0.2, c="#a940f5")
    #     axes.plot(xy_rotate[0][0], xy_rotate[1][0], '-', alpha=0.2, marker="o", markersize=5, color="brown")


    if trial_to_show_cluster_around_target != None:
      cluster_ff_pos = ffs_around_target_positions[currentTrial+trial_to_show_cluster_around_target]
      if len(cluster_ff_pos) > 0:
        ffs_around_target_rotate = np.matmul(R, np.stack((cluster_ff_pos.T[0], cluster_ff_pos.T[1])))
        axes.scatter(ffs_around_target_rotate[0],ffs_around_target_rotate[1], marker='o', s=30, color="blue", zorder=4) 
      if cluster_on_off_lines:
        # Find on_off_lines for ffs in the cluster
        for i in range(len(cluster_ff_pos)):
          index = np.array(ff_dataframe[(np.isclose(np.array(ff_dataframe['ff_x']),cluster_ff_pos[i, 0])) & (np.isclose(np.array(ff_dataframe['ff_y']),cluster_ff_pos[i, 1]))]['ff_index'])
          if len(index) > 0:
            index = index[0]
            #index = ffs_around_target_indices[currentTrial-trial_to_show_cluster_around_target][i]
            xy_onoff_lines = ff_dataframe.loc[ff_dataframe['target_index']==currentTrial]
            xy_onoff_lines2 = np.array(xy_onoff_lines.loc[(xy_onoff_lines['ff_index']==index) & (xy_onoff_lines['visible']==1)][['monkey_x', 'monkey_y']])
            onoff_lines_rotate = np.matmul(R, xy_onoff_lines2.T)
            axes.scatter(onoff_lines_rotate[0], onoff_lines_rotate[1], s=15-3*i, color=list_of_colors[i], alpha=0.4, zorder=3+i) 
            # Use corresponding color for that ff
            xy_onoff_lines3 = np.array(xy_onoff_lines[['ff_x', 'ff_y']]) 
            ffs_around_target_rotate = np.matmul(R, xy_onoff_lines3.T)
            axes.scatter(ffs_around_target_rotate[0],ffs_around_target_rotate[1], marker='o', s=100, alpha = 0.5, color=list_of_colors[i], zorder=3) 



    if show_scale_bar:
      scale1 = ScaleBar(  
      dx=1, length_fraction=0.2, fixed_value=100,
          location='upper left',  # in relation to the whole plot
          label_loc='left', scale_loc='bottom'  # in relation to the line
      )
      axes.add_artist(scale1)
      axes.xaxis.set_major_locator(ticker.NullLocator())
      axes.yaxis.set_major_locator(ticker.NullLocator())

    xmin, xmax = np.min(cum_mxy_rotate[0]), np.max(cum_mxy_rotate[0])
    ymin, ymax = np.min(cum_mxy_rotate[1]), np.max(cum_mxy_rotate[1])
    bigger_width = max(xmax-xmin, ymax-ymin)
    xmiddle, ymiddle = (xmin+xmax)/2, (ymin+ymax)/2
    xmin, xmax = xmiddle-bigger_width/2, xmiddle+bigger_width/2
    ymin, ymax = ymiddle-bigger_width/2, ymiddle+bigger_width/2
    margin = max(bigger_width/5, 150)
    axes.set_xlim((xmin-margin, xmax+margin))
    axes.set_ylim((ymin-margin, ymax+margin))
    axes.set_aspect('equal')


    if show_colorbar == True:
      # Make the black and red colorbar
      A = np.reshape([1,2,1,2,2,2], (2,3))# The numbers don't matter much
      norm_bins = np.array([0.5, 1.5, 2.5])
      # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
      col_dict={1:"black",2:"red"}
      # We create a colormar from our list of colors
      speed_cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
      ## Make normalizer and formatter
      norm = matplotlib.colors.BoundaryNorm(norm_bins, 2, clip=True)
      labels = np.array(["No Reward", "Reward"])
      fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
      # Plot our figure
      im = axes.imshow(A, cmap=speed_cm, extent=[0,0,0,0], norm=norm)
      cax2 = fig.add_axes([0.95, 0.15, 0.05, 0.2])
      cb = fig.colorbar(im, format=fmt, ticks=np.array([1., 2.]), cax=cax2)
      cb.ax.tick_params(width=0)
      cb.ax.set_title('Stopping Points', ha='left')
      if trail_color == "orange":
        A = np.reshape([1,2,1,2,2,2], (2,3))# The numbers don't matter much
        norm_bins = np.array([0.5, 1.5, 2.5])
        # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
        col_dict={1:"green",2:"orange"}
        # We create a colormar from our list of colors
        speed_cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
        ## Make normalizer and formatter
        norm = matplotlib.colors.BoundaryNorm(norm_bins, 2, clip=True)
        labels = np.array(["Top Target Visible", "Top Target Not Visible"])
        fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
        # Plot our figure
        im = axes.imshow(A, cmap=speed_cm, extent=[0,0,0,0], norm=norm)
        cax2 = fig.add_axes([0.95, 0.5, 0.05, 0.2])
        cb = fig.colorbar(im, format=fmt, ticks=np.array([1., 2.]), cax=cax2)
        cb.ax.tick_params(width=0)
        cb.ax.set_title('Path Colors', ha='left', y=1.04)
      elif trail_color == "viridis":
        cmap = cm.viridis
        norm = matplotlib.colors.Normalize(vmin=0, vmax=200)
        cax = fig.add_axes([0.95, 0.4, 0.05, 0.43])
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=cax, orientation='vertical')
        cbar.ax.set_title('Speed(cm/s)', ha='left', y=1.04)
        cbar.ax.tick_params(axis='y', color='white', direction="in", right=True,length=5, width=1.5)
        cbar.outline.remove()
      global Show_Colorbar 
      Show_Colorbar = False

"""### PlotPoints"""

def PlotPoints(point,
                total_time, 
                show_all_ff,
                show_flash_on_ff,
                show_visible_ff,
                show_in_memory_ff, 
                show_target,
                show_reward_boundary,
                show_legend,
                show_colorbar, 
                show_scale_bar,
                trial_num=None,
                **kwargs):


          
  
    alive_ff_indices= np.array([i for i,value in 
                               enumerate(ff_life_sorted) if (value[-1]>=time) and (value[0]<time)]) 
    alive_ff_positions = ff_real_position_sorted[alive_ff_indices]
    if show_all_ff:
      axes.scatter(alive_ff_positions.T[0], alive_ff_positions.T[1], color="grey", s=30)

    if show_flash_on_ff:
      on_ff_indices = [] # Gives the indices of the ffs that are on at this point
      # For each firefly in ff_flash_sorted:
      for index, firefly in enumerate(ff_flash_sorted):
        # If the firefly has flashed during that trial:
        if index in alive_ff_indices: 
            # Let's see if the firefly has flashed at that exact moment
            for interval in firefly:
              if interval[0] <= time <= interval[1]:
                on_ff_indices.append(index) 
                break  
      on_ff_indices = np.array(on_ff_indices) 
      on_ff_positions = ff_real_position_sorted[on_ff_indices]
      axes.scatter(on_ff_positions.T[0], on_ff_positions.T[1], color="red", s=120, marker = '*', alpha = 0.7)
    
    if show_visible_ff:
      visible_ffs = ff_dataframe[(ff_dataframe['point_index']==point)&(ff_dataframe['visible']==1)][['ff_x', 'ff_y']]
      axes.scatter(visible_ffs['ff_x'], visible_ffs['ff_y'], color="orange", s=40)

    if show_in_memory_ff:
      in_memory_ffs = ff_dataframe[(ff_dataframe['point_index']==point)&(ff_dataframe['visible']==0)][['ff_x', 'ff_y']]
      axes.scatter(in_memory_ffs['ff_x'], in_memory_ffs['ff_y'], color="green", s=40)
    
    if show_target:
      if trial_num == None:
        raise ValueError("If show_target, then trial_num cannot be None")
      target_num  = distance_dataframe['trial'].iloc[point]
      target_position = ff_real_position_sorted[trial_num]
      axes.scatter(target_position[0], target_position[1], marker = '*', s=200, color="grey", alpha=0.35)


    if show_legend == True:
      legend_names = []
      if show_all_ff:
        legend_names.append("Invisible")
      if show_flash_on_ff:
        legend_names.append("Flash On")
      if show_visible_ff:
        legend_names.append("Visible")
      if show_in_memory_ff:
        legend_names.append("In memory")
      if show_target:
        legend_names.append("Target")
      axes.legend(legend_names, loc='upper right')
      global Show_Legend
      Show_Legend = False

    if show_reward_boundary:
      if show_all_ff:
        for i in range(len(alive_ff_positions)):
          circle2 = plt.Circle((alive_ff_positions[i, 0], alive_ff_positions[i, 1]), 20, facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
          axes.add_patch(circle2)
      elif show_flash_on_ff:
        if show_flash_on_ff:
          for i in range(len(on_ff_positions)):
            circle2 = plt.Circle((on_ff_positions[i, 0], on_ff_positions[i, 1]), 20, facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
            axes.add_patch(circle2)
        if show_in_memory_ff:
          for i in range(len(in_memory_ffs)):
            circle2 = plt.Circle((in_memory_ffs['ff_x'].iloc[i], in_memory_ffs['ff_y'].iloc[i]), 20, facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
            axes.add_patch(circle2) 
      elif show_visible_ff:
        for i in range(len(visible_ffs)):
          circle2 = plt.Circle((visible_ffs['ff_x'].iloc[i], visible_ffs['ff_y'].iloc[i]), 20, facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
          axes.add_patch(circle2)  
        if show_in_memory_ff:
          for i in range(len(in_memory_ffs)):
            circle2 = plt.Circle((in_memory_ffs['ff_x'].iloc[i], in_memory_ffs['ff_y'].iloc[i]), 20, facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
            axes.add_patch(circle2)  

    axes.scatter(cum_mx, cum_my, s=15, c=index_temp, cmap="Blues") 

    xmin, xmax = np.min(cum_mx), np.max(cum_mx)
    ymin, ymax = np.min(cum_my), np.max(cum_my)
    bigger_width = max(xmax-xmin, ymax-ymin)
    xmiddle, ymiddle = (xmin+xmax)/2, (ymin+ymax)/2
    xmin, xmax = xmiddle-bigger_width/2, xmiddle+bigger_width/2
    ymin, ymax = ymiddle-bigger_width/2, ymiddle+bigger_width/2
    margin = max(bigger_width/5, 250)
    axes.set_xlim((xmin-margin, xmax+margin))
    axes.set_ylim((ymin-margin, ymax+margin))
    axes.set_aspect('equal')

    if show_scale_bar == True:
      scale1 = ScaleBar(  
      dx=1, length_fraction=0.2, fixed_value=100,
          location='upper left',  # in relation to the whole plot
          label_loc='left', scale_loc='bottom'  # in relation to the line
      )
      axes.add_artist(scale1)

    axes.xaxis.set_major_locator(ticker.NullLocator())
    axes.yaxis.set_major_locator(ticker.NullLocator()) 

    if show_colorbar == True:
      cmap = cm.Blues
      cax = fig.add_axes([0.95, 0.25, 0.05, 0.52]) #[left, bottom, width, height] 
      cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap),ticks=[0, 1],
                cax=cax, orientation='vertical')
      cbar.ax.set_title('Trajectory', ha='left', y=1.07)
      cbar.ax.tick_params(size = 0)
      cbar.outline.remove()
      cbar.ax.set_yticklabels(['Least recent',  'Most recent']) 
      global Show_Colorbar 
      Show_Colorbar = False

"""### make ff_dataframe"""

def MakeFFDataframe(monkey_information, ff_catched_T_sorted, ff_flash_sorted,  ff_real_position_sorted, max_distance = 400, data_folder_name = None, num_missed_index = -1):
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

  if len(ff_catched_T_sorted) > 800:
    ff_dataframe = ff_dataframe[ff_dataframe['time'] < ff_catched_T_sorted[-200]]

  if data_folder_name:
    filepath = 'gdrive/MyDrive/fireflies_data/' + data_folder_name + '/ff_dataframe.csv'
    os.makedirs('gdrive/MyDrive/fireflies_data/' + data_folder_name, exist_ok = True)
    ff_dataframe.to_csv(filepath) 

  return ff_dataframe

"""### make ff_dataframe (agent)
(only include ffs that are in obs space)
"""

def MakeFFDataframe(monkey_information, ff_catched_T_sorted, ff_flash_sorted,  ff_real_position_sorted, max_distance = 400, data_folder_name = None, num_missed_index = -1):
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
  catched_ff_num = len(ff_catched_T_sorted) - 200
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

  filepath = 'gdrive/MyDrive/fireflies_data/' + data_folder_name + '/ff_dataframe.csv'
  os.makedirs('gdrive/MyDrive/fireflies_data/' + data_folder_name, exist_ok = True)
  ff_dataframe.to_csv(filepath)