import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from numpy import linalg as LA
from contextlib import contextmanager
os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def plt_config(title=None, xlim=None, ylim=None, xlabel=None, ylabel=None, colorbar=False, sci=False):
    """
    Set some parameters for plotting
    
    """
    for field in ['title', 'xlim', 'ylim', 'xlabel', 'ylabel']:
        if eval(field) != None: getattr(plt, field)(eval(field))
    if isinstance(sci, str): plt.ticklabel_format(style='sci', axis=sci, scilimits=(0,0))
    if isinstance(colorbar,str): plt.colorbar(label=colorbar)
    elif colorbar: plt.colorbar(label = '$Number\ of\ Entries$')


@contextmanager
def initiate_plot(dimx=24, dimy=9, dpi=100, fontweight='normal'):
    """
    Set some parameters for plotting
    
    """
    plt.rcParams['figure.figsize'] = (dimx, dimy)
    plt.rcParams['font.weight'] = fontweight
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams["font.family"] = 'Arial'
    global fig; fig = plt.figure(dpi=dpi)
    yield
    plt.show()


def find_intersection(intervals, query):
    """
    Find intersections between intervals. Intervals are open and are 
    represented as pairs (lower bound, upper bound). 
    The source of the code is:
    source: https://codereview.stackexchange.com/questions/203468/
    find-the-intervals-which-have-a-non-empty-intersection-with-a-given-interval


    Parameters
    ----------
    intervals: array_like, shape=(N, 2) 
        Array of intervals.
    query: array_like, shape=(2,) 
        Interval to query

    Returns
    -------
    indices_of_overlapped_intervals: array
        Array of indexes of intervals that overlap with query
    
    """
    intervals = np.asarray(intervals)
    lower, upper = query
    indices_of_overlapped_intervals = np.where((lower < intervals[:, 1]) & (intervals[:, 0] < upper))[0]
    return indices_of_overlapped_intervals


def flash_on_ff_in_trial(ff_flash_sorted, duration):
    """
    Find the index of the fireflies that have flashed during the trial

    Parameters
    ----------
    ff_flash_sorted: list
        contains the time that each firefly flashes on and off
    duration: array_like, shape=(2,) 
        the starting time and ending time of the trial

    Returns
    -------
    flash_index: list
        the indices of the fireflies that have flashed during the trial (among all fireflies)
    
    """
    flash_index = []
    for index in range(len(ff_flash_sorted)):
      # Take out the flashing-on and flashing-off time of this particular firefly
      ff = ff_flash_sorted[index]
      if len(find_intersection(ff, duration)) > 0:
        flash_index.append(index)
    return flash_index


# Create a dictionary of {time: [indices of fireflies that are visible], ...}
def flash_on_ff_in_trial_by_time(anim_t, currentTrial, num_trials, ff_flash_sorted, ff_life_sorted, ff_catched_T_sorted):
    """
    Find the fireflies that are visible at each time point (for animation)

    Parameters
    ----------
    anim_t: array-like
        containing a list of time
    currentTrial: numeric
        the number of current trial 
    ff_flash_sorted: list
        containing the time that each firefly flashes on and off
    ff_life_sorted: np.array
        containing the time that each firefly comes into being and gets captured 
        (if the firefly is never captured, then capture time is replaced by the last point of time in data)
    ff_catched_T_sorted: np.array
        containing the time when each captured firefly gets captured

    Returns
    -------
    flash_on_ff_dict: dict
        contains the indices of the fireflies that have been flashing at each time point

    Examples
    -------
        flash_on_ff_dict = flash_on_ff_in_trial_by_time(anim_t, currentTrial, num_trials, ff_flash_sorted)
    
    """
    # Find indices of fireflies that have been alive during the trial
    alive_ff_during_this_trial = np.where((ff_life_sorted[:,1] > ff_catched_T_sorted[currentTrial-num_trials])\
                                          & (ff_life_sorted[:,0] < ff_catched_T_sorted[currentTrial]))[0]
    flash_on_ff_dict = {}
    for time in anim_t:
        # Find indicies of fireflies that have been on at this time point
        visible_ff_indices = [index for index in alive_ff_during_this_trial \
         if len(np.where(np.logical_and(ff_flash_sorted[index][:,0] <= time, \
                                           ff_flash_sorted[index][:,1] >= time))[0]) > 0]
        # Store the ff indices into the dictionary with the time being the key
        flash_on_ff_dict[time] = visible_ff_indices
    return flash_on_ff_dict



# Create a dictionary of {time: [[believed_ff_position], [believed_ff_position2], ...], ...}
def believed_ff(anim_t, currentTrial, num_trials, ff_believed_position_sorted, ff_catched_T_sorted):
    """
    Match the believed positions of the fireflies to the time when they are captured (for animation)

    Parameters
    ----------
    anim_t: array-like
        containing a list of time
    currentTrial: numeric
        the number of current trial 
    num_trials: numeric
        number of trials to span across when using this function
    ff_believed_position_sorted: np.array
        containing the locations of the monkey (or agent) when each captured firefly was captured 
    ff_catched_T_sorted: np.array
        containing the time when each captured firefly gets captured


    Returns
    -------
    believed_ff_dict: dictionary
        contains the locations of the captured fireflies that have been captured during a trial up to each time point;
        the indices are cummulative

    Examples
    -------
        believed_ff_dict = believed_ff(anim_t, currentTrial, num_trials, ff_believed_position_sorted, ff_catched_T_sorted)

    """
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




def distance_traveled(currentTrial, ff_catched_T_sorted, monkey_information):
    """
    Find the length of the trajectory run by the monkey in the current trial

    Parameters
    ----------
    currentTrial: numeric
        the number of current trial 
    ff_catched_T_sorted: np.array
        containing the time when each captured firefly gets captured
    monkey_information: dict
        containing the speed, angle, and location of the monkey at various points of time


    Returns
    -------
    distance: numeric
        the length of the trajectory run by the monkey in the current trial

    """
    duration = [ff_catched_T_sorted[currentTrial-1], ff_catched_T_sorted[currentTrial]]
    cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
    if len(cum_indices) > 5:
      cum_t = monkey_information['monkey_t'][cum_indices]
      cum_speed = monkey_information['monkey_speed'][cum_indices]
      distance = np.sum((cum_t[1:] - cum_t[:-1])*cum_speed)
    return distance



def abs_displacement(currentTrial, ff_catched_T_sorted, monkey_information, ff_believed_position_sorted):
    """
    Find the absolute displacement between the target for the currentTrial and the target for currentTrial.
    Return 9999 if the monkey has hit the border at one point.

    Parameters
    ----------
    currentTrial: numeric
        the number of current trial 
    ff_catched_T_sorted: np.array
        containing the time when each captured firefly gets captured
    monkey_information: dict
        containing the speed, angle, and location of the monkey at various points of time
    ff_believed_position_sorted: np.array
        containing the locations of the monkey (or agent) when each captured firefly was captured 


    Returns
    -------
    displacement: numeric
        the distance between the starting and ending points of the monkey during a trial; 
        returns 9999 if the monkey has hit the border at any point during the trial

    """
    duration = [ff_catched_T_sorted[currentTrial-1], ff_catched_T_sorted[currentTrial]]
    cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
    if len(cum_indices) > 5:
      cum_mx, cum_my = monkey_information['monkey_x'][cum_indices], monkey_information['monkey_y'][cum_indices] 
      # If the monkey has hit the boundary
      if np.any(cum_mx[1:]-cum_mx[:-1] > 10) or np.any(cum_my[1:]-cum_my[:-1] > 10):
        displacement = 9999
      else:
        displacement = LA.norm(ff_believed_position_sorted[currentTrial]-ff_believed_position_sorted[currentTrial-1])
    return displacement



def find_stops(currentTrial, ff_catched_T_sorted, monkey_information, player = "monkey", return_index = False, 
                  since_target_last_seen = False, t_last_visible = None):
    """
    Find the locations or indices of the stops that the monkey made during a trial (between currentTrial -1 and currentTrial)

    Parameters
    ----------
    currentTrial: numeric
        the number of current trial 
    ff_catched_T_sorted: np.array
        containing the time when each captured firefly gets captured
    monkey_information: dict
        containing the speed, angle, and location of the monkey at various points of time
    player: str
        "monkey" or "agent" 
    return_index: bool
        whether to return point indices of the stops relative to all data
    since_target_last_seen: bool
        whether to only include the stops after the target is last seen
    t_last_visible: array-like
        containing the time that elapses between the target last being visible and its capture for all captured fireflies

    Returns
    -------
    distinct_stops: list
        a list containing the locations of distinct stops (if return_index is True, then indices instead of locations are returned)

    """
    if since_target_last_seen is True:
      duration = [ff_catched_T_sorted[currentTrial]-t_last_visible[currentTrial-1], ff_catched_T_sorted[currentTrial]]
    else:
      duration = [ff_catched_T_sorted[currentTrial-1], ff_catched_T_sorted[currentTrial]]
    
    cum_indices = np.where((monkey_information['monkey_t'] > duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
    if len(cum_indices) > 5:
      cum_mx, cum_my = monkey_information['monkey_x'][cum_indices], monkey_information['monkey_y'][cum_indices] 
      cum_speeddummy = monkey_information['monkey_speeddummy'][cum_indices]
      # if the monkey has stopped at any point
      zerospeed_index = np.where(cum_speeddummy==0)[0]
      if len(zerospeed_index) > 0 :
        zerospeedx, zerospeedy = cum_mx[zerospeed_index], cum_my[zerospeed_index]
        zerospeedindex = cum_indices[zerospeed_index]
        # Get pairs of x, y coordinates of all the stops
        stop0 = np.array(list(zip(zerospeedx, zerospeedy)))
        # Get the locations indexes of unique stops
        _, stops_index = np.unique(stop0, axis=0, return_index=True)
        stops = stop0[stops_index[np.argsort(stops_index)]]
        stop_indices = zerospeedindex[stops_index[np.argsort(stops_index)]]
        # If player is monkey;
        if player == "monkey":
        # Find distinct stops from unique stops: two stops are considered distinct here if they are at least 0.6 cm apart; 
        # this number is an arbitrary but sensible choice.
          if return_index is True:
            distinct_stops = [stop_indices[0]] + [stop_indices[i+1] for i in range(len(stops)-1) if LA.norm(np.array((stops[i+1][0]-stops[i][0], stops[i+1][1]-stops[i][1]))) > 0.6]
          else:
            distinct_stops = [stops[0]] + [stops[i+1] for i in range(len(stops)-1) if LA.norm(np.array((stops[i+1][0]-stops[i][0], stops[i+1][1]-stops[i][1]))) > 0.6]
        else: # else the player is agent
          if return_index is True:
            distinct_stops = zerospeedindex[stops_index[np.argsort(stops_index)]]
          else:
            distinct_stops = stops
      else:
        # if there is no stop, then return an empty list
        distinct_stops = []
    else:
      # if there is no step in the current trial (this can happen if the monkey captures two fireflies together), then return an empty list
      distinct_stops = []
    return distinct_stops




def put_stops_into_clusters(currentTrial, distance_between_points, ff_catched_T_sorted, monkey_information, player = "monkey"):
    """
    Assign each stop with a number that indicates which cluster it belongs to

    Parameters
    ----------
    currentTrial: numeric
        the number of current trial 
    distance_between_points: numeric
        the maximum distance between two points for the two points to be considered as belonging to the same cluster
    ff_catched_T_sorted: np.array
        containing the time when each captured firefly gets captured
    monkey_information: dict
        containing the speed, angle, and location of the monkey at various points of time
    player: str
        "monkey" or "agent" 
    
    Returns
    -------
    clusters: list
        a list that assigns each stop with a number that indicates which cluster it belongs to

    """

    # First find locations of distinct stops made by the monkey/agent during the trial
    distinct_stops = find_stops(currentTrial, ff_catched_T_sorted, monkey_information, player = player)
    if len(distinct_stops) == 0: 
      clusters = []
    else:
      # change the format of distinct_stops into np.array
      distinct_stops2 = np.array([[stop[0], stop[1]] for stop in distinct_stops])
      # Let the number of cluster begin with 1
      current_cluster = 1
      # The first stop is naturally assigned number 1
      clusters = [1]
      for i in range(1, len(distinct_stops2)):
        if LA.norm(distinct_stops2[i]-distinct_stops2[i-1]) < distance_between_points:
          clusters.append(current_cluster)
        else: # Create a new cluster
          current_cluster = current_cluster+1
          clusters.append(current_cluster)
    return clusters

