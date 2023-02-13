import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK']='True'






def interpret_neural_network_func(sac_model,
                      sample_size = 1000, 
                      full_memory = 4, 
                      color_variable = "dv",
                      reward_boundary = 25, 
                      invisible_distance = 400, 
                      const_distance = None, 
                      const_angle = None, 
                      const_memory = None, 
                      num_default_ff = 0, 
                      norm_input = False, 
                      plot_in_xy_coord = False,
                      use_angle_to_boundary = False):
  """
  Plot actions based on observations to aid the understanding of the neural network of the agent;
  Note, currently this function can only be used for the SB3 agent because the LSTM agent needs hidden outputd to generate action


  Parameters
  ----------
  sac_model: obj
      the agent
  sample_size: num
      the number of dots to be plotted in the graph
  full_memory: num
      the value for memory when a firefly is visible; in other words, the maximum value for memory
  color_variable: str
      "dv" or "dw"; denotes whether the color signifies the linear velocity or the angular velocity
  reward_boundary: num
      the reward boundary of the firefly
  invisible_distance: num
      the distance beyond which a firefly will be invisible
  const_distance: num or None
      if num, then it denotes the distance of the ff used by all observations; otherwise, distance will be randomly sampled;
      note that one of the three variables -- distance, angle, and memory -- needs to be constant; otherwise, no plot will be made
  const_angle: num or None
      if num, it denotes the angle of the ff used by all observations; otherwise, distance will be randomly sampled
  const_memory: num or None
      if num, it denotes the memory of the ff used by all observations; otherwise, distance will be randomly sampled
  num_default_ff: num
      number of placeholder fireflies in the obseration; usually used when the number of ff in the obs space is greater than 0
  norm_input: bool
      whether input is normed
  plot_in_xy_coord: bool 
      whether to plot in the Cartesian coordinate system
  use_angle_to_boundary: bool
      whether to use the angle of the monkey to the reward boundary of the firefly instead of the center of the firefly
  """


  # Let's try if the second slot is with a placeholder

  fig, ax = plt.subplots()

  if const_distance is not None:
    # use the same firefly distance for all dots
    distances = np.ones(sample_size)*const_distance
    angle2center = np.random.uniform(low=-pi, high=pi, size=sample_size,)
    # for more information on how the angle to boundary is calculated, see the function calculate_angle_to_ff in env.py 
    angle2boundary_abs = np.clip(np.abs(angle2center)-np.abs(np.arcsin(np.divide(reward_boundary, np.clip(distances, a_min=reward_boundary)))), amin=0)
    angle2boundary = np.sign(angle2center) * angle2boundary_abs
    memories = np.random.randint(low=0, high=full_memory+1, size=[sample_size, ])
    plt.title("Angle-to-center vs Memory, colored by " + color_variable + ", with distance = ", + round(const_distance), y=1.08)
    plt.xlabel("Memory", labelpad=30) #
    plt.ylabel("Angle to center (rad)", labelpad=30)
  elif const_angle is not None:
    distances = np.random.uniform(low=0.0, high=invisible_distance, size=[sample_size, ])
    angle2center = np.ones([sample_size, ]) * const_angle
    angle2center = np.remainder(angle2center, 2*pi)
    angle2center[angle2center > pi] = angle2center[angle2center > pi] - 2*pi
    angle2boundary_abs = np.clip(np.abs(angle2center)-np.abs(np.arcsin(np.divide(reward_boundary, np.clip(distances, a_min=reward_boundary)))), amin=0)
    angle2boundary = np.sign(angle2center) * angle2boundary_abs
    memories = np.random.randint(low=0, high=full_memory, size=[sample_size, ])
    plt.title("Memory vs Distance, colored by " + color_variable + ", with angle = ", + round(const_angle, 2), y=1.08)
    plt.xlabel("Memory", labelpad=30)
    plt.ylabel("Distance (cm)", labelpad=30)
  elif const_memory:
    distances = np.random.uniform(low=0.0, high=invisible_distance, size=[sample_size,])
    angle2center = np.random.uniform(low=-pi, high=pi, size=sample_size,)
    angle2boundary_abs = np.clip(np.abs(angle2center)-np.abs(np.arcsin(np.divide(reward_boundary, np.clip(distances, a_min=reward_boundary)))), amin=0)
    angle2boundary = np.sign(angle2center) * angle2boundary_abs
    memories = np.ones(sample_size)*const_memory
    plt.title("Angle-to-center vs Distance, colored by " + color_variable + ", with memory = ", + round(const_memory), y=1.08)
    plt.xlabel("Egocentric x-coord (cm) ", labelpad=30)  
    plt.ylabel("Egocentric y-coord (cm)", labelpad=30)
      
  
  # stack the attributes together, so that each row is an observation 
  stacked_array = np.stack((angle2center, angle2boundary, distances,  memories), axis=1)

  if num_default_ff > 0:
    # add placeholders to fill up the obs space
    for i in range(len(num_default_ff)):
      placeholders = np.tile(np.array([[0], [0], [invisible_distance], [1]]), sample_size).T
      stacked_array = np.concatenate([stacked_array, placeholders], axis=1)

  if norm_input is True:
    stacked_array[0::4] = stacked_array[0::4]/pi
    stacked_array[1::4] = stacked_array[1::4]/pi
    stacked_array[2::4] = (stacked_array[2::4]/invisible_distance-0.5)*2
    stacked_array[3::4] = (stacked_array[3::4]/full_memory-0.5)*2

  # for each observation, use the network to generate the agent's action
  all_actions = np.zeros([sample_size, 2])
  for i in range(sample_size):
    obs = stacked_array[i]
    action, _ = sac_model.predict(obs, deterministic=True)
    all_actions[i] = action.copy()
  

  # plot the colorbar
  cmap = cm.viridis_r
  color_values = {"dv": all_actions[:, 1]+1, "dw": all_actions[:, 0]}
  max_value = {"dv": 200, "dw": pi}
  colorbar_title = {"dv": 'dv(cm/s)', "dw": 'dw(rad/s)'}

  norm = matplotlib.colors.Normalize(vmin=0, vmax=max_value[color_variable])
  cax = fig.add_axes([0.95, 0.4, 0.05, 0.43])
  cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
  cbar.ax.set_title(colorbar_title[color_variable], ha='left', y=1.04)
  cbar.ax.tick_params(axis='y', color='white', direction="in", right=True, length=5, width=1.5)
  cbar.outline.set_visible(False)
  
  # When plotting, which type of angles will be chosen depends on whether use_angle_to_boundary is True
  angle_for_plotting = {True: stacked_array[:, 1], False: stacked_array[:, 0]}


  # plot the dots
  if const_distance:
    ax.scatter(stacked_array[:, 3], angle_for_plotting[use_angle_to_boundary], s=2, c=color_values[color_variable], cmap=cmap)
  elif const_angle:
    ax.scatter(stacked_array[:, 3], stacked_array[:, 2], s=2, c=color_values[color_variable], cmap=cmap)
  elif const_memory:
    if plot_in_xy_coord is True:
      all_x = np.multiply(np.cos(angle2center+pi/2), distances)
      all_y = np.multiply(np.sin(angle2center+pi/2), distances)
      # only plot a dot if it's not behind the agent 
      valid_indices = np.where(all_y > 0)[0]
      ax.scatter(all_x[valid_indices], all_y[valid_indices], s=2, c=color_values[color_variable][valid_indices], cmap=cmap)
    else:
      ax.scatter(angle_for_plotting[use_angle_to_boundary], stacked_array[:, 2], s=2, c=color_values[color_variable], cmap=cmap)
      
  plt.show()