import os
import seaborn as sns
import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap
from numpy import linalg as LA
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def PlotTrials(duration, 
               monkey_information,
               ff_dataframe, 
               ff_life_sorted, 
               ff_real_position_sorted, 
               ff_believed_position_sorted, 
               ffs_around_target_indices, 
               ff_catched_T_sorted = None,
               currentTrial = None, # Can be None; then it means all trials in the duration will be plotted
               num_trials = None,
               fig = None, 
               axes = None, 
               rotation_matrix = None,
               player="monkey",
               trail_color="orange",  # "viridis" or None (default is "orange") or other colors
               visible_distance = 250,
               show_start=True,
               show_stops=False,
               show_believed_target_positions=False,
               show_reward_boundary=False,
               show_path_when_target_visible=False,  
               show_path_when_prev_target_visible=False,
               show_connect_path_ff=False, 
               show_path_when_cluster_visible=False,
               show_scale_bar=False,
               show_colorbar=False,
               trial_to_show_cluster=None,  # None, "current", or "previous"
               cluster_dataframe_point=None, 
               trial_to_show_cluster_around_target=None, # None, "current", or "previous"
               steps_to_be_marked=None,
               zoom_in = False,
               images_dir = None,
               hitting_boundary_ok = False,
               trial_too_short_ok = False, 
               subplots = False,
               return_whether_plotted= False,
               *args, 
               **kwargs
               ):

    """
    Visualize a trial or a few consecutive trials


    Parameters
    ----------
    duration: list
        the duration to be plotted, with a starting time and an ending time; 
    ff_dataframe: pd.dataframe
        containing various information about all visible or "in-memory" fireflies at each time point
    monkey_information: dict
        containing the speed, angle, and location of the monkey at various points of time
    ff_life_sorted: np.array
        containing the time that each firefly comes into being and gets captured 
        (if the firefly is never captured, then capture time is replaced by the last point of time in data)
    ff_real_position_sorted: np.array
        containing the real locations of the fireflies
    ff_believed_position_sorted: np.array
        containing the locations of the monkey (or agent) when each captured firefly was captured 
    ffs_around_target_indices: list
        for each trial, it contains the indices of fireflies around the target; 
        it contains an empty array when there is no firefly around the target
    fig: object
        the canvas of the plot
    axes: object
        The axes of the plot
    rotation_matrix: array
        The matrix by which the plot will be rotated
    currentTrial: numeric
        the number of current trial
    num_trials: numeric
        the number of trials to be plotted
    player: str
        "monkey" or "agent"
    trail_color: str
        the color of the trajectory of the monkey/agent; can be "viridis" or None (then "orange" will be used) or other colors
    visible_distance: num
        the distance beyond which a firefly will not be considered visible; default is 250
    show_start: bool
        whether to show the starting point of the monkey/agent
    show_stop: bool
        whether to show the stopping point of the monkey/agent
    show_believed_target_positions: bool
        whether to show the believed positions of the targets
    show_reward_boundary: bool
        whether to show the reward boundaries of fireflies
    show_path_when_target_visible: bool
        whether to mark the part of the trajectory where the target is visible  
    show_path_when_prev_target_visible: bool
        whether to mark the part of the trajectory where the previous target is visible  
    show_connect_path_ff: bool
        whether to draw lines between the trajectory and fireflies to indicate the part of the trajectory where a firefly is visible
    show_path_when_cluster_visible: bool
        whether to mark the part of the trajectory where any firefly in the cluster centered around the target is visible  
    show_scale_bar: bool
        whether to show the scale bar
    show_colorbar: bool
        whether to show the color bars
    trial_to_show_cluster: can be None, "current", or "previous"
        the trial for which to show clusters of fireflies 
    cluster_dataframe_point: dataframe
        information of the clusters for each time point that has at least one cluster; must not be None if trial_to_show_cluster is not None
    trial_to_show_cluster_around_target: can be None, "current", or "previous"
        the trial for which to show the cluster of fireflies centered around the target
    zoom_in: bool
        whether to zoom in on the plot
    images_dir: str or None
        directory of the file to store the images
    hitting_boundary_ok: bool
        whether to continue to plot the trial if the boundary is hit at any point
    trial_too_short_ok: bool
        whether to continue to plot the trial if the trial is very short (fewer than 5 time points)
    subplots: bool
        whether subplots are used
    return_whether_plotted: bool
        if True, then whether the plot is successfully made will be returned; the plot might fail to be made because the
        action sequence is too short (if trial_too_short_ok is False) or the monkey has hit the boundary at any point (if 
        hitting_boundary_ok is false)

    """

    # If currentTrial is not given, then it will be calculated based on the duration
    no_currentTrial_input = False
    if currentTrial is None:
        no_currentTrial_input = True
        captured_ff_within_duration = np.where((ff_catched_T_sorted >= duration[0]) & (ff_catched_T_sorted <= duration[1]))[0]
        if len(captured_ff_within_duration) > 1:
            currentTrial = captured_ff_within_duration[-1]
            num_trials = currentTrial - captured_ff_within_duration[0]
        else:
            currentTrial = 0
            num_trials = 1

    ff_position_during_this_trial = np.array([ff_real_position_sorted[ff_index] for ff_index, life in \
                                    enumerate(ff_life_sorted) if (life[-1] >= duration[0]) and (life[0] < duration[1])])
    target_indices = np.arange(currentTrial-num_trials+1, currentTrial+1)
    cum_indices = np.where((monkey_information['monkey_t'] > duration[0]) & (monkey_information['monkey_t'] <= duration[1]))[0]
    cum_t = monkey_information['monkey_t'][cum_indices]
    cum_mx, cum_my = monkey_information['monkey_x'][cum_indices], monkey_information['monkey_y'][cum_indices]
    cum_speed, cum_speeddummy = monkey_information['monkey_speed'][cum_indices], monkey_information['monkey_speeddummy'][cum_indices]
    
    
    if not hitting_boundary_ok:
        # Stop plotting for the trial if the monkey/agent has gone across the edge
        cum_r = LA.norm(np.stack((cum_mx, cum_my)), axis = 0)
        if (np.any(cum_r > 949)):
            if return_whether_plotted:
                return False
            else:
                return
    if not trial_too_short_ok:
        # Stop plotting for the trial if the trial is too short
        if (len(cum_t) < 5):
            if return_whether_plotted:
                return False
            else:
                return

    if fig is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)


    if rotation_matrix is None:
        # Find the angle from the starting point to the target
        theta = pi/2-np.arctan2(cum_my[-1]-cum_my[0], cum_mx[-1]-cum_mx[0])     
        c, s = np.cos(theta), np.sin(theta)
        # Rotation matrix
        R = np.array(((c, -s), (s, c)))
    else:
        R = rotation_matrix


    # Plot the trajectory of the monkey
    cum_mxy_rotate = np.matmul(R, np.stack((cum_mx, cum_my)))
    trail_size = {"agent": 70, "monkey": 10}
    if subplots == True:
        trail_size = {"agent": 10, "monkey": 10}
    if trail_color == "viridis": # the color of the path will vary by speed
        axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1], marker='o', s=trail_size[player], c=cum_speed, zorder=2)
    elif trail_color is None:
        axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1], marker='o', s=trail_size[player], color="orange", zorder=2)
    else:
        axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1], marker='o', s=trail_size[player], color=trail_color, zorder=2)


    if show_start:
        # Plot the start
        start_size = {"agent": 220, "monkey": 100}
        axes.scatter(cum_mxy_rotate[0, 0], cum_mxy_rotate[1, 0], marker='^', s=start_size[player], color="gold", zorder=3, alpha=0.7)

    if show_stops:
        stop_size = {"agent": 160, "monkey": 150}
        zerospeed_index = np.where(cum_speeddummy == 0)
        zerospeedx, zerospeedy = cum_mx[zerospeed_index], cum_my[zerospeed_index]
        zerospeed_rotate = np.matmul(R, np.stack((zerospeedx, zerospeedy)))
        axes.scatter(zerospeed_rotate[0], zerospeed_rotate[1], marker='*', s=stop_size[player], alpha=0.7, color="black",zorder=2)

    if steps_to_be_marked is not None:
        axes.scatter(cum_mxy_rotate[0, steps_to_be_marked], cum_mxy_rotate[1, steps_to_be_marked],marker = 'o',s = 120, color="darkgreen", zorder=3, alpha=0.5)


    ff_position_rotate = np.matmul(R, np.stack((ff_position_during_this_trial.T[0], ff_position_during_this_trial.T[1])))
    axes.scatter(ff_position_rotate[0], ff_position_rotate[1], marker='o', s=10, color="magenta", zorder=2)

    if show_believed_target_positions:
        target_size = {"agent": 185, "monkey": 120}
        if currentTrial is not None:
            ff_believed_position = ff_believed_position_sorted[currentTrial - num_trials + 1:currentTrial + 1]
        else:
            ff_believed_position = ff_believed_position
        ff_believed_position_rotate = np.matmul(R, np.stack((ff_believed_position.T[0], ff_believed_position.T[1])))
        axes.scatter(ff_believed_position_rotate[0], ff_believed_position_rotate[1], marker='*', s=target_size[player], color="red", alpha=0.75, zorder=2)

    if show_reward_boundary:
        for i in ff_position_rotate.T:
            circle = plt.Circle((i[0], i[1]), 25, facecolor='grey', edgecolor='orange', alpha=0.45, zorder=1)
            axes.add_patch(circle)

    if show_path_when_target_visible:
        path_size = {"agent": 50, "monkey": 30}
        temp_df = ff_dataframe.loc[ff_dataframe['target_index'].isin(target_indices)]
        temp_df = temp_df.loc[(temp_df['ff_index'] == currentTrial) & (temp_df['visible'] == 1) & (temp_df['ff_distance'] <= visible_distance)]
        monkey_xy_when_ff_visible = np.array(temp_df[['monkey_x', 'monkey_y']])
        ff_visible_path_rotate = np.matmul(R, monkey_xy_when_ff_visible.T)
        axes.scatter(ff_visible_path_rotate[0], ff_visible_path_rotate[1], s=path_size[player], c="green", alpha=0.8, zorder=5)

    if show_path_when_prev_target_visible: # for previous target
        path_size = {"agent": 65, "monkey": 40}
        temp_df = ff_dataframe.loc[ff_dataframe['target_index'].isin(target_indices)]
        temp_df = temp_df.loc[(temp_df['ff_index'] == currentTrial - 1) & (temp_df['visible'] == 1) & (temp_df['ff_distance'] <= visible_distance)]
        monkey_xy_when_ff_visible = np.array(temp_df[['monkey_x', 'monkey_y']])
        ff_visible_path_rotate = np.matmul(R, monkey_xy_when_ff_visible.T)
        axes.scatter(ff_visible_path_rotate[0], ff_visible_path_rotate[1], s=path_size[player], c="aqua", alpha=0.8, zorder=3)

    if show_connect_path_ff:
        connection_linewidth = {"agent": 1.5, "monkey": 1}
        connection_alpha = {"agent": 0.3, "monkey": 0.2}
        temp_df = ff_dataframe.loc[(ff_dataframe['visible']==1) & (ff_dataframe['ff_distance'] <= visible_distance)]
        if no_currentTrial_input == False:
            temp_df = temp_df.loc[temp_df['target_index'].isin(target_indices)]
        else:
            temp_df = temp_df.loc[temp_df['time'] <= duration[1]]
        if player == "monkey":
            # if the player is monkey, then the following code is used to avoid the lines between the monkey's position and the target since the lines might obscure the path
            temp_df = temp_df.loc[~temp_df['ff_index'].isin(target_indices)]
        temp_array = temp_df[['ff_x', 'ff_y', 'monkey_x', 'monkey_y']].to_numpy()
        temp_ff_positions = np.matmul(R, temp_array[:, :2].T)
        temp_monkey_positions = np.matmul(R, temp_array[:, 2:].T)
        for j in range(len(temp_array)):
            axes.plot(np.stack([temp_ff_positions[0, j], temp_monkey_positions[0, j]]),
                      np.stack([temp_ff_positions[1, j], temp_monkey_positions[1, j]]), 
                      '-', alpha=connection_alpha[player], linewidth=connection_linewidth[player], c="#a940f5")
            # uncomment the line below to mark the connected fireflies as brown circles
            # axes.plot(temp_ff_positions[0,j], temp_ff_positions[1,j], '-', alpha=0.2, marker="o", markersize=5, color="brown")


    if trial_to_show_cluster is not None:
        trial_conversion = {"current": 0, "previous": -1}
        # Find the indices of ffs in the cluster
        cluster_indices = cluster_dataframe_point[cluster_dataframe_point['target_index'] == currentTrial + trial_conversion[trial_to_show_cluster]].ff_index
        cluster_indices = np.unique(cluster_indices.to_numpy())
        cluster_ff_positions = ff_real_position_sorted[cluster_indices]
        cluster_ff_rotate = np.matmul(R, np.stack((cluster_ff_positions.T[0], cluster_ff_positions.T[1])))
        axes.scatter(cluster_ff_rotate[0], cluster_ff_rotate[1], marker='o', c="blue", s=25, zorder=4)


    if trial_to_show_cluster_around_target is not None:
        trial_conversion = {"current": 0, "previous": -1}
        cluster_ff_indices = ffs_around_target_indices[currentTrial + trial_conversion[trial_to_show_cluster_around_target]]
        cluster_ff_positions = ff_real_position_sorted[cluster_ff_indices]
        if len(cluster_ff_positions) > 0:
            ffs_around_target_rotate = np.matmul(R, np.stack((cluster_ff_positions.T[0], cluster_ff_positions.T[1])))
            axes.scatter(ffs_around_target_rotate[0], ffs_around_target_rotate[1], marker='o', s=30, color="blue", zorder=4)
        if show_path_when_cluster_visible: # Find where on the path the monkey/agent can see any member of the cluster around the target
            list_of_colors = ["navy", "magenta", "white", "gray", "brown", "black"]
            path_size = {"agent": [80, 10], "monkey": [15, 3]}
            path_alpha = {"agent": 0.8, "monkey": 0.4}
            ff_size = {"agent": 140, "monkey": 100}
            ff_alpha = {"agent": 0.8, "monkey": 0.5}
            for index in cluster_ff_indices:
                temp_df = ff_dataframe.loc[ff_dataframe['target_index'].isin(target_indices)]
                temp_df = temp_df.loc[(temp_df['ff_index'] == index) & (temp_df['visible'] == 1)]
                monkey_xy_when_ff_visible = np.array(temp_df[['monkey_x', 'monkey_y']])
                monkey_xy_rotate = np.matmul(R, monkey_xy_when_ff_visible.T)
                axes.scatter(monkey_xy_rotate[0], monkey_xy_rotate[1], s=path_size[player][0] - path_size[player][1] * i, color=list_of_colors[i], alpha=path_alpha[player], zorder=3+i)
                # Use a circle with the corresponding color to show that ff
                ff_position = np.array(temp_df[['ff_x', 'ff_y']])
                ff_position_rotate = np.matmul(R, ff_position.T)
                axes.scatter(ff_position_rotate[0], ff_position_rotate[1], marker='o', s=ff_size[player], alpha=ff_alpha[player], color=list_of_colors[i], zorder=3)


    if show_scale_bar:
        scale = ScaleBar(dx=1, length_fraction=0.2, fixed_value=100, location='upper left', label_loc='left', scale_loc='bottom') 
        axes.add_artist(scale)
        axes.xaxis.set_major_locator(mtick.NullLocator())
        axes.yaxis.set_major_locator(mtick.NullLocator())


    if show_colorbar:
        # Make the black and red colorbar to show whether a stopping point is rewarded or not
        new_colormap = ListedColormap(["black", "red"])
        labels = np.array(["No Reward", "Reward"])
        # Make normalizer and formatter
        norm = matplotlib.colors.BoundaryNorm(np.array([0.5, 1.5, 2.5]), 2, clip=True)
        formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
        # Plot our figure
        im = axes.imshow(np.array([[1, 2]]), cmap=new_colormap, extent=[0, 0, 0, 0], norm=norm)
        cbar = fig.colorbar(im, format=formatter, ticks=np.array([1, 2.]), cax=fig.add_axes([0.95, 0.15, 0.05, 0.2]))
        cbar.ax.tick_params(width=0)
        cbar.ax.set_title('Stopping Points', ha='left')

        # Then make the colorbar to show the meaning of color of the monkey/agent's path
        if trail_color == "orange":
            new_colormap = ListedColormap(["green", "orange"])
            labels = np.array(["Top Target Visible", "Top Target Not Visible"])
            formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
            # Plot our figure
            im = axes.imshow(np.array([[1, 2]]), cmap=new_colormap, extent=[0, 0, 0, 0], norm=norm)
            cbar2 = fig.colorbar(im, format=formatter, ticks=np.array([1., 2.]), cax=fig.add_axes([0.95, 0.5, 0.05, 0.2]))
            cbar2.ax.tick_params(width=0)
            cbar2.ax.set_title('Path Colors', ha='left', y=1.04)
        elif trail_color == "viridis":
            cmap = cm.viridis
            # Need a new normalizer
            norm = matplotlib.colors.Normalize(vmin=0, vmax=200)
            cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([0.95, 0.4, 0.05, 0.43]), orientation='vertical')
            cbar2.outline.remove()
            cbar2.ax.tick_params(axis='y', color='white', direction="in", right=True, length=5, width=1.5)
            cbar2.ax.set_title('Speed(cm/s)', ha='left', y=1.04)


    # Set the limits of the x-axis and y-axis
    mx_min, mx_max = np.min(cum_mxy_rotate[0]), np.max(cum_mxy_rotate[0])
    my_min, my_max = np.min(cum_mxy_rotate[1]), np.max(cum_mxy_rotate[1])
    
    bigger_width = max(mx_max - mx_min, my_max - my_min)
    margin = max(bigger_width/5, 150)
    xmiddle, ymiddle = (mx_min + mx_max)/ 2, (my_min + my_max) / 2
    xmin, xmax = xmiddle - bigger_width/2, xmiddle + bigger_width/2
    ymin, ymax = ymiddle - bigger_width/2, ymiddle + bigger_width/2

    if zoom_in is True:
        axes.set_xlim((xmin - 40, xmax + 40))
        axes.set_ylim((ymin - 20, ymax + 60))
    else:
        axes.set_xlim((xmin - margin, xmax + margin))
        axes.set_ylim((ymin - margin, ymax + margin))
    axes.set_aspect('equal')
    axes.set_title(f"Trial {currentTrial}", fontsize = 22)

    if images_dir is not None:
        filename = "trial_" + str(currentTrial)
        CHECK_FOLDER = os.path.isdir(images_dir)
        if not CHECK_FOLDER:
            os.makedirs(images_dir)
        plt.savefig(f"{images_dir}/{filename}.png")

    if return_whether_plotted:
        return True



def PlotPoints(point, 
               duration_of_trajectory, 
               monkey_information, 
               ff_dataframe, 
               ff_catched_T_sorted,
               ff_life_sorted, 
               ff_real_position_sorted, 
               ff_believed_position_sorted, 
               ff_flash_sorted, 
               fig = None, 
               axes = None,
               visible_distance = 250,
               show_all_ff = True,
               show_flash_on_ff = False,
               show_visible_ff = True,
               show_in_memory_ff = True, 
               show_target = False,
               show_reward_boundary = True,
               show_legend = True,
               show_scale_bar = True,
               show_colorbar = True, 
               hitting_boundary_ok = False,
               trial_too_short_ok = False, 
               images_dir = None,
               **kwargs):

    """
    Visualize a time point in the game
    Note: As of now, this function is only used for monkey. This function also does not utilize rotation.


    Parameters
    ----------
    point: num
        the index of the point to visualize
    duration_of_trajectory: list
        the duration of the trajectory to be plotted, with a starting time and an ending time; 
    ff_dataframe: pd.dataframe
        containing various information about all visible or "in-memory" fireflies at each time point
    monkey_information: dict
        containing the speed, angle, and location of the monkey at various points of time
    ff_catched_T_sorted: np.array
        containing the time when each captured firefly gets captured
    ff_life_sorted: np.array
        containing the time that each firefly comes into being and gets captured 
        (if the firefly is never captured, then capture time is replaced by the last point of time in data)
    ff_real_position_sorted: np.array
        containing the real locations of the fireflies
    ff_believed_position_sorted: np.array
        containing the locations of the monkey (or agent) when each captured firefly was captured 
    ff_flash_sorted: list
        containing the time that each firefly flashes on and off
    fig: object
        the canvas of the plot
    axes: object
        The axes of the plot
    visible_distance: num
        the distance beyond which a firefly will not be considered visible; default is 250
    show_all_ff: bool
        whether to show all the fireflies that are alive at that point as grey
    show_flash_on_ff: bool
        whether to show all the fireflies that are flashing on at that point as red
    show_visible_ff: bool
        whether to show all the fireflies visible at that point as orange
    show_in_memory_ff: bool
        whether to show all the fireflies in memory at that point as orange
    show_target: bool
        whether to show the target using star shape
    show_reward_boundary: bool
        whether to show the reward boundaries of fireflies
    show_legend: bool
        whether to show a legend
    show_scale_bar: bool
        whether to show the scale bar
    show_colorbar: bool
        whether to show the color bar
    hitting_boundary_ok: bool
        whether to continue to plot the trial if the boundary is hit at any point
    trial_too_short_ok: bool
        whether to continue to plot the trial if the trial is very short (fewer than 5 time points)
    images_dir: str or None
        directory of the file to store the images


    """

    time = np.array(monkey_information['monkey_t'])[point]
    duration = [time - duration_of_trajectory, time]
    cum_indices = np.where((monkey_information['monkey_t'] >= duration[0]) & (monkey_information['monkey_t'] <= duration[1]))
    cum_t, cum_mx, cum_my = monkey_information['monkey_t'][cum_indices], monkey_information['monkey_x'][cum_indices], monkey_information['monkey_y'][cum_indices]
    
    if not hitting_boundary_ok:
        # Stop plotting for the trial if the monkey/agent has gone across the edge
        cum_r = LA.norm(np.stack((cum_mx, cum_my)), axis = 0)
        if (np.any(cum_r > 949)):
            return
    if not trial_too_short_ok:
        # Stop plotting for the trial if the trial is too short
        if (len(cum_t) < 5):
            return

    if fig is None:
        fig, axes = plt.subplots()

    alive_ff_indices = np.array([ff_index for ff_index, life_duration in enumerate(ff_life_sorted) 
                                if (life_duration[-1] >= time) and (life_duration[0] < time)])
    alive_ff_positions = ff_real_position_sorted[alive_ff_indices]
    
    if show_all_ff:
        axes.scatter(alive_ff_positions.T[0], alive_ff_positions.T[1], color="grey", s=30)

    if show_flash_on_ff:
        # Initialize a list to store the indices of the ffs that are flashing-on at this point
        flashing_ff_indices = []  
        # For each firefly in ff_flash_sorted:
        for ff_index, ff_flash_intervals in enumerate(ff_flash_sorted):
            # If the firefly has flashed during that trial:
            if ff_index in alive_ff_indices:
                # Let's see if the firefly has flashed at that exact moment
                for interval in ff_flash_intervals:
                    if interval[0] <= time <= interval[1]:
                        flashing_ff_indices.append(ff_index)
                        break
        flashing_ff_indices = np.array(flashing_ff_indices)
        flashing_ff_positions = ff_real_position_sorted[flashing_ff_indices]
        axes.scatter(flashing_ff_positions.T[0], flashing_ff_positions.T[1], color="red", s=120, marker='*', alpha=0.7)

    if show_visible_ff:
        visible_ffs = ff_dataframe[(ff_dataframe['point_index'] == point) & (ff_dataframe['visible'] == 1) &
                                   (ff_dataframe['ff_distance'] <= visible_distance)]
        axes.scatter(visible_ffs['ff_x'], visible_ffs['ff_y'], color="orange", s=40)

    if show_in_memory_ff:
        in_memory_ffs = ff_dataframe[(ff_dataframe['point_index'] == point) & (ff_dataframe['visible'] == 0)]
        axes.scatter(in_memory_ffs['ff_x'], in_memory_ffs['ff_y'], color="green", s=40)

    if show_target:
        trial_num  = np.digitize(time, ff_catched_T_sorted)
        if trial_num is None:
            raise ValueError("If show_target, then trial_num cannot be None")
        target_position = ff_real_position_sorted[trial_num]
        axes.scatter(target_position[0], target_position[1], marker='*', s=200, color="grey", alpha=0.35)

    if show_legend is True:
        # Need to consider what elements are used in the plot
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


    if show_reward_boundary:
        if show_all_ff: 
            for position in alive_ff_positions:
                circle = plt.Circle((position[0], position[1]), 20, facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
                axes.add_patch(circle)
        if show_visible_ff:
            for index, row in visible_ffs.iterrows():
                circle = plt.Circle((row['ff_x'], row['ff_y']), 20, facecolor='yellow', edgecolor='orange', alpha=0.25, zorder=1)
                axes.add_patch(circle)
            if show_in_memory_ff:
                for index, row in in_memory_ffs.iterrows():
                    circle = plt.Circle((row['ff_x'], row['ff_y']), 20, facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
                    axes.add_patch(circle)
        elif show_flash_on_ff:
            for index, row in flashing_ff_positions.iterrows():
                circle = plt.Circle((row['ff_x'], row['ff_y']), 20, facecolor='red', edgecolor='orange', alpha=0.25, zorder=1)
                axes.add_patch(circle)
            for ff in flashing_ff_positions:
                circle = plt.Circle((ff[0], ff[1]), 20, facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
                axes.add_patch(circle)
            if show_in_memory_ff:
                for index, row in in_memory_ffs.iterrows():
                    circle = plt.Circle((row['ff_x'], row['ff_y']), 20, facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
                    axes.add_patch(circle)

    # Also plot the trajectory of the monkey/agent
    axes.scatter(cum_mx, cum_my, s=15, c=cum_indices, cmap="Blues")

    # Set the limits of the x-axis and y-axis
    mx_min, mx_max = np.min(cum_mx[0]), np.max(cum_mx[0])
    my_min, my_max = np.min(cum_my[1]), np.max(cum_my[1])
    bigger_width = max(mx_max - mx_min, my_max - my_min)
    margin = max(bigger_width/5, 250)
    xmiddle, ymiddle = (mx_min + mx_max)/2, (my_min + my_max) /2
    xmin, xmax = xmiddle - bigger_width/2, xmiddle + bigger_width/2
    ymin, ymax = ymiddle - bigger_width/2, ymiddle + bigger_width/2
    axes.set_xlim((xmin - margin, xmax + margin))
    axes.set_ylim((ymin - margin, ymax + margin))
    axes.set_aspect('equal')


    if show_scale_bar == True:
        scale1 = ScaleBar(dx=1, length_fraction=0.2, fixed_value=100, location='upper left', label_loc='left', scale_loc='bottom')
        axes.add_artist(scale1)

    if show_colorbar == True:
        cmap = cm.Blues
        cax = fig.add_axes([0.95, 0.25, 0.05, 0.52])  # [left, bottom, width, height]
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ticks=[0, 1], cax=cax, orientation='vertical')
        cbar.ax.set_title('Trajectory', ha='left', y=1.07)
        cbar.ax.tick_params(size=0)
        cbar.outline.remove()
        cbar.ax.set_yticklabels(['Least recent', 'Most recent'])

    axes.xaxis.set_major_locator(mtick.NullLocator())
    axes.yaxis.set_major_locator(mtick.NullLocator())


    if images_dir is not None:
      filename = "time_point_" + str(point)
      CHECK_FOLDER = os.path.isdir(images_dir)
      if not CHECK_FOLDER:
          os.makedirs(images_dir)
      plt.savefig(f"{images_dir}/{filename}.png")





def histogram_per_attribute_from_trial_char(valid_trials_agent, valid_trials_monkey):
    """
    For each attribute in valid_trials (which comes from trial_char), plot a histogram that compares the monkey and the agent


    Parameters
    ----------
    valid_trials_agent: dataframe
        belonging to the agent, containing various characteristics of each trial
    valid_trials_monkey: dataframe
        belonging to the monkey, containing various characteristics of each trial

    """

    variable_of_interest = "t_last_visible"
    fig, axes = plt.subplots(figsize=(4, 5))
    sns.histplot(data = valid_trials_agent[variable_of_interest], kde = False, alpha = 0.4, color = "green", binwidth = 0.25)
    sns.histplot(data = valid_trials_monkey[variable_of_interest], kde = False, alpha = 0.4, color = "blue", binwidth = 0.22)
    axes.legend(labels=["Agent(LSTM)", "Monkey"], fontsize = 14)
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
    sns.histplot(data = valid_trials_agent[variable_of_interest], kde = False, alpha = 0.3, color = "green", binwidth=0.5, binrange=(-0.25,5.25), stat="probability", edgecolor='grey')
    sns.histplot(data = valid_trials_monkey[variable_of_interest], kde = False, alpha = 0.3, color = "blue", binwidth=0.5, binrange=(-0.1,5.4), stat="probability", edgecolor='grey')
    axes.legend(labels=["Agent(LSTM)","Monkey"], fontsize = 14)
    # sns.kdeplot(data = valid_trials_agent, x = variable_of_interest, bw=1, color = "green")
    # sns.kdeplot(data = valid_trials_monkey, x = variable_of_interest, bw=1, color = "blue")
    axes.set_ylabel("Probability",fontsize=15)
    #axes.set_yticklabels("")
    plt.xlim(-0.25,5)
    plt.title("Number of Stops Near Targets", fontsize=17)
    plt.xlabel("Number of Stops", fontsize=15)
    axes.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    plt.show()
    plt.close()


    variable_of_interest = "n_ff_in_a_row"
    fig, axes = plt.subplots(figsize=(4, 5))
    sns.histplot(data = valid_trials_agent[variable_of_interest], kde = False, alpha = 0.3, binrange=(-0.25,5.25), color = "green", binwidth=0.5, stat="probability",  edgecolor='grey')
    sns.histplot(data = valid_trials_monkey[variable_of_interest], kde = False, alpha = 0.3, binrange=(-0.1,5.4), color = "blue", binwidth=0.5, stat="probability",  edgecolor='grey')
    axes.legend(labels=["Agent(LSTM)","Monkey"], fontsize = 14)
    # sns.kdeplot(data = valid_trials_agent, x = variable_of_interest, bw=1, color = "green")
    # sns.kdeplot(data = valid_trials_monkey, x = variable_of_interest, bw=1, color = "blue")
    axes.set_ylabel("Probability",fontsize=15)
    #axes.set_yticklabels("")
    plt.xlim(0.25,5.25)
    plt.title("Number of fireflies caught in a cluster", fontsize=17)
    plt.xlabel("Number of Fireflies", fontsize=15)
    axes.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    plt.show()
    plt.close()


    variable_of_interest = "d_last_visible"
    fig, axes = plt.subplots(figsize=(4, 5))
    sns.histplot(data = valid_trials_agent[variable_of_interest]/100, kde = False, alpha = 0.3,  color = "green", binwidth=40, stat="probability",  edgecolor='grey')
    sns.histplot(data = valid_trials_monkey[variable_of_interest]/100, kde = False, alpha = 0.3,  color = "blue", binwidth=30, stat="probability",  edgecolor='grey')
    axes.legend(labels=["Agent(LSTM)","Monkey"], fontsize = 14)
    # sns.kdeplot(data = valid_trials_agent, x = variable_of_interest, color = "green", bw=1)
    # sns.kdeplot(data = valid_trials_monkey, x = variable_of_interest, color = "blue", bw=1)
    axes.set_ylabel("Probability",fontsize=15)
    #axes.set_yticklabels("")
    plt.xlim(0, 400)
    plt.title("Distance of Target Last Visible", fontsize=17)
    plt.xlabel("Distance (100 cm)", fontsize=15)
    axes.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    axes.tick_params(axis = "x", width=0)
    xticklabels=axes.get_xticks().tolist()
    xticklabels = [str(int(label)) for label in xticklabels]
    xticklabels[-1]='400+'
    axes.set_xticklabels(xticklabels)
    plt.show()
    plt.close()


    variable_of_interest = "abs_angle_last_visible"
    fig, axes = plt.subplots(figsize=(4, 5))
    sns.histplot(data = valid_trials_agent[variable_of_interest], kde = False, binwidth=0.04, alpha = 0.3, color = "green", stat="probability", edgecolor='grey')
    sns.histplot(data = valid_trials_monkey[variable_of_interest], kde = False,  binwidth=0.05, alpha = 0.3, color = "blue", stat="probability", edgecolor='grey')
    axes.legend(labels=["Agent(LSTM)","Monkey"], fontsize = 14)
    # sns.kdeplot(data = valid_trials_agent, x = variable_of_interest, color = "green", bw=1)
    # sns.kdeplot(data = valid_trials_monkey, x = variable_of_interest, color = "blue", bw=1)
    axes.set_ylabel("Probability",fontsize=15)
    #axes.set_yticklabels("")
    plt.title("Abs Angle of Target Last Visible", fontsize=17)
    plt.xlabel("Angle (rad)", fontsize=15)
    axes.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    axes.tick_params(axis = "x", width=0)
    axes.set_xticks(np.arange(0.0, 0.9, 0.2))
    axes.set_xticklabels(np.arange(0.0, 0.9, 0.2).round(1))
    plt.xlim(0, 0.7)
    plt.show()
    plt.close()


    variable_of_interest = "t"
    fig, axes = plt.subplots(figsize=(4, 5))
    sns.histplot(data = valid_trials_agent[variable_of_interest], kde = False, binwidth=1,  alpha = 0.3, color = "green", stat="probability", edgecolor='grey')
    sns.histplot(data = valid_trials_monkey[variable_of_interest], kde = False, binwidth=1.1,  alpha = 0.3, color = "blue", stat="probability", edgecolor='grey')
    axes.legend(labels=["Agent(LSTM)","Monkey"], fontsize = 14)
    # sns.kdeplot(data = valid_trials_agent, x = variable_of_interest, color = "green", bw=1)
    # sns.kdeplot(data = valid_trials_monkey, x = variable_of_interest, color = "blue", bw=1)
    axes.set_ylabel("Probability",fontsize=15)
    #axes.set_yticklabels("")
    plt.title("Trial Duration", fontsize=17)
    plt.xlabel("Duration (s)", fontsize=15)
    axes.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    plt.show()
    plt.close()


    variable_of_interest = "num_stops"
    fig, axes = plt.subplots(figsize=(4, 5))
    sns.histplot(data = valid_trials_agent[variable_of_interest], binwidth=1, binrange=(0.5, 10.5), alpha = 0.3, color = "green", stat="probability", edgecolor='grey')
    sns.histplot(data = valid_trials_monkey[variable_of_interest], binwidth=1, binrange=(0.6, 10.6), alpha = 0.3, color = "blue", stat="probability", edgecolor='grey')
    axes.legend(labels=["Agent(LSTM)","Monkey"], fontsize = 14)
    # sns.kdeplot(data = valid_trials_agent, x = variable_of_interest,  bw=2, color = "green")
    # sns.kdeplot(data = valid_trials_monkey, x = variable_of_interest, bw=2, color = "blue")
    plt.xlabel("Number of Stops", fontsize=15)
    plt.xlim(0.7, 12)
    axes.set_ylabel("Probability",fontsize=15)
    axes.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    #axes.set_yticklabels("")
    plt.title("Number of Stops During Trials", fontsize=17)
    plt.show()
    plt.close()




def plot_merged_df_in_one_graph(merged_df):
    """
    Make a grouped barplot to visualize all categories in the merged_df (can be merged_stats_df or merged_medians_df)
    in one plot, comparing the monkey and the agent(s)

    Parameters
    ----------
    merged_df: dataframe
        containing various characteristics of each trial for both the monkey and the agent(s)


    """
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 8))
    # grouped barplot
    ax = sns.barplot(x="Category", y="Value", hue="Player", data=merged_df, ci=None);
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.show()
    plt.close()




def plot_merged_df_by_category(merged_df, percentage=True):
    """
    Make one barplot for each category in the merged_df (can be merged_stats_df or merged_medians_df), comparing the monkey and the agent(s)

    Parameters
    ----------
    merged_df: dataframe
        containing various characteristics of each trial for both the monkey and the agent(s)

    """

    for category in merged_df.Category.unique():
        category_df = merged_df[merged_df['Category']==category]
        sns.set(style="darkgrid")
        plt.figure(figsize=(4, 8))
        ## If wanting to differentiate the color by player:
        # ax = sns.barplot(x="Player", y="Value", hue="Player", data=category_df, ci=None);
        ax = sns.barplot(x="Player", y="Value", data=category_df, ci=None);
        ax.set_xlabel("")
        ax.set_ylabel("")
        if percentage == True:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.xticks(fontsize= 22)
        plt.yticks(fontsize= 15) 
        plt.title(str(category), fontsize= 22)
        plt.tight_layout()
        ## Other optional arguments:
        # ax.set_xticklabels("")
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        # ax.set_ylabel("Percentage of captured fireflies", fontsize = 22)
        plt.show()
        plt.close()





class HiddenPrints:
    """
    Hide all the printed statements while running the coded

    Parameters
    ----------
    merged_df: dataframe
        containing various characteristics of each trial for both the monkey and the agent(s)

    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout




def get_overall_lim(axes, axes2):
    """
    Get the x-limits and y-limits of the graphs based on both the monkey data and the agent data


    Parameters
    ----------
    axes: obj
        axes for one plot (e.g. for the monkey data)
    axes2: obj
        axes for another plot (e.g. for the agent data)


    Returns
    -------
    overall_xmin: num
        the minimum value of the x-axis that will be shared by both plots
    overall_xmax: num
        the maximum value of the x-axis that will be shared by both plots     
    overall_ymin: num
        the minimum value of the y-axis that will be shared by both plots
    overall_ymax: num
        the maximum value of the y-axis that will be shared by both plots

    """

    monkey_xmin, monkey_xmax = axes.get_xlim()
    monkey_ymin, monkey_ymax = axes.get_ylim()
    agent_xmin, agent_xmax = axes2.get_xlim()
    agent_ymin, agent_ymax = axes2.get_ylim()

    overall_xmin = min(monkey_xmin, agent_xmin)
    overall_xmax = max(monkey_xmax, agent_xmax)
    overall_ymin = min(monkey_ymin, agent_ymin)
    overall_ymax = max(monkey_ymax, agent_ymax)
    return overall_xmin, overall_xmax, overall_ymin, overall_ymax





def PlotSidebySide(graph_whole_duration,                   
                  info_of_monkey,
                  info_of_agent,
                  num_imitation_steps_monkey,
                  num_imitation_steps_agent,
                  currentTrial,
                  num_trials, 
                  rotation_matrix,
                  plotting_params = None
                  ):
    """
    Plot the monkey's graph and the agent's graph side by side


    Parameters
    ----------
    graph_whole_duration: list of 2 elements
        containing the start time and the end time in respect to the monkey data
    info_of_monkey: dict
        contains various important arrays, dataframes, or lists derived from the real monkey data
    info_of_agent: dict
        contains various important arrays, dataframes, or lists derived from the RL environmentthe and the agent's behaviours
    num_imitation_steps_monkey: num
        the number of steps used by the monkey for the part of the trajectory shared by the monkey and the agent (with the agent copying the monkey)
    num_imitation_steps_agent: num
        the number of steps used by the agent for the part of the trajectory shared by the monkey and the agent (with the agent copying the monkey)
    currentTrial: num
        the current trial to be plotted
    num_trials: num
        the number of trials (counting from the currentTrial into the past) to be plotted
    rotation_matrix: np.array
        to be used to rotate the graph when plotting
    plotting_params: dict, optional
        keyword arguments to be passed into the plotTrials function
    """  


    #===================================== Monkey =====================================
    fig = plt.figure()
    axes = fig.add_subplot(121)
    axes.set_title(f"Monkey: Trial {currentTrial}", fontsize = 22)

    PlotTrials(graph_whole_duration, 
            info_of_monkey['monkey_information'],
            info_of_monkey['ff_dataframe'], 
            info_of_monkey['ff_life_sorted'], 
            info_of_monkey['ff_real_position_sorted'], 
            info_of_monkey['ff_believed_position_sorted'], 
            info_of_monkey['ffs_around_target_indices'], 
            currentTrial = currentTrial,
            num_trials = num_trials,
            fig = fig, 
            axes = axes, 
            rotation_matrix = rotation_matrix,
            player = "monkey",
            steps_to_be_marked = num_imitation_steps_monkey,
            **plotting_params
            )
    

    #===================================== Agent =====================================

    axes2 = fig.add_subplot(122)
    axes2.set_title(f"Agent: Trial {currentTrial}", fontsize = 15)
    # Agent duration needs to start from 0, unlike the duration for the monkey, because the 
    # RL environment starts from 0 
    agent_duration = [0, graph_whole_duration[1]-graph_whole_duration[0]]      

    PlotTrials(agent_duration, 
              info_of_agent['monkey_information'],
              info_of_agent['ff_dataframe'], 
              info_of_agent['ff_life_sorted'], 
              info_of_agent['ff_real_position_sorted'], 
              info_of_agent['ff_believed_position_sorted'], 
              info_of_agent['ffs_around_target_indices'], 
              info_of_agent['ff_catched_T_sorted'], 
              currentTrial = None,
              num_trials = None,
              fig = fig, 
              axes = axes2, 
              rotation_matrix = rotation_matrix,
              player = "agent",
              steps_to_be_marked = num_imitation_steps_agent,
              **plotting_params
              )

    overall_xmin, overall_xmax, overall_ymin, overall_ymax = get_overall_lim(axes, axes2)
    plt.setp([axes, axes2], xlim=[overall_xmin, overall_xmax], ylim=[overall_ymin, overall_ymax])








def PlotPolar(duration,
              ff_dataframe, 
              rmax = 400,
              ff_catched_T_sorted = None,
              currentTrial = None, # Can be None; then it means all trials in the duration will be plotted
              num_trials = None,
              show_visible_ff = True,
              show_visible_target = True,
              show_ff_in_memory = False,
              show_target_in_memory = False,
              *args, 
              **kwargs
               ):
    """
    Plot the positions of the fireflies from the monkey's perspective (the monkey is always at the origin of the polar graph)


    Parameters
    ----------
    duration: list
        the duration to be plotted, with a starting time and an ending time; 
    ff_dataframe: pd.dataframe
        containing various information about all visible or "in-memory" fireflies at each time point
    rmax: num
        the radius of the polar graph
    currentTrial: numeric
        the number of current trial
    num_trials: numeric
        the number of trials to be plotted
    show_visible_ff: bool
        whether to show fireflies (other than the target) that are visible
    show_visible_target: bool
        whether to show the target when it is visible
    show_ff_in_memory: bool
        whether to show fireflies (other than the target) that are in memory
    show_target_in_memory: bool
        whether to show the target when it is in memory
    """  


    duration = [ff_catched_T_sorted[currentTrial-num_trials], ff_catched_T_sorted[currentTrial]]
    ff_info = ff_dataframe[(ff_dataframe["target_index"] == currentTrial) & (ff_dataframe["ff_index"] != currentTrial)]
    target_info = ff_dataframe[(ff_dataframe["target_index"] == currentTrial) & (ff_dataframe["ff_index"] == currentTrial)]
    colors_Reds = plt.get_cmap("Reds")(np.linspace(0, 1, 101))
    colors_Purples = plt.get_cmap("Purples")(np.linspace(0, 1, 101))
    
    if duration[1]-duration[0] > 0:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        ax.set_theta_zero_location("N")
        ax.set_rlabel_position(292.5)
        ax.set_ylim(0, rmax)
        # Draw the boundary of the monkey's vision (use width = np.pi*4/9 for 40 degrees of vision)
        ax.bar(0, rmax, width=np.pi/2, bottom=0.0, color="grey", alpha=0.1)
        plt.setp(ax, rorigin=0, rmin=0, rmax=rmax)   # rmax can be changed

        # Change the labels for the angles 
        labels = list(ax.get_xticks())
        labels[5], labels[6], labels[7] = -labels[3], -labels[2], -labels[1]
        labels_degrees = [str(int(math.degrees(label))) + chr(176) for label in labels]
        ax.set_xticklabels(labels_degrees)
        

        if show_ff_in_memory:
            ax.scatter(ff_info['ff_angle'], ff_info['ff_distance'], c=colors_Reds[np.array(ff_info['memory'].astype('int'))], s=15, alpha=0.8)
        elif show_visible_ff:
            ax.scatter(ff_info[(ff_info['visible'] == 1)]['ff_angle'], ff_info[(ff_info['visible'] == 1)]['ff_distance'], marker='.', s=20, alpha=1)

        if show_target_in_memory:
            ax.scatter(target_info['ff_angle'], target_info['ff_distance'], c=colors_Purples[target_info['memory'].astype('int')], s=15, alpha=0.8)     
        elif show_visible_target: 
            ax.scatter(target_info[(target_info['visible'] == 1)]['ff_angle'], target_info[(target_info['visible'] == 1)]['ff_distance'], marker='.', s=20, cmap='YlGn', alpha=1)


        # Add a legend
        colors = ['purple', 'red']
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
        labels = ['Captured firefly', 'Other fireflies']
        ax.legend(lines, labels, loc='lower right')


        # Add a colorbar
        cax = fig.add_axes([0.95, 0.05, 0.05, 0.4])
        norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
        # In this trial, the max speed is 199.99. In other trials, one might want to check with max(cum_speed)
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.Reds), cax=cax, orientation='vertical')
        cbar.ax.tick_params(axis='y', color='white', direction="in", right=True, length=5, width=1.5)
        cbar_labels = ['Visible 1.67s Ago', 'Visible 1.33s Ago', 'Visible 1s Ago', 'Visible 0.67s Ago',  'Visible 0.33s Ago', 'Visible']
        cbar.ax.set_yticklabels(cbar_labels)
        cbar.outline.remove()
        # cbar.ax.set_title('Visibility', ha='left', x=0, y=1.05)
        # cbar.ax.set_yticks([0,20,40,60,80,100])
        plt.show()


